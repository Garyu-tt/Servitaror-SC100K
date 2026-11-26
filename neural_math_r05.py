"""SC100K R05: RV32M microsequencer over dual-rail GENESIS-Q4 gates.

Host schedules fixed rounds and wire permutations. All guest arithmetic,
carry, comparison and sign selection are computed by threshold gates.
No guest integer multiply/divide or data-dependent arithmetic fallback.
"""
import json, random, struct, time
import numpy as np
from neural_alu_r01 import Circuit, MASK, ROOT
from neural_decoder_r02 import Decoder, PATTERNS
from neural_control_r04 import ControlCPU
from neural_register_r03 import encode, decode, validate, fire
from neural_rv32i_r01 import I,R

MOPS=('MUL','MULH','MULHSU','MULHU','DIV','DIVU','REM','REMU')
class MDecoder(Decoder):
    patterns=tuple(PATTERNS)+tuple((n,0xfe00707f,0x02000033|(f<<12)) for f,n in enumerate(MOPS))

class Math:
    # One reusable 64-bit Kogge-Stone lane plus input, carry and output
    # storage. Conservative logical-cell budget for fixed gates (see docs).
    neurons=30000
    def __init__(self):
        self.gates=0;self.rounds=0;self.calls=0;self.ablate=False
    def gate(self,a,b,op):
        a,b=np.broadcast_arrays(a,b)
        assert a.shape[-2:]==(2,5)
        v=a.sum(axis=-1,dtype=np.int32)+b.sum(axis=-1,dtype=np.int32)
        w=np.array([7,4] if op=='and' else [4,7],dtype=np.int32)
        self.gates+=v.size//2
        return fire(v*w*(not self.ablate))
    def AND(self,a,b):return self.gate(a,b,'and')
    def OR(self,a,b):return self.gate(a,b,'or')
    @staticmethod
    def NOT(a):return a[...,::-1,:]
    def XOR(self,a,b):return self.OR(self.AND(a,self.NOT(b)),self.AND(self.NOT(a),b))
    def mux(self,s,a,b):return self.OR(self.AND(self.NOT(s),a),self.AND(s,b))
    def add(self,a,b,cin):
        # Parallel-prefix carry circuit. Each slice is fixed wiring.
        n=len(a);p=self.XOR(a,b);g=self.AND(a,b);prop=p
        d=1
        while d<n:
            g=np.concatenate((g[:d],self.OR(g[d:],self.AND(prop[d:],g[:-d]))))
            prop=np.concatenate((prop[:d],self.AND(prop[d:],prop[:-d])))
            d*=2
        carries=self.OR(g,self.AND(prop,cin))
        out=self.XOR(p,np.concatenate((cin[None,...],carries[:-1])))
        return out,carries[-1]
    def sub(self,a,b):
        return self.add(a,self.NOT(b),encode(1,1)[0])
    def neg(self,a):
        return self.sub(encode(0,len(a)),a)[0]
    def nonzero(self,a):
        q=a
        while len(q)>1:
            end=len(q)//2*2
            q=np.concatenate((self.OR(q[:end:2],q[1:end:2]),q[end:]))
        return q[0]
    def run(self,op,a,b,ablate=False):
        if op not in MOPS:raise ValueError('unsupported M operation')
        self.ablate=ablate
        aa=encode(a&MASK,32);bb=encode(b&MASK,32)
        zero=encode(0,1)[0]
        if op in MOPS[:4]:
            # Signed magnitudes use neural sign selection, then a 64-bit
            # product, followed by neural two's-complement correction.
            sa=aa[-1] if op in ('MULH','MULHSU') else zero
            sb=bb[-1] if op=='MULH' else zero
            aabs=self.mux(sa,aa,self.neg(aa));babs=self.mux(sb,bb,self.neg(bb))
            term=np.concatenate((aabs,encode(0,32)))
            acc=encode(0,64)
            for i in range(32):
                acc,_=self.add(acc,self.AND(babs[i],term),zero)
                term=np.concatenate((zero[None,...],term[:-1]))
                self.rounds+=1
            acc=self.mux(self.XOR(sa,sb),acc,self.neg(acc))
            result=acc[:32] if op=='MUL' else acc[32:]
        else:
            signed=op in ('DIV','REM')
            sa=aa[-1] if signed else zero;sb=bb[-1] if signed else zero
            aabs=self.mux(sa,aa,self.neg(aa));babs=self.mux(sb,bb,self.neg(bb))
            rem=encode(0,33);divisor=np.concatenate((babs,zero[None,...]))
            quotient=[]
            for i in range(31,-1,-1):
                shifted=np.concatenate((aabs[i:i+1],rem[:-1]))
                difference,ge=self.sub(shifted,divisor)
                rem=self.mux(ge,shifted,difference)
                quotient.append(ge)
                self.rounds+=1
            q=np.stack(quotient[::-1]);r=rem[:32]
            # RISC-V DIV by zero yields all ones even for negative dividend;
            # REM returns the original dividend. Overflow naturally wraps.
            qsign=self.AND(self.XOR(sa,sb),self.nonzero(bb))
            q=self.mux(qsign,q,self.neg(q));r=self.mux(sa,r,self.neg(r))
            result=q if op in ('DIV','DIVU') else r
        value=decode(result)
        self.calls+=1
        return value

class MCPU(ControlCPU):
    def __init__(self,circuit,decoder,image):
        super().__init__(circuit,decoder,image);self.math=Math()
    def extension(self,name,a,b):
        if name not in MOPS: return super().extension(name,a,b)
        return self.math.run(name,a,b)

def oracle(op,a,b):
    s=lambda x:x if x<1<<31 else x-(1<<32)
    if op=='MUL':return a*b&MASK
    if op=='MULH':return (s(a)*s(b)>>32)&MASK
    if op=='MULHSU':return (s(a)*b>>32)&MASK
    if op=='MULHU':return (a*b>>32)&MASK
    if b==0:return MASK if op in ('DIV','DIVU') else a
    if op in ('DIVU','REMU'):return (a//b if op=='DIVU' else a%b)&MASK
    x,y=s(a),s(b)
    q=abs(x)//abs(y)
    if (x<0)!=(y<0):q=-q
    return (q if op=='DIV' else x-q*y)&MASK

def main():
    m=Math();rng=random.Random(3111)
    pairs=[(0,0),(1,0),(MASK,0),(0x80000000,0),(0x80000000,MASK),
           (MASK,MASK),(0x7fffffff,0x80000000),(0x80000000,1),
           (0x12345678,0x9abcdef0),(123,7),(MASK,2),(0,17)]
    pairs += [(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(8)]
    t=time.perf_counter();checks=0
    for op in MOPS:
        for a,b in pairs:
            got=m.run(op,a,b);want=oracle(op,a,b)
            assert got==want,(op,hex(a),hex(b),hex(got),hex(want))
            checks+=1
    elapsed=time.perf_counter()-t
    try:m.run('MUL',3,5,True)
    except RuntimeError:pass
    else:raise AssertionError('ablation failed')
    d=MDecoder().compile();c=Circuit().compile()
    # real guest program: signed and unsigned operations through neural decode,
    # register bank, PC and control. Read back all eight result registers.
    words=[I(0x13,1,0,0,-13),I(0x13,2,0,0,5)]
    words.extend(R(3+i,i,1,2,1) for i in range(8))
    words.append(0x00100073)
    cpu=MCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    start=time.perf_counter()
    while not cpu.halted and cpu.steps<100:cpu.step()
    guest_s=time.perf_counter()-start
    assert cpu.halted and cpu.steps==11 and cpu.math.calls==8
    for i,op in enumerate(MOPS):assert cpu.reg[3+i]==oracle(op,(-13)&MASK,5)
    total=c.n+d.n+cpu.reg.neurons+cpu.control.neurons+cpu.counter.neurons+m.neurons
    assert total<=131072
    out=ROOT/'math-r05';out.mkdir(exist_ok=True)
    report=dict(status='PASS',operations=list(MOPS),arithmetic_vectors=checks,errors=0,
        seconds=round(elapsed,3),operations_per_sec=round(checks/elapsed,2),
        fixed_rounds=32,ablation_rejected=True,guest_instructions=cpu.steps,
        guest_m_ops=cpu.math.calls,guest_seconds=round(guest_s,3),
        decoder_neurons=d.n,math_budget_cells=m.neurons,total_budget_cells=total,
        gate_evaluations=m.gates,linux_booted=False,
        limitations='Vectorized fixed threshold microsequencer, estimated reusable lane capacity, not an enumerated integrated GENESIS graph. Host schedules phases and wire permutations. No RV32A, privileged ISA or Linux.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

