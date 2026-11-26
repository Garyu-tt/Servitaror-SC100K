"""SC100K R02: dual-rail instruction classifier using GENESIS Q4 thresholds."""
import json, random, time, struct
import numpy as np
from neural_alu_r01 import Circuit, MASK, ROOT
from neural_rv32i_r01 import CPU, I, R, S, B, J
# mask/value table for the implemented subset, not used as a host runtime decoder.
PATTERNS = []
def add(name, mask, value): PATTERNS.append((name,mask,value))
for f,name in enumerate(('ADD','SLL','SLT','SLTU','XOR','SRL','OR','AND')):
    add(name,0xfe00707f,0x33|(f<<12))
for f,name in ((0,'SUB'),(5,'SRA')):add(name,0xfe00707f,0x40000033|(f<<12))
for f,name in enumerate(('ADDI','SLLI','SLTI','SLTIU','XORI','SRLI','ORI','ANDI')):
    add(name,0xfe00707f if f in (1,5) else 0x707f,0x13|(f<<12))
add('SRAI',0xfe00707f,0x40005013)
for f,name in ((0,'LB'),(1,'LH'),(2,'LW'),(4,'LBU'),(5,'LHU')):add(name,0x707f,3|(f<<12))
for f,name in enumerate(('SB','SH','SW')):add(name,0x707f,0x23|(f<<12))
for f,name in ((0,'BEQ'),(1,'BNE'),(4,'BLT'),(5,'BGE'),(6,'BLTU'),(7,'BGEU')):add(name,0x707f,0x63|(f<<12))
for op,name in ((0x37,'LUI'),(0x17,'AUIPC'),(0x6f,'JAL')):add(name,0x7f,op)
add('JALR',0x707f,0x67)
# FENCE requires funct3=0; reserved fields are treated as hints by base ISA.
add('FENCE',0x707f,0x0f)
add('EBREAK',MASK,0x00100073)
class Decoder(Circuit):
    def compile(self):
        self.bits=[self.input(str(i)) for i in range(32)]
        self.outputs=[]
        for name,mask,value in PATTERNS:
            q=[self.bits[i] if value>>i&1 else self.neg(self.bits[i]) for i in range(32) if mask>>i&1]
            while len(q)>1:q=[self.AND(q[i],q[i+1]) if i+1<len(q) else q[i] for i in range(0,len(q),2)]
            self.outputs.append(q[0])
        e=np.array(self.edges,dtype=np.int32)
        self.stages=[]
        self.packed=__import__('q4').pack_q4(e[:,2].astype(np.int8))
        depths=np.array([self.level[int(x)] for x in e[:,1]])
        for d in sorted(set(depths)):
            sub=e[depths==d]; targets,inv=np.unique(sub[:,1],return_inverse=True)
            self.stages.append((sub[:,0],targets,inv,sub[:,2]))
        return self
    def decode(self,word,ablate=False):
        spike=np.zeros(self.n,dtype=np.int8)
        for i,p in enumerate(self.bits):spike[list(p[(word>>i)&1])]=1
        for src,targets,inv,w in self.stages:
            syn=np.bincount(inv,weights=spike[src]*w*(not ablate),minlength=len(targets)).astype(np.int32)
            spike[targets]=(np.clip(syn//3,-32,31)>=11)
        matches=[]
        for (name,_,_),p in zip(PATTERNS,self.outputs):
            lo,hi=(int(spike[list(r)].sum()) for r in p)
            if (lo,hi) not in ((5,0),(0,5)):raise RuntimeError('invalid neural decode rails')
            if hi:matches.append(name)
        if len(matches)>1:raise RuntimeError('ambiguous neural decode')
        return matches[0] if matches else None
class NeuralCPU(CPU):
    def __init__(self,circuit,decoder,image):
        super().__init__(circuit,image);self.decoder=decoder;self.decode_calls=0
    def step(self):
        if self.halted:raise RuntimeError('already halted')
        ins=self.load(self.pc,4);name=self.decoder.decode(ins);self.decode_calls+=1
        if name is None:raise RuntimeError('illegal instruction')
        # Field extraction and routing remain host functions.
        rd=(ins>>7)&31;r1=(ins>>15)&31;r2=(ins>>20)&31
        from neural_rv32i_r01 import sext
        a=self.reg[r1];b=self.reg[r2];imm=sext(ins>>20,12)
        pc=self.pc;nxt=self.alu('ADD',pc,4)
        if name in ('ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND'):self.rd(rd,self.alu(name,a,b))
        elif name in ('ADDI','SLTI','SLTIU','XORI','ORI','ANDI','SLLI','SRLI','SRAI'):
            self.rd(rd,self.alu(name[:-1],a,r2 if name in ('SLLI','SRLI','SRAI') else imm))
        elif name in ('LB','LH','LW','LBU','LHU'):
            n,sgn={'LB':(1,True),'LH':(2,True),'LW':(4,False),'LBU':(1,False),'LHU':(2,False)}[name]
            self.rd(rd,self.load(self.alu('ADD',a,imm),n,sgn))
        elif name in ('SB','SH','SW'):
            off=sext(((ins>>25)<<5)|rd,12)
            self.store(self.alu('ADD',a,off),{'SB':1,'SH':2,'SW':4}[name],b)
        elif name in ('BEQ','BNE','BLT','BGE','BLTU','BGEU'):
            off=sext(((ins>>31)<<12)|(((ins>>7)&1)<<11)|(((ins>>25)&63)<<5)|(((ins>>8)&15)<<1),13)
            if name in ('BEQ','BNE'):take=(self.alu('XOR',a,b)==0)==(name=='BEQ')
            else:take=bool(self.alu('SLTU' if name.endswith('U') else 'SLT',a,b))==(name.startswith('BLT'))
            if take:nxt=self.alu('ADD',pc,off)
        elif name=='LUI':self.rd(rd,ins&0xfffff000)
        elif name=='AUIPC':self.rd(rd,self.alu('ADD',pc,ins&0xfffff000))
        elif name=='JAL':
            off=sext(((ins>>31)<<20)|(((ins>>12)&255)<<12)|(((ins>>20)&1)<<11)|(((ins>>21)&1023)<<1),21)
            self.rd(rd,nxt);nxt=self.alu('ADD',pc,off)
        elif name=='JALR':
            target=self.alu('AND',self.alu('ADD',a,imm),0xfffffffe);self.rd(rd,nxt);nxt=target
        elif name=='FENCE':pass
        elif name=='EBREAK':self.halted=True
        else:raise RuntimeError('unrouted neural instruction')
        self.pc=nxt;self.steps+=1
def main():
    d=Decoder().compile(); rng=random.Random(3108)
    vectors=[rng.getrandbits(32) for _ in range(2048)]
    vectors += [value|(rng.getrandbits(32)&(~mask&MASK)) for name,mask,value in PATTERNS for _ in range(32)]
    legal=0
    t=time.perf_counter()
    for word in vectors:
        expected=[name for name,mask,value in PATTERNS if word&mask==value]
        assert len(expected)<=1
        assert d.decode(word)==(expected[0] if expected else None),hex(word)
        legal+=bool(expected)
    elapsed=time.perf_counter()-t
    try:d.decode(0x00100073,ablate=True)
    except RuntimeError:pass
    else:raise AssertionError('ablation failed')
    words=[I(0x13,1,0,0,1),I(0x13,2,0,0,17),I(0x13,3,0,0,0),
        R(3,0,3,1),I(0x13,1,0,1,1),B(1,1,2,-8),
        I(0x13,4,0,0,1024),S(2,4,3,0),I(0x03,5,2,4,0),
        J(6,20),I(0x13,8,0,0,-1),S(0,4,8,4),I(0x03,9,0,4,4),
        J(0,12),I(0x13,7,0,5,1),I(0x67,0,0,6,0),
        B(1,3,5,8),I(0x13,0,0,0,123),0x00100073]
    c=Circuit().compile();cpu=NeuralCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    start=time.perf_counter()
    while not cpu.halted and cpu.steps<1000:cpu.step()
    seconds=time.perf_counter()-start
    assert cpu.halted and cpu.steps==cpu.decode_calls==64 and cpu.reg[3]==cpu.reg[5]==136 and cpu.reg[7]==137 and cpu.reg[9]==MASK
    assert c.n+d.n<=131072
    report=dict(status='PASS',patterns=len(PATTERNS),vectors=len(vectors),legal=legal,illegal=len(vectors)-legal,decode_errors=0,ablation_rejected=True,decoder_neurons=d.n,decoder_synapses=len(d.packed)*2,total_circuit_neurons=c.n+d.n,decode_per_sec=round(len(vectors)/elapsed,2),guest_instructions=cpu.steps,guest_ips=round(cpu.steps/seconds,2),neural_decode_calls=cpu.decode_calls,neural_alu_calls=cpu.alu_calls,linux_booted=False,limitations='Host field extraction, dispatch, register storage, memory and exception handling. Decoder is fixed wiring, not learned. Mask-table oracle shares pattern specification; not independent ISA conformance.')
    out=ROOT/'decoder-r02';out.mkdir(exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

