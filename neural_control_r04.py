"""SC100K R04: recurrent PC and spiking control signals.

Host schedules fixed phases, extracts instruction fields and routes buses.
The PC is solely dual-rail state. Branch decision and next-PC selection
are computed by Q4 threshold circuits, not a host branch predicate.
"""
import json, random, struct, time
import numpy as np
from neural_alu_r01 import Circuit, MASK, ROOT
from neural_decoder_r02 import Decoder
from neural_register_r03 import RegisterCPU, encode, decode, fire, validate
from neural_rv32i_r01 import I,R,S,B,J,sext

class PC:
    neurons=960 # 320 data, 320 feedback, 320 readout
    def __init__(self):self.state=encode(0,32);self.commits=0
    def read(self):return decode(fire(7*self.state.sum(axis=-1,dtype=np.int32)))
    def commit(self,value,ablate=False):
        candidate=fire(7*encode(value&MASK,32).sum(axis=-1,dtype=np.int32)*(not ablate))
        validate(candidate)
        self.state=candidate;self.commits+=1

class Control:
    # 32 XOR bits, 31 reduction gates, predicate and enable rails,
    # 32-bit target mux with 3 gates per bit, 32-bit input buses.
    # Actual count below from circuit compiler.
    def __init__(self):
        c=Circuit()
        self.c=c
        self.inputs={}
        def port(name,n=1):
            p=[c.input(name+str(i)) for i in range(n)];self.inputs[name]=p;return p
        x=port('xor',32);s=port('signed')[0];u=port('unsigned')[0]
        enables={n:port(n)[0] for n in ('BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR')}
        seq=port('seq',32);target=port('target',32)
        def reduce_or(q):
            while len(q)>1:q=[c.OR(q[i],q[i+1]) if i+1<len(q) else q[i] for i in range(0,len(q),2)]
            return q[0]
        neq=reduce_or(x)
        cond=[c.AND(enables[n],p) for n,p in zip(('BEQ','BNE','BLT','BGE','BLTU','BGEU'),(c.neg(neq),neq,s,c.neg(s),u,c.neg(u)))]
        self.take=reduce_or(cond+[enables['JAL'],enables['JALR']])
        self.outputs=[c.mux(self.take,a,b) for a,b in zip(seq,target)]
        e=np.array(c.edges,dtype=np.int32)
        depth=np.array([c.level[int(i)] for i in e[:,1]])
        self.stages=[]
        for d in sorted(set(depth)):
            sub=e[depth==d];targets,inv=np.unique(sub[:,1],return_inverse=True)
            self.stages.append((sub[:,0],targets,inv,sub[:,2]))
        self.neurons=c.n;self.synapses=len(e);self.calls=0
    def route(self,name,seq,target,xor=0,signed=0,unsigned=0,ablate=False):
        vals={'seq':seq,'target':target,'xor':xor,'signed':signed,'unsigned':unsigned}
        spikes=np.zeros(self.neurons,dtype=np.int8)
        for key,ports in self.inputs.items():
            value=vals.get(key,int(key==name))
            for i,p in enumerate(ports):spikes[list(p[(value>>i)&1])]=1
        for src,targets,inv,w in self.stages:
            syn=np.bincount(inv,weights=spikes[src]*w*(not ablate),minlength=len(targets)).astype(np.int32)
            spikes[targets]=(np.clip(syn//3,-32,31)>=11)
        def read(p):
            lo,hi=(int(spikes[list(r)].sum()) for r in p)
            if (lo,hi) not in ((5,0),(0,5)):raise RuntimeError('invalid control rails')
            return int(hi==5)
        taken=read(self.take)
        value=sum(read(p)<<i for i,p in enumerate(self.outputs))
        self.calls+=1
        return value,taken

class ControlCPU(RegisterCPU):
    def __init__(self,circuit,decoder,image):
        self.counter=PC()
        super().__init__(circuit,decoder,image)
        self.control=Control()
    @property
    def pc(self):return self.counter.read()
    @pc.setter
    def pc(self,v):self.counter.commit(v)
    def extension(self,name,a,b):
        raise RuntimeError("unrouted neural instruction")
    def prepare_extension(self,name,a,b):
        return self.extension(name,a,b),None
    def finish_extension(self,name,a,b):
        pass
    def prepare_instruction_extension(self,name,a,b,ins):
        return self.prepare_extension(name,a,b)
    def finish_instruction_extension(self,name,a,b,ins):
        self.finish_extension(name,a,b)
    def fetch(self,pc):
        return self.load(pc,4)
    def illegal(self,ins):
        raise RuntimeError("illegal instruction")
    def check_next_pc(self,nxt):
        if nxt&3:raise RuntimeError("instruction address misaligned")
    def special_step(self,name,ins,pc):
        return False
    def step(self):
        if self.halted:raise RuntimeError('already halted')
        pc=self.pc;ins=self.fetch(pc);name=self.decoder.decode(ins);self.decode_calls+=1
        if name is None:self.illegal(ins)
        if self.special_step(name,ins,pc):return
        rd=(ins>>7)&31;r1=(ins>>15)&31;r2=(ins>>20)&31
        a=self.reg[r1];b=self.reg[r2];imm=sext(ins>>20,12)
        seq=self.alu('ADD',pc,4);target=seq
        x=s=u=0;pending=None;store=None;halt=False
        if name in ('ADD','SUB','SLL','SLT','SLTU','XOR','SRL','SRA','OR','AND'):pending=(rd,self.alu(name,a,b))
        elif name in ('ADDI','SLTI','SLTIU','XORI','ORI','ANDI','SLLI','SRLI','SRAI'):pending=(rd,self.alu(name[:-1],a,r2 if name in ('SLLI','SRLI','SRAI') else imm))
        elif name in ('LB','LH','LW','LBU','LHU'):
            n,sgn={'LB':(1,True),'LH':(2,True),'LW':(4,False),'LBU':(1,False),'LHU':(2,False)}[name]
            pending=(rd,self.load(self.alu('ADD',a,imm),n,sgn))
        elif name in ('SB','SH','SW'):
            off=sext(((ins>>25)<<5)|rd,12)
            store=(self.alu('ADD',a,off),{'SB':1,'SH':2,'SW':4}[name],b)
        elif name in ('BEQ','BNE','BLT','BGE','BLTU','BGEU'):
            off=sext(((ins>>31)<<12)|(((ins>>7)&1)<<11)|(((ins>>25)&63)<<5)|(((ins>>8)&15)<<1),13)
            target=self.alu('ADD',pc,off)
            if name in ('BEQ','BNE'):x=self.alu('XOR',a,b)
            elif name in ('BLT','BGE'):s=self.alu('SLT',a,b)
            else:u=self.alu('SLTU',a,b)
        elif name=='LUI':pending=(rd,ins&0xfffff000)
        elif name=='AUIPC':pending=(rd,self.alu('ADD',pc,ins&0xfffff000))
        elif name=='JAL':
            off=sext(((ins>>31)<<20)|(((ins>>12)&255)<<12)|(((ins>>20)&1)<<11)|(((ins>>21)&1023)<<1),21)
            pending=(rd,seq);target=self.alu('ADD',pc,off)
        elif name=='JALR':
            target=self.alu('AND',self.alu('ADD',a,imm),0xfffffffe);pending=(rd,seq)
        elif name=='FENCE':pass
        elif name=='EBREAK':halt=True
        else:
            result,store=self.prepare_instruction_extension(name,a,b,ins);pending=(rd,result)
        nxt,taken=self.control.route(name,seq,target,x,s,u)
        # IALIGN=32; detect target fault before changing registers or memory.
        self.check_next_pc(nxt)
        if store:self.store(*store)
        if pending:self.rd(*pending)
        self.pc=nxt;self.steps+=1;self.halted=halt
        self.finish_instruction_extension(name,a,b,ins)

def main():
    ctrl=Control();rng=random.Random(3110);checks=0;t=time.perf_counter()
    for name in ('ADD','BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR'):
        for i in range(128):
            x=0 if i%2 else rng.getrandbits(32)|1;s=rng.randrange(2);u=rng.randrange(2)
            seq=rng.getrandbits(32);target=rng.getrandbits(32)
            want={'BEQ':x==0,'BNE':x!=0,'BLT':bool(s),'BGE':not s,'BLTU':bool(u),'BGEU':not u,'JAL':True,'JALR':True}.get(name,False)
            assert ctrl.route(name,seq,target,x,s,u)==(target if want else seq,int(want))
            checks+=1
    pc=PC()
    for i in range(256):
        v=rng.getrandbits(32);pc.commit(v);assert pc.read()==v
    old=pc.state.copy()
    for f in (lambda:pc.commit(42,True),lambda:ctrl.route('JAL',4,8,ablate=True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation failed')
    assert np.array_equal(old,pc.state)
    test_seconds=time.perf_counter()-t
    words=[I(0x13,1,0,0,1),I(0x13,2,0,0,17),I(0x13,3,0,0,0),
        R(3,0,3,1),I(0x13,1,0,1,1),B(1,1,2,-8),
        I(0x13,4,0,0,1024),S(2,4,3,0),I(0x03,5,2,4,0),
        J(6,20),I(0x13,8,0,0,-1),S(0,4,8,4),I(0x03,9,0,4,4),
        J(0,12),I(0x13,7,0,5,1),I(0x67,0,0,6,0),
        B(1,3,5,8),I(0x13,0,0,0,123),0x00100073]
    c=Circuit().compile();d=Decoder().compile()
    def cpu(words):return ControlCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    m=cpu(words);t=time.perf_counter()
    while not m.halted and m.steps<1000:m.step()
    elapsed=time.perf_counter()-t
    assert m.halted and m.steps==m.control.calls==m.decode_calls==64 and m.counter.commits==65
    assert m.reg[3]==m.reg[5]==136 and m.reg[7]==137 and m.reg[9]==MASK and m.reg[0]==0
    assert m.load(1024,4)==136
    fault=cpu([J(5,2)])
    try:fault.step()
    except RuntimeError as e:assert 'misaligned' in str(e)
    else:raise AssertionError('misaligned jump accepted')
    assert fault.pc==0 and fault.reg[5]==0 and fault.steps==0
    # Untaken misaligned branch must not raise.
    untaken=cpu([B(1,0,0,2),0x00100073]);untaken.step();assert untaken.pc==4
    total=c.n+d.n+m.reg.neurons+ctrl.neurons+PC.neurons
    assert total<=131072
    report=dict(status='PASS',control_vectors=checks,pc_vectors=256,errors=0,
        ablation_rejected=True,misalignment_cases=2,control_neurons=ctrl.neurons,
        control_synapses=ctrl.synapses,pc_neurons=PC.neurons,total_logical_neurons=total,
        test_seconds=round(test_seconds,3),guest_instructions=m.steps,guest_ips=round(m.steps/elapsed,2),
        neural_alu_calls=m.alu_calls,neural_control_calls=m.control.calls,neural_pc_commits=m.counter.commits-1,
        linux_booted=False,limitations='Host extracts fields, dispatches instruction classes, gates bus operations and handles traps. Fixed separate circuits, not continuous GENESIS. No privileged ISA or Linux yet.')
    out=ROOT/'control-r04';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

