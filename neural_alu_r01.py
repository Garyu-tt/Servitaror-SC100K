#!/usr/bin/env python3
"""Cogitator R01: clocked Q4 spiking ALU derived from GENESIS R02.
Host encodes inputs, schedules delayed spikes and decodes output rails.
No host arithmetic implements any guest ALU result. Oracle is test-only.
"""
import json, time, hashlib, random
from pathlib import Path
import numpy as np
import q4 as g
ROOT=Path(__file__).resolve().parent
BASE=ROOT/'brain-131072.npz'
OUT=ROOT/'alu-r01'
CELLS=131072
OPS=('ADD','SUB','AND','OR','XOR','SLL','SRL','SRA','SLT','SLTU')
MASK=(1<<32)-1
class Circuit:
    def __init__(self):
        self.n=0; self.level={}; self.edges=[]; self.inputs={}
    def alloc(self,depth):
        s=self.n; self.n+=10
        if self.n>CELLS: raise ValueError('cortex capacity')
        p=(tuple(range(s,s+5)),tuple(range(s+5,s+10)))
        for rail in p:
            for i in rail:self.level[i]=depth
        return p
    def input(self,name):
        p=self.alloc(0); self.inputs[name]=p; return p
    def neg(self,a):return a[::-1]
    def gate(self,op,a,b):
        d=max(self.level[a[0][0]],self.level[b[0][0]])+1
        o=self.alloc(d)
        # rail 0 false, rail 1 true; AND needs two 20-current bundles,
        # OR needs either 35-current bundle. R02 threshold is 11 after /3.
        for r in (0,1):
            w=4 if (op=='AND' and r==1) or (op=='OR' and r==0) else 7
            for inp in (a,b):
                self.edges.extend((s,t,w) for s in inp[r] for t in o[r])
        return o
    def AND(self,a,b):return self.gate('AND',a,b)
    def OR(self,a,b):return self.gate('OR',a,b)
    def XOR(self,a,b):return self.OR(self.AND(a,self.neg(b)),self.AND(self.neg(a),b))
    def mux(self,s,a,b):return self.OR(self.AND(self.neg(s),a),self.AND(s,b))
    def add(self,a,b,carry):
        out=[]
        for x,y in zip(a,b):
            p=self.XOR(x,y);out.append(self.XOR(p,carry))
            carry=self.OR(self.AND(x,y),self.AND(p,carry))
        return out,carry
    def shift(self,a,b,zero,right=False,signed=False):
        x=list(a)
        for k in range(5):
            n=1<<k;fill=a[-1] if signed else zero
            y=(x[n:]+[fill]*n) if right else ([zero]*n+x[:-n])
            x=[self.mux(b[k],u,v) for u,v in zip(x,y)]
        return x
    def compile(self):
        a=[self.input('a'+str(i)) for i in range(32)]
        b=[self.input('b'+str(i)) for i in range(32)]
        z=self.input('zero');sels=[self.input(o) for o in OPS]
        add,_=self.add(a,b,z); sub,carry=self.add(a,[self.neg(x) for x in b],self.neg(z))
        slt=self.mux(self.XOR(a[-1],b[-1]),sub[-1],a[-1])
        words=[add,sub,[self.AND(x,y) for x,y in zip(a,b)],
               [self.OR(x,y) for x,y in zip(a,b)],[self.XOR(x,y) for x,y in zip(a,b)],
               self.shift(a,b,z),self.shift(a,b,z,True),self.shift(a,b,z,True,True),
               [slt]+[z]*31,[self.neg(carry)]+[z]*31]
        self.outputs=[]
        for i in range(32):
            q=[self.AND(s,w[i]) for s,w in zip(sels,words)]
            while len(q)>1:q=[self.OR(q[j],q[j+1]) if j+1<len(q) else q[j] for j in range(0,len(q),2)]
            self.outputs.append(q[0])
        # Each synaptic source spike is retained until its scheduled arrival.
        # This is explicit delay scheduling, not a Boolean gate in host code.
        e=np.array(self.edges,dtype=np.int32)
        order=np.argsort(np.array([self.level[int(x)] for x in e[:,1]]),kind='stable')
        e=e[order]; self.src=e[:,0]; self.dst=e[:,1]
        weights=e[:,2].astype(np.int8)
        self.packed=g.pack_q4(weights)
        assert np.array_equal(g.unpack_q4(self.packed),weights)
        self.stages=[]
        for d in range(1,max(self.level.values())+1):
            idx=np.flatnonzero(np.array([self.level[int(x)] for x in self.dst])==d)
            if not len(idx):continue
            targets,inv=np.unique(self.dst[idx],return_inverse=True)
            self.stages.append((self.src[idx],targets,inv,weights[idx]))
        return self
    def run(self,op,a,b,ablate=False):
        if op not in OPS:raise ValueError(f"unknown ALU operation: {op}")
        spike=np.zeros(self.n,dtype=np.int8)
        for name,p in self.inputs.items():
            value=0 if name=='zero' else ((a>>int(name[1:]))&1) if name.startswith('a') else ((b>>int(name[1:]))&1) if name.startswith('b') else int(name==op)
            spike[list(p[value])]=1
        for src,targets,inv,w in self.stages:
            syn=np.bincount(inv,weights=spike[src]*w*(not ablate),minlength=len(targets)).astype(np.int32)
            # Original R02 dynamics: initial potential zero, synaptic /3,
            # clip to [-32,31], threshold 11, reset -5 on spike.
            vv=np.clip(syn//3,-32,31)
            spike[targets]=(vv>=11)
        result=0
        for i,p in enumerate(self.outputs):
            lo=int(spike[list(p[0])].sum());hi=int(spike[list(p[1])].sum())
            if (lo,hi) not in ((5,0),(0,5)):raise RuntimeError('invalid dual-rail readout')
            result|=int(hi==5)<<i
        return result,int(spike.sum())
def oracle(op,a,b):
    s=lambda x:x if x<1<<31 else x-(1<<32)
    return {'ADD':lambda:a+b,'SUB':lambda:a-b,'AND':lambda:a&b,'OR':lambda:a|b,'XOR':lambda:a^b,
            'SLL':lambda:a<<(b&31),'SRL':lambda:a>>(b&31),'SRA':lambda:s(a)>>(b&31),
            'SLT':lambda:int(s(a)<s(b)),'SLTU':lambda:int(a<b)}[op]()&MASK
def main():
    before=g.sha(BASE) if BASE.is_file() else None
    if before is not None:
        with np.load(BASE,allow_pickle=False) as cp:
            assert len(cp['v'])==CELLS
            assert len(cp['packed'])==CELLS*g.OUTDEGREE//2
    t=time.perf_counter(); c=Circuit().compile();build=time.perf_counter()-t
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT/'cortex.npz',src=c.src,dst=c.dst,packed=c.packed,
                        levels=np.array([c.level[i] for i in range(c.n)],dtype=np.int16),
                        input_rails=np.array(list(c.inputs.values())),output_rails=np.array(c.outputs))
    rng=random.Random(3107)
    values=[0,1,2,31,32,0x7fffffff,0x80000000,0xffffffff,0xaaaaaaaa,0x55555555]
    pairs=[(a,b) for a in values for b in values]+[(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(64)]
    t=time.perf_counter();count=0;spikes=0
    for op in OPS:
        for a,b in pairs:
            got,n=c.run(op,a,b); want=oracle(op,a,b)
            assert got==want,(op,a,b,got,want)
            count+=1;spikes+=n
    elapsed=time.perf_counter()-t
    try:c.run('ADD',1,2,ablate=True)
    except RuntimeError:ablation=True
    else:ablation=False
    assert ablation and (before is None or g.sha(BASE)==before)
    report=dict(status='PASS',base_checkpoint_sha256=before,brain_capacity=CELLS,
        active_circuit_neurons=c.n,synapses=len(c.src),stages=len(c.stages),
        operations=list(OPS),test_vectors=count,errors=0,ablation_rejected=ablation,
        build_seconds=round(build,3),test_seconds=round(elapsed,3),
        operations_per_second=round(count/elapsed,2),spikes=spikes,
        cortex_sha256=g.sha(OUT/'cortex.npz'),
        limitations='Engineered feedforward Q4 cortical overlay; background not stepped; no learning, ISA decoder, registers, MMU or kernel yet.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()

