"""R24: packed native synaptic boundary; exact threshold truth tables.

The R13 proof validates five-cell replicas and static synaptic topology.
Only the representation at graph boundaries changes. Every internal
representative is evaluated, including inactive operation paths.
"""
import argparse,ctypes,hashlib,itertools,json,random,subprocess,time
from pathlib import Path
from native_bundle import native_library
import numpy as np
from neural_resume_r13 import Quotient,snapshot,restore,save,advance
from neural_alu_r01 import OPS,oracle
from neural_linux_r12 import fixture,load_image
from neural_recurrent_r15 import Engine
from neural_tlb_r17 import PageCache
from neural_math_r05 import oracle as moracle
import neural_native_r14 as r14

class Packed:
    def __init__(self,n,edges,level,ports,outputs,args):
        verified=Quotient(n,edges,level,ports,outputs)
        groups=[{} for _ in range(n//5)]
        for s,t,w in edges:
            if t%5==0:
                q=groups[t//5];q[s//5]=q.get(s//5,0)+w
        self.nin=len(ports);self.nout=len(outputs)
        self.niw=(self.nin+31)//32;self.now=(self.nout+31)//32
        lines=['typedef unsigned int u32;',
               'int evaluate(const u32 *in,int en,u32 *out){',
               f'unsigned char r[{n//5}];int count=0;']
        for i,p in enumerate(ports):
            lines.append(f'r[{p[1][0]//5}]=(in[{i//32}]>>{i%32})&1; r[{p[0][0]//5}]=1-r[{p[1][0]//5}];')
        self.simplified=0
        for k in sorted(range(n//5),key=lambda j:level[5*j]):
            q=groups[k]
            if not q:continue
            sources=sorted(q)
            terms='+'.join(f'{q[s]}*r[{s}]' for s in sources)
            expr=f'(({terms})>=33)'
            # Specialize only after evaluating the entire local truth table.
            if len(sources) in (1,2):
                table=tuple(int(sum(q[s]*v for s,v in zip(sources,bits))>=33) for bits in itertools.product((0,1),repeat=len(sources)))
                x=f'r[{sources[0]}]'
                patterns={(0,0):'0',(1,1):'1',(0,1):x,(1,0):f'(1-{x})'}
                if len(sources)==2:
                    y=f'r[{sources[1]}]'
                    patterns={(0,0,0,0):'0',(1,1,1,1):'1',(0,0,0,1):f'({x}&{y})',(0,1,1,1):f'({x}|{y})',(0,0,1,1):x,(0,1,0,1):y}
                if table in patterns:expr=patterns[table];self.simplified+=1
            lines.append(f'r[{k}]=en & {expr};')
        for j in range(self.now):lines.append(f'out[{j}]=0;')
        for i,p in enumerate(outputs):
            l,h=p[0][0]//5,p[1][0]//5
            lines.append(f'if((r[{l}]+r[{h}])!=1)return -1; out[{i//32}]|=((u32)r[{h}])<<{i%32};')
        lines.append(f'for(int j=0;j<{n//5};j++)count+=r[j];return 5*count;}}')
        source='\n'.join(lines)+'\n';key=hashlib.sha256(source.encode()).hexdigest()[:20]
        root=args.build_dir;root.mkdir(parents=True,exist_ok=True)
        c=root/(key+'.c');o=root/(key+'.o');so=root/(key+'.so');c.write_text(source)
        so=native_library(c,so,args.cc,args.ld,opt=2)
        self.lib=ctypes.CDLL(str(so.resolve()));self.fn=self.lib.evaluate
        ptr=ctypes.POINTER(ctypes.c_uint32);self.fn.argtypes=[ptr,ctypes.c_int,ptr];self.fn.restype=ctypes.c_int
        self.ib=(ctypes.c_uint32*self.niw)();self.ob=(ctypes.c_uint32*self.now)()
    def run(self,word,ablate=False):
        for j in range(self.niw):self.ib[j]=(word>>(32*j))&0xffffffff
        count=self.fn(self.ib,int(not ablate),self.ob)
        if count<0:raise RuntimeError('invalid dual-rail readout')
        return sum(int(self.ob[j])<<(32*j) for j in range(self.now)),count

def accelerate(b,a):
    c,d,ctrl=b.c,b.decoder,b.control
    names=list(c.inputs)
    assert names==[f'a{i}' for i in range(32)]+[f'b{i}' for i in range(32)]+['zero']+list(OPS)
    assert list(ctrl.inputs)==['xor','signed','unsigned','BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR','seq','target']
    assert [len(v) for v in ctrl.inputs.values()]==[32]+[1]*10+[32,32]
    al=Packed(c.n,c.edges,c.level,list(c.inputs.values()),c.outputs,a)
    de=Packed(d.n,d.edges,d.level,d.bits,d.outputs,a)
    co=Packed(ctrl.c.n,ctrl.c.edges,ctrl.c.level,[p for seq in ctrl.inputs.values() for p in seq],[ctrl.take]+ctrl.outputs,a)
    ops={k:1<<(65+i) for i,k in enumerate(OPS)}
    cn={k:1<<(34+i) for i,k in enumerate(('BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR'))}
    def ar(op,x,y,ablate=False):
        if op not in ops:raise ValueError('unknown ALU operation')
        return al.run((x&0xffffffff)|((y&0xffffffff)<<32)|ops[op],ablate)
    def dr(word,ablate=False):
        v,_=de.run(word,ablate)
        if not v:return None
        if v&(v-1):raise RuntimeError('ambiguous neural decode')
        return d.patterns[v.bit_length()-1][0]
    def cr(name,seq,target,xor=0,signed=0,unsigned=0,ablate=False):
        v,_=co.run((xor&0xffffffff)|((signed&1)<<32)|((unsigned&1)<<33)|cn.get(name,0)|((seq&0xffffffff)<<42)|((target&0xffffffff)<<74),ablate)
        ctrl.calls+=1;return v>>1,v&1
    old=(c.run,d.decode,ctrl.route);c.run=ar;d.decode=dr;ctrl.route=cr
    return old,[al.simplified,de.simplified,co.simplified]

def setup(b,a):
    _,g=accelerate(b,a);e=Engine(a.build_dir,a.cc,a.ld);return e,g
def test(a):
    b,_=load_image(fixture(),16)
    r14.NativeQuotient.build_dir=a.build_dir/'reference';r14.NativeQuotient.compiler=a.cc;r14.NativeQuotient.linker=a.ld
    r14.accelerate(b);old,g=accelerate(b,a);rng=random.Random(31024);checks=0
    for op in OPS:
        for i in range(64):
            x,y=rng.getrandbits(32),rng.getrandbits(32)
            assert b.c.run(op,x,y)==old[0](op,x,y);checks+=1
    for name,mask,v in b.decoder.patterns:
        for i in range(8):
            w=v|(rng.getrandbits(32)&~mask)
            assert b.decoder.decode(w)==old[1](w);checks+=1
    for name in ('BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR','ADD'):
        for i in range(32):
            args=(name,rng.getrandbits(32),rng.getrandbits(32),rng.getrandbits(32) if i%2 else 0,rng.randrange(2),rng.randrange(2))
            assert b.control.route(*args)==old[2](*args);checks+=1
    for f in (lambda:b.c.run('ADD',3,4,True),lambda:b.decoder.decode(0x13,True),lambda:b.control.route('JAL',4,8,ablate=True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    h=hashlib.sha256(fixture()).hexdigest()
    q,_=load_image(fixture(),16);r14.accelerate(q)
    advance(q,64);advance(b,64)
    sx=snapshot(q,h);sy=snapshot(b,h)
    assert all(np.array_equal(sx[k],sy[k]) for k in sx if k!='meta')
    mx=json.loads(sx['meta'].tobytes());my=json.loads(sy['meta'].tobytes());my['control_calls']=mx['control_calls'];assert mx==my
    return dict(status='PASS',comparisons=checks,ablation_cases=3,full_state_equal=True,specialized_rails=g)
def main():
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--budget',type=int,default=512);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.budget<=100000:p.error('bounded budget')
    if not a.image:r=test(a)
    else:
        if not a.input or not a.checkpoint:p.error('checkpoint required')
        data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b,_=load_image(data,64)
        e,g=setup(b,a);parent=json.loads(a.input.with_suffix('.json').read_text())
        if parent.get('revision')!='R19-corrected' or parent.get('checkpoint_sha256')!=hashlib.sha256(a.input.read_bytes()).hexdigest():raise ValueError('bad checkpoint provenance')
        restore(b,a.input,h);e.attach(b);cache=PageCache(b);old=b.c.run;oldm=b.math.run;checks=0
        def ar(op,x,y,ablate=False):
            nonlocal checks
            v=old(op,x,y,ablate)
            if not ablate:assert v[0]==oracle(op,x,y);checks+=1
            return v
        def mr(op,x,y,ablate=False):
            nonlocal checks
            v=oldm(op,x,y,ablate)
            if not ablate:assert v==moracle(op,x,y);checks+=1
            return v
        b.c.run=ar;b.math.run=mr
        r=advance(b,a.budget);save(b,a.checkpoint,h)
        r.update(revision='R19-corrected',evaluator='R24',image_sha256=h,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),oracle_checks=checks,cache=cache.report(),specialized_rails=g)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()

