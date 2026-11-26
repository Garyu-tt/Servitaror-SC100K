"""R14: native evaluation of verified synchronized neural rails.

Compile the R13 verified graph into explicit weighted threshold equations.
No ISA operations or guest instruction batching in generated C.
"""
import argparse,ctypes,hashlib,json,os,random,subprocess,tempfile,time
from pathlib import Path
import numpy as np
import neural_resume_r13 as r13
from neural_linux_r12 import load_image,fixture
from neural_alu_r01 import OPS
class NativeQuotient(r13.Quotient):
    build_dir=None
    compiler='cc'
    linker=None
    def __init__(self,n,edges,level,ports,outputs):
        super().__init__(n,edges,level,ports,outputs)
        # R13 has already verified identical incoming synapses per replica.
        groups=[{} for _ in range(n//5)]
        for s,t,w in edges:
            if t%5==0:
                q=groups[t//5];q[s//5]=q.get(s//5,0)+w
        lines=['int evaluate(const unsigned char *bits,int enabled,unsigned char *out){',
            f'unsigned char r[{n//5}];int count=0;']
        for i,p in enumerate(ports):
            lines.append(f'r[{p[0][0]//5}]=1-bits[{i}];r[{p[1][0]//5}]=bits[{i}];')
        for r in sorted(range(n//5),key=lambda r:level[5*r]):
            if not groups[r]:continue
            terms='+'.join(f'({w}*r[{s}])' for s,w in sorted(groups[r].items()))
            lines.append(f'r[{r}]=enabled && ({terms})>=33;')
        for i,p in enumerate(outputs):
            lines.append(f'out[{2*i}]=r[{p[0][0]//5}];out[{2*i+1}]=r[{p[1][0]//5}];')
        lines.append(f'for(int i=0;i<{n//5};i++)count+=r[i];return 5*count;}}')
        source='\n'.join(lines)+'\n';key=hashlib.sha256(source.encode()).hexdigest()[:20]
        root=Path(self.build_dir);root.mkdir(parents=True,exist_ok=True)
        c=root/(key+'.c');so=root/(key+'.so');obj=root/(key+'.o')
        c.write_text(source)
        if self.linker:
            subprocess.run([self.compiler,'-c','-o',str(obj),str(c)],check=True)
            subprocess.run([self.linker,'-shared','-z','noexecstack','-o',str(so),str(obj)],check=True)
        else:
            subprocess.run([self.compiler,'-O2','-fPIC','-shared','-Wl,-z,noexecstack','-o',str(so),str(c)],check=True)
        self.lib=ctypes.CDLL(str(so.resolve()));self.fn=self.lib.evaluate
        self.fn.argtypes=[ctypes.POINTER(ctypes.c_ubyte),ctypes.c_int,ctypes.POINTER(ctypes.c_ubyte)]
        self.fn.restype=ctypes.c_int
        self.inbuf=(ctypes.c_ubyte*len(ports))();self.outbuf=(ctypes.c_ubyte*(2*len(outputs)))()
    def run(self,bits,ablate=False):
        self.inbuf[:]=bits
        count=self.fn(self.inbuf,int(not ablate),self.outbuf)
        result=[]
        for i in range(0,len(self.outbuf),2):
            l,h=self.outbuf[i],self.outbuf[i+1]
            if (l,h) not in ((1,0),(0,1)):raise RuntimeError('invalid dual-rail readout')
            result.append(h)
        return result,count

def accelerate(b):
    prev=r13.Quotient
    try:r13.Quotient=NativeQuotient;return r13.accelerate(b)
    finally:r13.Quotient=prev

def selftest():
    b,_=load_image(fixture(),16);r13.accelerate(b)
    old,groups=accelerate(b);rng=random.Random(31014);checks=0
    for op in OPS:
        for i in range(32):
            a,v=rng.getrandbits(32),rng.getrandbits(32)
            assert b.c.run(op,a,v)==old[0](op,a,v);checks+=1
    for _ in range(256):
        w=rng.getrandbits(32);assert b.decoder.decode(w)==old[1](w);checks+=1
    for name in ('BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR','ADD'):
        for i in range(16):
            args=(name,rng.getrandbits(32),rng.getrandbits(32),i%2,rng.randrange(2),rng.randrange(2))
            assert b.control.route(*args)==old[2](*args);checks+=1
    for f in (lambda:b.c.run('ADD',3,4,True),lambda:b.decoder.decode(0x13,True),lambda:b.control.route('JAL',4,8,ablate=True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    h=hashlib.sha256(fixture()).hexdigest();r13.advance(b,3)
    before={k:v.copy() for k,v in r13.snapshot(b,h).items()}
    # Independent reference execution from identical neural state.
    r13.advance(b,32);native={k:v.copy() for k,v in r13.snapshot(b,h).items()}
    with tempfile.TemporaryDirectory() as tmp:
        cp=Path(tmp)/'test.npz'
        with cp.open('wb') as f:np.savez_compressed(f,**before)
        r13.restore(b,cp,h)
        b.c.run,b.decoder.decode,b.control.route=old
        r13.advance(b,32)
        assert all(np.array_equal(v,r13.snapshot(b,h)[k]) for k,v in native.items())
    return dict(status='PASS',equivalence_checks=checks,ablation=True,full_state_equal=True,quotient_groups=groups)
def main():
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--budget',type=int,default=512);p.add_argument('--output',type=Path);a=p.parse_args()
    NativeQuotient.build_dir=a.build_dir;NativeQuotient.compiler=a.cc;NativeQuotient.linker=a.ld
    if not 1<=a.budget<=100000:p.error('bounded budget')
    if not a.image:r=selftest()
    else:
        if not a.input or not a.checkpoint:p.error('input and output checkpoint required')
        data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest()
        b,_=load_image(data,64);_,groups=accelerate(b);r13.restore(b,a.input,h)
        r=r13.advance(b,a.budget);r13.save(b,a.checkpoint,h)
        r.update(image_sha256=h,quotient_groups=groups,checkpoint_bytes=a.checkpoint.stat().st_size)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()

