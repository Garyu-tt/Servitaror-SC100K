"""R15: native recurrent rail updates, retaining full spike state.

The register bank still evaluates all 32 selectors and all storage/read
rails. C only schedules the original weighted sums and threshold updates.
No integer shadow file or indexing a register by guest address.
"""
import argparse,ctypes,hashlib,json,random,subprocess,tempfile,time
from pathlib import Path
from native_bundle import native_library
import numpy as np
from neural_register_r03 import RegisterBank,validate
from neural_linux_r12 import load_image,fixture
import neural_native_r14 as r14
import neural_resume_r13 as r13

SOURCE=r"""
typedef unsigned char u8;
typedef unsigned int u32;
static int fire(int x){return x>=33;}
static int vote(const u8 *s){return s[0]+s[1]+s[2]+s[3]+s[4];}
static int valid(int lo,int hi){return (lo==5&&hi==0)||(lo==0&&hi==5);}
static int select_rails(const signed char *w,u32 addr,int en,int sel[32][2]){
    int src[5][2],matches=0;
    for(int i=0;i<5;i++){int v=(addr>>i)&1;src[i][0]=5*(1-v);src[i][1]=5*v;}
    for(int r=0;r<32;r++){
        for(int a=0;a<2;a++){
            int sum=0;
            for(int i=0;i<5;i++)for(int j=0;j<2;j++)sum+=w[((r*2+a)*5+i)*2+j]*src[i][j];
            sel[r][a]=5*fire(en*sum);
        }
        if(!valid(sel[r][0],sel[r][1]))return -1;
        matches+=sel[r][1]==5;
    }
    return matches==1?0:-1;
}
int reg_read(const signed char *w,const u8 *state,u32 addr,int en,u32 *value){
    int sel[32][2];if(select_rails(w,addr,en,sel))return -1;
    u32 out=0;
    for(int i=0;i<32;i++){
        int rail[2]={0,0};
        for(int a=0;a<2;a++){
            int sum=0;
            for(int r=0;r<32;r++)sum+=5*fire(en*4*(sel[r][1]+vote(state+((r*32+i)*2+a)*5)));
            rail[a]=5*fire(7*sum);
        }
        if(!valid(rail[0],rail[1]))return -2;
        out|=((u32)(rail[1]==5))<<i;
    }
    *value=out;return 0;
}
int reg_write(const signed char *w,const u8 *state,u8 *next,u32 addr,u32 value,int en){
    int sel[32][2];if(select_rails(w,addr,en,sel))return -1;
    sel[0][0]=5;sel[0][1]=0;
    for(int r=0;r<32;r++)for(int i=0;i<32;i++){
        int bit=(value>>i)&1,rail[2];
        for(int a=0;a<2;a++){
            int offset=((r*32+i)*2+a)*5;
            int hold=5*fire(en*4*(vote(state+offset)+sel[r][0]));
            int put=5*fire(en*4*(5*(a==bit)+sel[r][1]));
            rail[a]=5*fire(7*(hold+put));
            for(int j=0;j<5;j++)next[offset+j]=rail[a]/5;
        }
        if(!valid(rail[0],rail[1]))return -2;
    }
    return 0;
}
int feedback_read(const u8 *state,int width,u32 *out){
    u32 v=0;
    for(int i=0;i<width;i++){
        int lo=5*fire(7*vote(state+i*10)),hi=5*fire(7*vote(state+i*10+5));
        if(!valid(lo,hi))return -1;
        v|=((u32)(hi==5))<<i;
    }
    *out=v;return 0;
}
int feedback_write(u8 *state,int width,u32 value,int en){
    if(!en)return -1;
    for(int i=0;i<width;i++)for(int a=0;a<2;a++){
        int v=fire(en*7*5*(a==((value>>i)&1)));
        for(int j=0;j<5;j++)state[i*10+a*5+j]=v;
    }
    return 0;
}
"""
class Engine:
    def __init__(self,root,cc,ld):
        root=Path(root);root.mkdir(parents=True,exist_ok=True)
        c=root/'recurrent.c';o=root/'recurrent.o';so=root/'recurrent.so';c.write_text(SOURCE)
        so=native_library(c,so,cc,ld,opt=2)
        self.lib=ctypes.CDLL(str(so.resolve()));p=ctypes.c_void_p;u=ctypes.c_uint32;i=ctypes.c_int
        for n,args in [('reg_read',[p,p,u,i,p]),('reg_write',[p,p,p,u,u,i]),('feedback_read',[p,i,p]),('feedback_write',[p,i,u,i])]:
            fn=getattr(self.lib,n);fn.argtypes=args;fn.restype=i
    def read(self,s,width):
        out=ctypes.c_uint32()
        if self.lib.feedback_read(s.ctypes.data,width,ctypes.byref(out)):raise RuntimeError('invalid dual-rail feedback')
        return out.value
    def write(self,width,v,ablate=False):
        s=np.empty((width,2,5),dtype=np.int8)
        if self.lib.feedback_write(s.ctypes.data,width,v,int(not ablate)):raise RuntimeError('invalid dual-rail feedback')
        return s
    def attach(self,b):
        bank=b.reg;pc=b.counter;priv=b.priv;csr=b.csrs
        def read(i,*,ablate=False):
            bank._address(i);out=ctypes.c_uint32()
            if self.lib.reg_read(bank._w.ctypes.data,bank._state.ctypes.data,i,int(not ablate),ctypes.byref(out)):raise RuntimeError('invalid register rails')
            bank.reads+=1;return out.value
        def write(i,value,*,ablate=False):
            bank._address(i);s=np.empty_like(bank._state)
            if self.lib.reg_write(bank._w.ctypes.data,bank._state.ctypes.data,s.ctypes.data,i,value&0xffffffff,int(not ablate)):raise RuntimeError('invalid register rails')
            bank._state=s;bank.writes+=1
        def commit(v,ablate=False):
            pc.state=self.write(32,v&0xffffffff,ablate);pc.commits+=1
        def mode_set(v):
            if v not in (0,1,3):raise ValueError('bad privilege mode')
            priv.state=self.write(2,v)
        def csr_raw(a):return self.read(csr.state[a] if a in csr.state else csr.fixed[a],32)
        def csr_commit(pending,ablate=False):
            if pending is None:return
            a,v=pending;csr.state[a]=self.write(32,v&0xffffffff,ablate);csr.writes+=1
        bank.read=read;bank.write=write;pc.read=lambda:self.read(pc.state,32);pc.commit=commit
        priv.get=lambda:self.read(priv.state,2);priv.set=mode_set;csr.raw=csr_raw;csr.commit=csr_commit
def setup(b,a):
    r14.NativeQuotient.build_dir=a.build_dir
    r14.NativeQuotient.compiler=a.cc;r14.NativeQuotient.linker=a.ld
    r14.accelerate(b)
    eng=Engine(a.build_dir,a.cc,a.ld)
    return eng
def selftest(a):
    b,_=load_image(fixture(),16);ref=RegisterBank();ref._state=b.reg._state.copy();eng=setup(b,a);eng.attach(b)
    rng=random.Random(31015);checks=0
    for _ in range(512):
        i=rng.randrange(32);v=rng.getrandbits(32);b.reg.write(i,v);ref.write(i,v)
        assert np.array_equal(b.reg._state,ref._state)
        for j in (i,0,rng.randrange(32)):
            assert b.reg.read(j)==ref.read(j);checks+=1
    for fn in (lambda:b.reg.read(1,ablate=True),lambda:b.reg.write(1,2,ablate=True),lambda:b.counter.commit(3,True),lambda:b.csrs.commit((0x340,3),True)):
        try:fn()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    h=hashlib.sha256(fixture()).hexdigest()
    q,_=load_image(fixture(),16);r14.accelerate(q)
    # Compare complete state after execution, not just output.
    f,_=load_image(fixture(),16);r14.accelerate(f);eng.attach(f)
    r13.advance(q,64);r13.advance(f,64)
    assert all(np.array_equal(v,r13.snapshot(f,h)[k]) for k,v in r13.snapshot(q,h).items())
    return dict(status='PASS',register_reads=checks,register_writes=512,full_state_equal=True,ablation_cases=4)
def main():
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--budget',type=int,default=512);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.budget<=100000:p.error('bounded budget')
    if not a.image:r=selftest(a)
    else:
        if not a.input or not a.checkpoint:p.error('checkpoint required')
        data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b,_=load_image(data,64)
        eng=setup(b,a);r13.restore(b,a.input,h);eng.attach(b)
        r=r13.advance(b,a.budget);r13.save(b,a.checkpoint,h)
        r.update(image_sha256=h,checkpoint_bytes=a.checkpoint.stat().st_size)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()

