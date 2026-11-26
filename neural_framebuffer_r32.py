"""SC500K R06: 320x200 indexed-color synaptic framebuffer.
The legacy 8x8 raster remains intact. New pixels are Q4 coefficients,
read/written by the same native threshold engine as synaptic RAM.
No host framebuffer shadow; scanout decoding is a diagnostic/I/O boundary.
"""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from neural_paged_r29 import PagedCrossbar,SynapticBoard,board
from neural_devices_r30 import DeviceBoard,attach,restore as restore_v1,state as state_v1,GPU,DISK
from neural_native_memory_r31 import Native
from neural_board_r11 import RAM,UART
from neural_irq_r09 import Timer
from neural_register_r03 import validate
from neural_mmu_r10 import V,R,W,X,pte
from neural_priv_r08 import GuestTrap
from neural_rv32i_r01 import I,S
from neural_linux_r12 import fixture
from neural_resume_r13 import advance
from neural_tlb_r17 import PageCache
from neural_alu_r01 import oracle
from neural_math_r05 import oracle as moracle
WIDTH=320;HEIGHT=200;FRAME_BYTES=WIDTH*HEIGHT;FB=GPU+0x10000
class VisualBoard(DeviceBoard):
    def device(self,pa,n):
        if n==4 and pa%4==0 and FB<=pa<FB+FRAME_BYTES:return 'fb'
        return super().device(pa,n)
    def physical(self,va,n,kind):
        ac,mis={'fetch':(1,0),'load':(5,4),'store':(7,6)}[kind]
        if n not in (1,2,4) or va%n:raise GuestTrap(mis,va)
        pa=va if self._atomic_physical else self.translate(va,kind)
        if self.in_ram(pa,n):return pa
        if kind!='fetch' and self.mode!=0:
            dev=self.device(pa,n)
            if dev in ('disk','fb') or (dev=='gpu' and ((kind=='store' and pa==GPU) or (kind=='load' and pa!=GPU))):return pa
            if UART<=pa<UART+8 and n==1:return pa
            if Timer.contains(pa) and n==4 and self.mode==3:return pa
        raise GuestTrap(ac,va)
    def load(self,a,n,signed=False):
        pa=self.physical(a,n,'load');dev=self.device(pa,n)
        if dev=='fb':return self.framebuffer.read((pa-FB)//4)
        if dev=='disk':return self.disk.read((pa-DISK)//4)
        if dev=='gpu':
            pix=validate(self.gpu.state)[:,:,1].reshape(-1);off=0 if pa==GPU+4 else 32
            return sum(int(x==5)<<i for i,x in enumerate(pix[off:off+32]))
        if UART<=pa<UART+8:
            i=pa-UART
            return 0x60 if i==5 else 1 if i==2 else self.uart_regs[i]
        if Timer.contains(pa):return self.timer.read(pa)
        return self.phys_read(pa,n,signed)
    def store(self,a,n,v):
        pa=self.physical(a,n,'store');dev=self.device(pa,n)
        if dev=='fb':self.framebuffer.write((pa-FB)//4,v);return
        if dev=='disk':self.disk.write((pa-DISK)//4,v);return
        if dev=='gpu':self.gpu.rectangle(v&255,(v>>8)&255,(v>>16)&1);self.gpu_writes+=1;return
        if UART<=pa<UART+8:
            i=pa-UART
            if i==0 and not self.uart_regs[3]&0x80:self.serial.append(v&255)
            else:self.uart_regs[i]=v&255
            return
        if Timer.contains(pa):self.timer.write(pa,v);self.reservation=None;self.sync_pending();return
        self.phys_write(pa,n,v)
    def scanout(self):
        return self.framebuffer.export_bytes(0,FRAME_BYTES)

def upgrade(b,a):
    b.__class__=VisualBoard;b.framebuffer=PagedCrossbar(4,rows=256,cols=16)
    n=Native(a)
    for m in (b.memory,b.disk,b.framebuffer):n.attach(m)
    return b

def state(b,h):
    s=state_v1(b,h);s.update(fb_weights=b.framebuffer.weights,
        fb_meta=np.frombuffer(json.dumps(dict(schema=1,width=WIDTH,height=HEIGHT,
        reads=b.framebuffer.reads,writes=b.framebuffer.writes),sort_keys=True).encode(),dtype=np.uint8))
    return s
def save(b,p,h):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**state(b,h));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def restore(data,a,p):
    b=upgrade(restore_v1(data,a,p),a)
    with np.load(p,allow_pickle=False) as z:
        if 'fb_meta' not in z.files:return b
        m=json.loads(z['fb_meta'].tobytes());w=z['fb_weights']
        if (m['schema'],m['width'],m['height'])!=(1,WIDTH,HEIGHT) or w.shape!=b.framebuffer.weights.shape or w.dtype!=np.uint8:raise ValueError('framebuffer geometry')
        if not np.all((w==0x17)|(w==0x71)):raise ValueError('framebuffer coefficients')
        b.framebuffer.weights=w;b.framebuffer.reads=m['reads'];b.framebuffer.writes=m['writes']
    return b
def test(a):
    data=fixture();h=hashlib.sha256(data).hexdigest()
    b=board(data,a);b.__class__=SynapticBoard;b.install(PagedCrossbar(256));attach(b);upgrade(b,a)
    root=RAM+0x1000;table=RAM+0x2000;code=RAM+0x3000
    va=0x40000000;fv=va+4096;last=va+8192
    b.phys_write(root+256*4,4,pte(table,V))
    for j,pa in enumerate((code,FB,FB+0xf000)):
        b.phys_write(table+j*4,4,pte(pa,V|R|W|(X if j==0 else 0)))
    # Four color indices via real guest store; first and last visible words.
    words=[0x400010b7,0x44332137,I(0x13,2,0,2,0x211),S(2,1,2,0),
           I(3,3,2,1,0),0x40002237,I(0x13,4,0,4,0x1fc),
           I(0x13,5,0,0,0x55),S(2,4,5,0),I(3,6,2,4,0),0x0000006f]
    # Last visible word offset in final page is 0x9fc (64000-61440-4).
    words[6]=I(0x13,4,0,4,0x7fc);words.insert(7,I(0x13,4,0,4,0x200))
    for i,w in enumerate(words):b.phys_write(code+4*i,4,w)
    b.put(0x180,0x80000000|(root>>12));b.mode=1;b.pc=va;PageCache(b)
    r=advance(b,len(words));assert r['delta']==len(words) and r['stop']=='budget',r
    assert b.reg[3]==0x44332211 and b.reg[6]==0x55
    frame=b.scanout();assert frame[:4]==bytes((0x11,0x22,0x33,0x44)) and frame[-4:]==bytes((0x55,0,0,0))
    assert not any(frame[4:-4])
    checks=0
    for f,cause in ((lambda:b.load(fv,1),5),(lambda:b.store(last+0xa00,4,1),7),(lambda:b.fetch(fv),12)):
        try:f()
        except GuestTrap as ex:assert ex.cause==cause;checks+=1
        else:raise AssertionError('access')
    before=b.framebuffer.weights.copy()
    for f in (lambda:b.framebuffer.read(0,True),lambda:b.framebuffer.write(0,1,True)):
        try:f()
        except RuntimeError:checks+=1
        else:raise AssertionError('ablation')
    assert np.array_equal(before,b.framebuffer.weights)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'full.npz';save(b,p,h);q=restore(data,a,p)
        assert all(np.array_equal(v,state(q,h)[k]) for k,v in state(b,h).items())
        assert q.scanout()==frame
    cells=123110+b.memory.cells+b.disk.cells+b.gpu.cells+b.framebuffer.cells
    assert cells<=500000
    return dict(status='PASS',width=WIDTH,height=HEIGHT,indexed_colors=256,frame_bytes=FRAME_BYTES,
        framebuffer_cells=b.framebuffer.cells,estimated_cells=cells,guest_instructions=r['delta'],
        guest_pixels_first=list(frame[:4]),guest_pixels_last=list(frame[-4:]),checks=checks,
        exact_checkpoint=True,frame_sha256=hashlib.sha256(frame).hexdigest())
def run(a):
    data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b=restore(data,a,a.input)
    cache=PageCache(b);old=b.c.run;oldm=b.math.run;checks=0
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
    b.c.run=ar;b.math.run=mr;before=(b.memory.reads,b.memory.writes)
    r=advance(b,a.steps);save(b,a.checkpoint,h)
    r.update(oracle_checks=checks,ram_read_delta=b.memory.reads-before[0],ram_write_delta=b.memory.writes-before[1],
        fb_reads=b.framebuffer.reads,fb_writes=b.framebuffer.writes,cache=cache.report(),
        no_guest_bytearray=b.ram is None,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=2000);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.steps<=50000:p.error('bounded steps')
    if a.image and not(a.input and a.checkpoint):p.error('input/output checkpoint required')
    r=run(a) if a.image else test(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

