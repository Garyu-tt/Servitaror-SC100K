"""SC500K R07: synaptic RGB palette and polled neural input FIFO.
Host injects key events; queue payload is recurrent dual-rail state.
The FIFO indices/count and bus scheduling are host control state.
"""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from neural_framebuffer_r32 import VisualBoard,restore as restore_v2,state as state_v2,upgrade as visual_upgrade,FB,WIDTH,HEIGHT,FRAME_BYTES
from neural_devices_r30 import attach,GPU,DISK
from neural_paged_r29 import PagedCrossbar,SynapticBoard,board
from neural_native_memory_r31 import Native
from neural_register_r03 import encode,decode,fire,validate
from neural_hippocampus_r27 import votes,Selector
from neural_board_r11 import RAM,UART
from neural_irq_r09 import Timer
from neural_mmu_r10 import V,R,W,X,pte
from neural_priv_r08 import GuestTrap
from neural_rv32i_r01 import I,S
from neural_linux_r12 import fixture
from neural_resume_r13 import advance
from neural_tlb_r17 import PageCache
from neural_alu_r01 import oracle
from neural_math_r05 import oracle as moracle
PAL=GPU+0x20000;KEY=GPU+0x21000
class Keyboard:
    size=32;cells=32*16*40+16*20+400
    def __init__(self):
        self.selector=Selector(self.size)
        self.state=np.repeat(encode(0,16)[None,...],self.size,axis=0)
        self.head=self.tail=self.count=0;self.events=self.reads=0
    def push(self,code,pressed=True):
        if type(code)!=int or not 1<=code<=255 or type(pressed)!=bool:raise ValueError('key event')
        if self.count==self.size:return False
        sel=votes(self.selector.select(self.tail))
        old=votes(self.state);data=votes(encode(code|(int(pressed)<<8),16))
        hold=fire(4*(old+sel[:,0,None,None]))
        put=fire(4*(data[None,:,:]+sel[:,1,None,None]))
        new=fire(7*(votes(hold)+votes(put)));validate(new)
        self.state=new;self.tail=(self.tail+1)%self.size;self.count+=1;self.events+=1
        return True
    def pop(self,ablate=False):
        if not self.count:return 0
        sel=votes(self.selector.select(self.head,ablate))[:,1]
        g=fire(4*(sel[:,None,None]+votes(self.state)))
        v=decode(fire(7*g.sum(axis=(0,3),dtype=np.int32)))
        self.head=(self.head+1)%self.size;self.count-=1;self.reads+=1;return v

class IOBoard(VisualBoard):
    def device(self,pa,n):
        if n==4 and pa%4==0:
            if PAL<=pa<PAL+1024:return 'palette'
            if pa in (KEY,KEY+4):return 'keyboard'
        return super().device(pa,n)
    def physical(self,va,n,kind):
        ac,mis={'fetch':(1,0),'load':(5,4),'store':(7,6)}[kind]
        if n not in (1,2,4) or va%n:raise GuestTrap(mis,va)
        pa=va if self._atomic_physical else self.translate(va,kind)
        if self.in_ram(pa,n):return pa
        if kind!='fetch' and self.mode!=0:
            dev=self.device(pa,n)
            if dev in ('disk','fb','palette') or (dev=='keyboard' and kind=='load') or (dev=='gpu' and ((kind=='store' and pa==GPU) or (kind=='load' and pa!=GPU))):return pa
            if UART<=pa<UART+8 and n==1:return pa
            if Timer.contains(pa) and n==4 and self.mode==3:return pa
        raise GuestTrap(ac,va)
    def load(self,a,n,signed=False):
        pa=self.physical(a,n,'load');dev=self.device(pa,n)
        if dev=='palette':return self.palette.read((pa-PAL)//4)
        if dev=='keyboard':return self.keyboard.count if pa==KEY else self.keyboard.pop()
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
        if dev=='palette':self.palette.write((pa-PAL)//4,v&0xffffff);return
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
    def rgb(self):
        # External display boundary only, not a guest execution path.
        p=np.frombuffer(self.palette.export_bytes(0,1024),dtype='<u4')
        idx=np.frombuffer(self.scanout(),dtype=np.uint8);c=p[idx]
        return np.stack((c>>16&255,c>>8&255,c&255),axis=1).astype(np.uint8).tobytes()

def upgrade(b,a):
    b.__class__=IOBoard;b.palette=PagedCrossbar(4,rows=4,cols=16)
    Native(a).attach(b.palette);b.keyboard=Keyboard();return b
def state(b,h):
    s=state_v2(b,h);k=b.keyboard
    s.update(palette_weights=b.palette.weights,key_state=k.state,
        io_meta=np.frombuffer(json.dumps(dict(schema=1,palette_reads=b.palette.reads,palette_writes=b.palette.writes,
        head=k.head,tail=k.tail,count=k.count,events=k.events,reads=k.reads),sort_keys=True).encode(),dtype=np.uint8))
    return s
def save(b,p,h):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**state(b,h));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def restore(data,a,p):
    b=upgrade(restore_v2(data,a,p),a)
    with np.load(p,allow_pickle=False) as z:
        if 'io_meta' not in z.files:return b
        m=json.loads(z['io_meta'].tobytes());w=z['palette_weights'];k=z['key_state']
        if m['schema']!=1 or w.shape!=b.palette.weights.shape or w.dtype!=np.uint8 or not np.all((w==0x17)|(w==0x71)):raise ValueError('palette state')
        if k.shape!=b.keyboard.state.shape or k.dtype!=np.int8 or not np.all((k==0)|(k==1)):raise ValueError('keyboard state')
        validate(k)
        if not (0<=m['head']<32 and 0<=m['tail']<32 and 0<=m['count']<=32 and (m['head']+m['count'])%32==m['tail']):raise ValueError('FIFO indices')
        if not all(type(m[x])==int and m[x]>=0 for x in ('events','reads','palette_reads','palette_writes')) or m['events']-m['reads']!=m['count']:raise ValueError('FIFO counts')
        b.palette.weights=w;b.palette.reads=m['palette_reads'];b.palette.writes=m['palette_writes']
        b.keyboard.state=k.copy()
        for x in ('head','tail','count','events','reads'):setattr(b.keyboard,x,m[x])
    return b
def test(a):
    data=fixture();h=hashlib.sha256(data).hexdigest()
    b=board(data,a);b.__class__=SynapticBoard;b.install(PagedCrossbar(256));attach(b);visual_upgrade(b,a);upgrade(b,a)
    root=RAM+0x1000;table=RAM+0x2000;code=RAM+0x3000
    va=0x40000000;pv=va+4096;fv=va+8192;kv=va+12288
    b.phys_write(root+256*4,4,pte(table,V))
    for j,pa in enumerate((code,PAL,FB,KEY)):b.phys_write(table+j*4,4,pte(pa,V|R|W|(X if j==0 else 0)))
    # Guest writes RGB 0x123456 to palette index 1, fills first 4 pixels
    # with index 1, polls and consumes press/release events.
    words=[0x400010b7,0x00123137,I(0x13,2,0,2,0x456),S(2,1,2,4),I(3,3,2,1,4),
           0x40002237,0x010102b7,I(0x13,5,0,5,0x101),S(2,4,5,0),
           0x40003337,I(3,7,2,6,0),I(3,8,2,6,4),I(3,9,2,6,4),I(3,10,2,6,4),0x6f]
    assert b.keyboard.push(30,True) and b.keyboard.push(30,False)
    for i,w in enumerate(words):b.phys_write(code+4*i,4,w)
    b.put(0x180,0x80000000|(root>>12));b.mode=1;b.pc=va;PageCache(b)
    r=advance(b,len(words));assert r['delta']==len(words) and r['stop']=='budget',r
    assert [b.reg[i] for i in (3,7,8,9,10)]==[0x123456,2,0x11e,30,0]
    rgb=b.rgb();assert rgb[:12]==bytes((0x12,0x34,0x56))*4 and not any(rgb[12:])
    checks=0
    for f,cause in ((lambda:b.store(kv,4,0),7),(lambda:b.load(pv,1),5),(lambda:b.fetch(kv),12),(lambda:b.prepare_instruction_extension('LR.W',pv,0,0),5)):
        try:f()
        except GuestTrap as ex:assert ex.cause==cause;checks+=1
        else:raise AssertionError('access')
    k=b.keyboard
    for i in range(32):assert k.push(i+1,i%2==0)
    assert not k.push(99)
    before=k.state.copy();head=k.head
    try:k.pop(True)
    except RuntimeError:checks+=1
    else:raise AssertionError('ablation')
    assert k.head==head and np.array_equal(before,k.state)
    for i in range(17):assert k.pop()==(i+1)|((i%2==0)<<8)
    for i in range(17):assert k.push(i+100)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'full.npz';save(b,p,h);q=restore(data,a,p)
        assert all(np.array_equal(v,state(q,h)[j]) for j,v in state(b,h).items())
        assert q.rgb()==rgb
        expected=[(i+1)|((i%2==0)<<8) for i in range(17,32)]+[i+100+256 for i in range(17)]
        assert [q.keyboard.pop() for _ in range(32)]==expected
        assert q.keyboard.pop()==0
    cells=123110+b.memory.cells+b.disk.cells+b.gpu.cells+b.framebuffer.cells+b.palette.cells+k.cells
    assert cells<=500000
    return dict(status='PASS',guest_instructions=r['delta'],palette_entries=256,rgb_bits=24,keyboard_capacity=32,
        guest_key_events=[hex(b.reg[8]),hex(b.reg[9])],rgb_first_pixel=list(rgb[:3]),rgb_sha256=hashlib.sha256(rgb).hexdigest(),
        fifo_wraparound=True,overflow_rejected=True,checks=checks,exact_checkpoint=True,estimated_cells=cells)
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
        fb_reads=b.framebuffer.reads,fb_writes=b.framebuffer.writes,palette_writes=b.palette.writes,key_reads=b.keyboard.reads,cache=cache.report(),
        no_guest_bytearray=b.ram is None,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=2000);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.steps<=50000:p.error('bounded steps')
    if a.image and not(a.input and a.checkpoint):p.error('input/output checkpoint required')
    r=run(a) if a.image else test(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

