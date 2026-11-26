"""SC500K R10: bidirectional polled UART with recurrent RX data.

Host provides serial transport, FIFO pointers and UART control registers.
RX payload is authoritative dual-rail state. No host command interpreter.
No PLIC or interrupt-driven 8250 claim.
"""
import argparse,hashlib,json,os,tempfile
from pathlib import Path
import numpy as np
from neural_io_r33 import IOBoard,restore as restore_v3,state as state_v3,upgrade as io_upgrade
from neural_framebuffer_r32 import upgrade as visual_upgrade,FB
from neural_devices_r30 import attach,GPU,DISK
from neural_paged_r29 import PagedCrossbar,SynapticBoard,board
from neural_register_r03 import encode,decode,fire,validate
from neural_hippocampus_r27 import Selector,votes
from neural_board_r11 import RAM,UART
from neural_irq_r09 import Timer
from neural_mmu_r10 import V,R,W,X,pte
from neural_rv32i_r01 import I,S
from neural_linux_r12 import fixture
from neural_resume_r13 import advance
from neural_tlb_r17 import PageCache
from neural_alu_r01 import oracle
from neural_math_r05 import oracle as moracle

class RX:
    size=32
    def __init__(self):
        self.selector=Selector(self.size)
        self.state=np.repeat(encode(0,8)[None,...],self.size,axis=0)
        self.head=self.tail=self.count=self.received=self.reads=self.dropped=0
        self.cells=self.selector.cells+40*self.size*8+20*8
    def push(self,v):
        if type(v)!=int or not 0<=v<=255:raise ValueError('byte')
        if self.count==self.size:self.dropped+=1;return False
        sel=votes(self.selector.select(self.tail));old=votes(self.state);data=votes(encode(v,8))
        hold=fire(4*(old+sel[:,0,None,None]));put=fire(4*(data[None,:,:]+sel[:,1,None,None]))
        nxt=fire(7*(votes(hold)+votes(put)));validate(nxt);self.state=nxt
        self.tail=(self.tail+1)%self.size;self.count+=1;self.received+=1;return True
    def pop(self,ablate=False):
        if not self.count:return None
        sel=votes(self.selector.select(self.head,ablate))[:,1]
        g=fire(4*(sel[:,None,None]+votes(self.state)))
        v=decode(fire(7*g.sum(axis=(0,3),dtype=np.int32)))
        self.head=(self.head+1)%self.size;self.count-=1;self.reads+=1;return v
    def clear(self):
        self.head=self.tail;self.reads+=self.count;self.count=0

class SerialBoard(IOBoard):
    def uart_input(self,data):
        if not isinstance(data,bytes):raise ValueError('bytes')
        # Backpressure: entire batch accepted or none; no silent truncation.
        if len(data)>self.rx.size-self.rx.count:return False
        for v in data:assert self.rx.push(v)
        return True
    def uart_read(self,i):
        r=self.uart_regs;dlab=bool(r[3]&128)
        if i==0:
            if dlab:return self.dll
            v=self.rx.pop();return 0 if v is None else v
        if i==1:return self.dlm if dlab else r[1]
        if i==2:return (0xc0 if self.fcr&1 else 0)|1 # polled, no IRQ
        if i==5:return 0x60|int(bool(self.rx.count))
        if i==6:return 0xb0 # CTS, DSR, DCD
        return r[i]
    def uart_write(self,i,v):
        v&=255;r=self.uart_regs;dlab=bool(r[3]&128)
        if i==0:
            if dlab:self.dll=v
            else:self.serial.append(v)
        elif i==1:
            if dlab:self.dlm=v
            else:r[1]=v&15
        elif i==2:
            self.fcr=v&0xc1
            if v&2:self.rx.clear()
        elif i not in (5,6):r[i]=v
    def load(self,a,n,signed=False):
        pa=self.physical(a,n,'load');dev=self.device(pa,n)
        if UART<=pa<UART+8:
            v=self.uart_read(pa-UART)
            return v|0xffffff00 if signed and v&128 else v
        if dev=='palette':return self.palette.read((pa-0x31020000)//4)
        if dev=='keyboard':return self.keyboard.count if pa==0x31021000 else self.keyboard.pop()
        if dev=='fb':return self.framebuffer.read((pa-FB)//4)
        if dev=='disk':return self.disk.read((pa-DISK)//4)
        if dev=='gpu':
            pix=validate(self.gpu.state)[:,:,1].reshape(-1);off=0 if pa==GPU+4 else 32
            return sum(int(x==5)<<i for i,x in enumerate(pix[off:off+32]))
        if Timer.contains(pa):return self.timer.read(pa)
        return self.phys_read(pa,n,signed)
    def store(self,a,n,v):
        pa=self.physical(a,n,'store');dev=self.device(pa,n)
        if UART<=pa<UART+8:self.uart_write(pa-UART,v);return
        if dev=='palette':self.palette.write((pa-0x31020000)//4,v&0xffffff);return
        if dev=='fb':self.framebuffer.write((pa-FB)//4,v);return
        if dev=='disk':self.disk.write((pa-DISK)//4,v);return
        if dev=='gpu':self.gpu.rectangle(v&255,(v>>8)&255,(v>>16)&1);self.gpu_writes+=1;return
        if Timer.contains(pa):self.timer.write(pa,v);self.reservation=None;self.sync_pending();return
        self.phys_write(pa,n,v)
    def sbi(self):
        if self.reg[17]==2:
            v=self.rx.pop();self.reg[10]=0xffffffff if v is None else v;self.sbi_calls+=1;return
        return super().sbi()

def upgrade(b):
    b.__class__=SerialBoard;b.rx=RX()
    # Legacy snapshot did not separate DLAB; preserve existing normal
    # UART registers, initialize divisor latches independently.
    b.dll=0;b.dlm=0;b.fcr=0;return b
def state(b,h):
    s=state_v3(b,h);r=b.rx
    s.update(rx_state=r.state,serial_meta=np.frombuffer(json.dumps(dict(schema=1,dll=b.dll,dlm=b.dlm,fcr=b.fcr,
        **{k:getattr(r,k) for k in ('head','tail','count','received','reads','dropped')}),sort_keys=True).encode(),dtype=np.uint8))
    return s
def save(b,p,h):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**state(b,h));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def restore(data,a,p):
    b=upgrade(restore_v3(data,a,p))
    with np.load(p,allow_pickle=False) as z:
        if 'serial_meta' not in z.files:return b
        m=json.loads(z['serial_meta'].tobytes());r=z['rx_state']
        if m['schema']!=1 or r.shape!=b.rx.state.shape or r.dtype!=np.int8 or not np.all((r==0)|(r==1)):raise ValueError('rx state')
        validate(r)
        if not all(type(m[k])==int and m[k]>=0 for k in ('dll','dlm','fcr','head','tail','count','received','reads','dropped')):raise ValueError('rx metadata')
        if not(0<=m['head']<32 and 0<=m['tail']<32 and m['count']<=32 and (m['head']+m['count'])%32==m['tail'] and m['received']-m['reads']==m['count'] and m['dll']<=255 and m['dlm']<=255 and not m['fcr']&~0xc1):raise ValueError('rx indices')
        b.rx.state=r.copy()
        for k in ('head','tail','count','received','reads','dropped'):setattr(b.rx,k,m[k])
        for k in ('dll','dlm','fcr'):setattr(b,k,m[k])
    return b

def fresh(data,a):
    b=board(data,a);b.__class__=SynapticBoard;b.install(PagedCrossbar(256));attach(b);visual_upgrade(b,a);io_upgrade(b,a)
    return upgrade(b)
def test(a):
    data=fixture();h=hashlib.sha256(data).hexdigest();b=fresh(data,a)
    root=RAM+0x1000;table=RAM+0x2000;code=RAM+0x3000;va=0x40000000
    b.phys_write(root+256*4,4,pte(table,V))
    b.phys_write(table,4,pte(code,V|R|W|X))
    b.phys_write(table+4,4,pte(UART,V|R|W))
    # Guest loads LSR and RBR, echoes each byte to THR, reads empty LSR.
    words=[0x400010b7,I(3,2,4,1,5),I(3,3,4,1,0),S(0,1,3,0),
           I(3,4,0,1,0),S(0,1,4,0),I(3,5,4,1,0),S(0,1,5,0),I(3,6,4,1,5),0x6f]
    assert b.uart_input(b'A\xff\0')
    for i,w in enumerate(words):b.phys_write(code+4*i,4,w)
    b.put(0x180,0x80000000|(root>>12));b.mode=1;b.pc=va;PageCache(b)
    r=advance(b,len(words));assert r['delta']==len(words) and r['stop']=='budget',r
    assert b.serial==b'A\xff\0' and [b.reg[i] for i in (2,3,4,5,6)]==[0x61,65,0xffffffff,0,0x60]
    addr=va+4096;b.store(addr+3,1,128);b.store(addr,1,0x34);b.store(addr+1,1,0x12)
    assert b.load(addr,1)==0x34 and b.load(addr+1,1)==0x12
    b.store(addr+3,1,3);assert b.serial==b'A\xff\0'
    assert b.uart_input(bytes(range(32))) and not b.uart_input(b'!') and b.rx.count==32
    before=b.rx.state.copy()
    try:b.rx.pop(True)
    except RuntimeError:pass
    else:raise AssertionError('ablation')
    assert np.array_equal(before,b.rx.state) and b.rx.count==32
    assert [b.load(addr,1) for _ in range(17)]==list(range(17))
    assert b.uart_input(bytes(range(100,117)))
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'full.npz';save(b,p,h);q=restore(data,a,p)
        x=state(b,h);y=state(q,h);assert all(np.array_equal(v,y[k]) for k,v in x.items())
        assert [q.load(addr,1) for _ in range(32)]==list(range(17,32))+list(range(100,117))
        assert q.load(addr+5,1)==0x60
        assert q.uart_input(b'Z');q.reg[17]=2;q.sbi();assert q.reg[10]==90;q.sbi();assert q.reg[10]==0xffffffff
        assert q.uart_input(b'abc');q.store(addr+2,1,3);assert q.rx.count==0 and q.load(addr+2,1)==0xc1
    cells=437560+b.rx.cells;assert cells<=500000
    return dict(status='PASS',guest_instructions=r['delta'],echo_hex=bytes(b.serial).hex(),
        rx_capacity=32,rx_cells=b.rx.cells,estimated_cells=cells,dlab=True,wraparound=True,
        backpressure=True,ablation=True,checkpoint_exact=True,sbi_getchar=True,polled=True)
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
        rx_received=b.rx.received,rx_reads=b.rx.reads,cache=cache.report(),no_guest_bytearray=b.ram is None,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=2000);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.steps<=50000:p.error('bounded steps')
    if a.image and not(a.input and a.checkpoint):p.error('input/output checkpoint required')
    r=run(a) if a.image else test(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

