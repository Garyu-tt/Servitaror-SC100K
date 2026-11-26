"""SC500K R04: privileged synaptic disk and visual MMIO.
No Linux driver yet. Existing kernel FDT is not modified on resume.
"""
import argparse,hashlib,json,os,struct,tempfile,time
from pathlib import Path
import numpy as np
from neural_paged_r29 import SynapticBoard,PagedCrossbar,board,load_neural,neural_snapshot
from neural_hippocampus_r27 import VisualRegion
from neural_register_r03 import validate
from neural_board_r11 import RAM,UART
from neural_irq_r09 import Timer
from neural_priv_r08 import GuestTrap
from neural_mmu_r10 import V,R,W,X,A,D,pte
from neural_rv32i_r01 import I,S
from neural_linux_r12 import fixture
from neural_resume_r13 import advance
from neural_tlb_r17 import PageCache
from neural_alu_r01 import oracle
from neural_math_r05 import oracle as moracle
DISK=0x30000000
GPU=0x31000000
class DeviceBoard(SynapticBoard):
    def attach_devices(self,disk=None):
        self.disk=disk if disk is not None else PagedCrossbar(64)
        self.gpu=VisualRegion();self.gpu_writes=0
    def device(self,pa,n):
        if n!=4 or pa%4:return None
        if DISK<=pa<DISK+self.disk.n*4:return 'disk'
        if pa in (GPU,GPU+4,GPU+8):return 'gpu'
        return None
    def physical(self,va,n,kind):
        ac,mis={'fetch':(1,0),'load':(5,4),'store':(7,6)}[kind]
        if n not in (1,2,4) or va%n:raise GuestTrap(mis,va)
        pa=va if self._atomic_physical else self.translate(va,kind)
        if self.in_ram(pa,n):return pa
        if kind!='fetch' and self.mode!=0:
            dev=self.device(pa,n)
            if dev=='disk' or (dev=='gpu' and ((kind=='store' and pa==GPU) or (kind=='load' and pa!=GPU))):return pa
            if UART<=pa<UART+8 and n==1:return pa
            if Timer.contains(pa) and n==4 and self.mode==3:return pa
        raise GuestTrap(ac,va)
    def load(self,a,n,signed=False):
        pa=self.physical(a,n,'load');dev=self.device(pa,n)
        if dev=='disk':return self.disk.read((pa-DISK)//4)
        if dev=='gpu':
            # Readout only: packed 64-bit pixel state in two words.
            pix=validate(self.gpu.state)[:,:,1].reshape(-1)
            off=0 if pa==GPU+4 else 32
            return sum(int(x==5)<<i for i,x in enumerate(pix[off:off+32]))
        if UART<=pa<UART+8:
            i=pa-UART
            return 0x60 if i==5 else 1 if i==2 else self.uart_regs[i]
        if Timer.contains(pa):return self.timer.read(pa)
        return self.phys_read(pa,n,signed)
    def store(self,a,n,v):
        pa=self.physical(a,n,'store');dev=self.device(pa,n)
        if dev=='disk':
            self.disk.write((pa-DISK)//4,v);return
        if dev=='gpu':
            self.gpu.rectangle(v&255,(v>>8)&255,(v>>16)&1);self.gpu_writes+=1;return
        if UART<=pa<UART+8:
            i=pa-UART
            if i==0 and not self.uart_regs[3]&0x80:self.serial.append(v&255)
            else:self.uart_regs[i]=v&255
            return
        if Timer.contains(pa):self.timer.write(pa,v);self.reservation=None;self.sync_pending();return
        self.phys_write(pa,n,v)

def attach(b,disk=None):
    b.__class__=DeviceBoard;b.attach_devices(disk);return b
def state(b,h):
    s=neural_snapshot(b,h)
    s.update(disk_weights=b.disk.weights,gpu_state=b.gpu.state,
        device_meta=np.frombuffer(json.dumps(dict(schema=1,disk_geometry=list(b.disk.weights.shape),
        disk_reads=b.disk.reads,disk_writes=b.disk.writes,gpu_writes=b.gpu_writes),sort_keys=True).encode(),dtype=np.uint8))
    return s
def save(b,path,h):
    p=Path(path);tmp=p.with_suffix(p.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**state(b,h));f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def restore(data,a,path,disk=None):
    b=attach(load_neural(data,a,path),disk)
    with np.load(path,allow_pickle=False) as z:
        if 'device_meta' not in z.files:return b
        m=json.loads(z['device_meta'].tobytes());w=z['disk_weights'];g=z['gpu_state']
        if m['schema']!=1 or m['disk_geometry']!=list(b.disk.weights.shape) or w.shape!=b.disk.weights.shape or w.dtype!=np.uint8:raise ValueError('disk geometry')
        for bank in w:
            if not np.all((bank==0x17)|(bank==0x71)):raise ValueError('disk coefficients')
        if g.shape!=(8,8,2,5) or g.dtype!=np.int8 or not np.all((g==0)|(g==1)):raise ValueError('GPU state')
        validate(g);b.disk.weights=w;b.gpu.state=g.copy()
        b.disk.reads=m['disk_reads'];b.disk.writes=m['disk_writes'];b.gpu_writes=m['gpu_writes']
    return b
def test(a):
    data=fixture();h=hashlib.sha256(data).hexdigest()
    b=board(data,a);b.__class__=SynapticBoard;b.install(PagedCrossbar(256));attach(b)
    root=RAM+0x1000;table=RAM+0x2000;code=RAM+0x3000
    va=0x40000000;dv=va+4096;gv=va+8192
    b.phys_write(root+256*4,4,pte(table,V))
    for j,pa in enumerate((code,DISK,GPU)):
        b.phys_write(table+j*4,4,pte(pa,V|R|W|(X if j==0 else 0)))
    # S-mode guest: disk write/read, draw, read low and high pixels,
    # then self-loop. The fixture never accesses host payload directly.
    words=[0x400010b7,I(0x13,2,0,0,42),S(2,1,2,0),I(3,3,2,1,0),
           0x40002237,0x000182b7,I(0x13,5,0,5,-386),S(2,4,5,0),
           0x000042b7,I(0x13,5,0,5,-964),S(2,4,5,0),I(3,6,2,4,4),I(3,7,2,4,8),0x0000006f]
    for i,w in enumerate(words):b.phys_write(code+4*i,4,w)
    b.put(0x180,0x80000000|(root>>12));b.mode=1;b.pc=va;PageCache(b)
    r=advance(b,len(words));assert r['delta']==len(words) and r['stop']=='budget'
    assert b.reg[3]==42 and b.disk.read(0)==42 and b.gpu_writes==2
    frame=b.gpu.frame();assert frame==['........','.######.','.#....#.','.#....#.','.#....#.','.#....#.','.######.','........']
    assert b.reg[6]==0x42427e00 and b.reg[7]==0x007e4242
    checks=0
    for f,cause in ((lambda:b.load(dv,1),5),(lambda:b.store(gv+4,4,0),7),
        (lambda:b.fetch(dv),12),(lambda:b.prepare_instruction_extension('LR.W',dv,0,0),5)):
        try:f()
        except GuestTrap as ex:assert ex.cause==cause;checks+=1
        else:raise AssertionError('permission failure')
    # U mappings permit translation but the peripheral bus still denies U.
    b.phys_write(table+4,4,pte(DISK,V|R|W|16|A|D));b.mode=0
    try:b.load(dv,4)
    except GuestTrap as ex:assert ex.cause==5;checks+=1
    else:raise AssertionError('U access')
    b.mode=1
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'full.npz';save(b,p,h);q=restore(data,a,p)
        assert all(np.array_equal(v,state(q,h)[k]) for k,v in state(b,h).items())
        assert q.disk.read(0)==42 and q.gpu.frame()==frame
    return dict(status='PASS',guest_instructions=r['delta'],guest_disk_value=42,guest_gpu_words=[hex(b.reg[6]),hex(b.reg[7])],
        gpu_frame=frame,sv32_device_access=True,permission_checks=checks,full_checkpoint_equal=True,
        ram_bytes=b.memory.n*4,disk_bytes=b.disk.n*4,estimated_cells=123110+b.memory.cells+b.disk.cells+b.gpu.cells)
def run(a):
    data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest()
    disk=None
    if a.disk:
        disk=PagedCrossbar(64);disk.restore(a.disk)
    b=restore(data,a,a.input,disk);cache=PageCache(b);old=b.c.run;oldm=b.math.run;checks=0
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
        disk_reads=b.disk.reads,disk_writes=b.disk.writes,gpu_writes=b.gpu_writes,cache=cache.report(),
        no_guest_bytearray=b.ram is None,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--disk',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=2000);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.steps<=10000:p.error('bounded steps')
    if a.image and not(a.input and a.checkpoint):p.error('input/output checkpoint required')
    r=run(a) if a.image else test(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

