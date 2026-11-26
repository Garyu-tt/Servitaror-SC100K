"""SC500K R03: hierarchical synaptic memory and privileged Linux board.

The host routes only selected firing banks/rows. Column read and recurrent
programming use R02's Q4 threshold equations. Bulk import/export is explicitly
a firmware/checkpoint boundary, not a guest memory operation.
"""
import argparse,hashlib,json,os,random,tempfile,time
from pathlib import Path
import numpy as np
from neural_hippocampus_r27 import Selector,VisualRegion,votes
from neural_crossbar_r28 import Crossbar
from neural_register_r03 import encode,decode,validate,fire
from neural_alu_r01 import Circuit,MASK,oracle
from neural_board_r11 import Board,RAM
from neural_mmu_r10 import MMUDecoder
from neural_linux_r12 import image_header,fixture,load_image
from neural_packed_r24 import setup
from neural_resume_r13 import restore,snapshot,advance,COUNTERS
from neural_tlb_r17 import PageCache

class PagedCrossbar(Crossbar):
    def __init__(self,banks,rows=1024,cols=64):
        self.banks=banks;self.rows=rows;self.cols=cols;self.n=banks*rows*cols;self.width=32
        self.bank=Selector(banks);self.row=Selector(rows);self.col=Selector(cols)
        self.weights=np.full((banks,rows,cols,32),0x17,dtype=np.uint8)
        self.cells=self.bank.cells+self.row.cells+self.col.cells+50*cols*32+640
        self.reads=self.writes=0
    @staticmethod
    def active(selector,index,ablate):
        r=np.flatnonzero(votes(selector.select(index,ablate))[:,1]==5)
        if len(r)!=1:raise RuntimeError('invalid route spikes')
        return int(r[0])
    def select(self,a,ablate=False):
        if type(a) is not int or not 0<=a<self.n:raise ValueError('address')
        page,offset=divmod(a,self.rows*self.cols)
        row,col=divmod(offset,self.cols)
        p=self.active(self.bank,page,ablate);r=self.active(self.row,row,ablate)
        return (p,r),votes(self.col.select(col,ablate))
    def import_bytes(self,offset,data):
        """Firmware-only bulk coefficient programming, word aligned."""
        if offset%4 or len(data)%4 or offset<0 or offset+len(data)>self.n*4:raise ValueError('bulk range')
        flat=self.weights.reshape(-1,32)
        raw=np.frombuffer(data,dtype=np.uint8)
        for i in range(0,len(raw),1<<20):
            part=raw[i:i+(1<<20)]
            bits=np.unpackbits(part,bitorder='little').reshape(-1,32)
            start=(offset+i)//4
            flat[start:start+len(bits)]=np.where(bits,0x71,0x17).astype(np.uint8)
    def export_bytes(self,offset,size):
        """Diagnostic-only exact decode, never called by guest memory bus."""
        if offset%4 or size%4 or offset<0 or offset+size>self.n*4:raise ValueError('bulk range')
        flat=self.weights.reshape(-1,32)
        result=bytearray()
        for i in range(offset//4,(offset+size)//4,1<<18):
            part=flat[i:min(i+(1<<18),(offset+size)//4)]
            if not np.all((part==0x17)|(part==0x71)):raise ValueError('invalid coefficient pair')
            result.extend(np.packbits(part==0x71,axis=1,bitorder='little').tobytes())
        return bytes(result)

class SynapticBoard(Board):
    memory=None
    def in_ram(self,a,n):
        if self.memory is None:return super().in_ram(a,n)
        return n in (1,2,4) and RAM<=a and a+n<=RAM+self.memory.n*4 and a%n==0
    def phys_read(self,a,n,signed=False):
        if self.memory is None:return super().phys_read(a,n,signed)
        if not self.in_ram(a,n):raise ValueError('physical access')
        v=self.memory.read((a-RAM)//4)
        if n!=4:
            v=self.alu('AND',self.alu('SRL',v,8*(a%4)),(1<<(8*n))-1)
            if signed:v=self.alu('SRA',self.alu('SLL',v,32-8*n),32-8*n)
        return v
    def phys_write(self,a,n,v):
        if self.memory is None:return super().phys_write(a,n,v)
        if not self.in_ram(a,n):raise ValueError('physical access')
        addr=(a-RAM)//4
        if n!=4:
            shift=8*(a%4);mask=((1<<(8*n))-1)<<shift
            v=self.alu('OR',self.alu('AND',self.memory.read(addr),(~mask)&MASK),self.alu('AND',self.alu('SLL',v,shift),mask))
        self.memory.write(addr,v)
        if self.reservation is not None and a<self.reservation+4 and self.reservation<a+n:self.reservation=None
    def install(self,memory):
        if self.memory is not None or len(self.ram)!=memory.n*4:raise ValueError('memory install')
        memory.import_bytes(0,self.ram)
        self.memory=memory;self.ram=None
    def boot(self,entry=RAM,fdt=RAM+0x100000):
        if self.memory is None:return super().boot(entry,fdt)
        raise RuntimeError('firmware boot must precede memory install')

def neural_snapshot(b,h):
    if b.ram is not None:raise ValueError('ordinary RAM is present')
    m=dict(schema=2,image_hash=h,ram_bytes=b.memory.n*4,geometry=list(b.memory.weights.shape),
        counters={k:getattr(b,k) for k in COUNTERS},timer=vars(b.timer).copy(),
        reservation=b.reservation,deadline=b.deadline,poweroff=b.poweroff,halted=b.halted,
        csr_addresses=sorted(b.csrs.state),fixed_addresses=sorted(b.csrs.fixed),
        reg_counts=[b.reg.reads,b.reg.writes],csr_counts=[b.csrs.reads,b.csrs.writes],
        pc_commits=b.counter.commits,control_calls=b.control.calls,
        math_counts=[b.math.calls,b.math.rounds,b.math.gates],
        memory_counts=[b.memory.reads,b.memory.writes])
    return dict(meta=np.frombuffer(json.dumps(m,sort_keys=True).encode(),dtype=np.uint8),
        weights=b.memory.weights,regs=b.reg._state,pc=b.counter.state,mode=b.priv.state,
        csr=np.stack([b.csrs.state[x] for x in m['csr_addresses']]),
        fixed=np.stack([b.csrs.fixed[x] for x in m['fixed_addresses']]),
        uart=np.frombuffer(bytes(b.uart_regs),dtype=np.uint8),serial=np.frombuffer(bytes(b.serial),dtype=np.uint8))

def save_neural(b,path,h):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**neural_snapshot(b,h));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)

def load_neural(data,a,path):
    h=hashlib.sha256(data).hexdigest();b=board(data,a)
    b.__class__=SynapticBoard;b.memory=PagedCrossbar(256);b.ram=None
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['meta'].tobytes())
        if m['schema']!=2 or m['image_hash']!=h or m['ram_bytes']!=b.memory.n*4 or m['geometry']!=list(b.memory.weights.shape):raise ValueError('checkpoint identity')
        if m['csr_addresses']!=sorted(b.csrs.state) or m['fixed_addresses']!=sorted(b.csrs.fixed):raise ValueError('CSR layout')
        shapes={'weights':b.memory.weights.shape,'regs':(32,32,2,5),'pc':(32,2,5),'mode':(2,2,5),
                'csr':(len(b.csrs.state),32,2,5),'fixed':(len(b.csrs.fixed),32,2,5),'uart':(8,)}
        for k,shape in shapes.items():
            if z[k].shape!=shape:raise ValueError('checkpoint shape '+k)
        for k in ('regs','pc','mode','csr','fixed'):
            if z[k].dtype!=np.int8 or not np.all((z[k]==0)|(z[k]==1)):raise ValueError('invalid spike state')
            validate(z[k])
        if np.any(z['regs'][0,:,1]):raise ValueError('x0')
        w=z['weights']
        if w.dtype!=np.uint8:raise ValueError('coefficient dtype')
        for bank in w:
            if not np.all((bank==0x17)|(bank==0x71)):raise ValueError('invalid coefficient pair')
        b.memory.weights=w
        b.reg._state=z['regs'].copy();b.counter.state=z['pc'].copy();b.priv.state=z['mode'].copy()
        b.csrs.state={x:r.copy() for x,r in zip(m['csr_addresses'],z['csr'])}
        b.csrs.fixed={x:r.copy() for x,r in zip(m['fixed_addresses'],z['fixed'])}
        b.uart_regs[:]=z['uart'].tobytes();b.serial[:]=z['serial'].tobytes()
        for k in COUNTERS:setattr(b,k,m['counters'][k])
        for k in vars(b.timer):setattr(b.timer,k,m['timer'][k])
        for k in ('reservation','deadline','poweroff','halted'):setattr(b,k,m[k])
        b.reg.reads,b.reg.writes=m['reg_counts'];b.csrs.reads,b.csrs.writes=m['csr_counts']
        b.counter.commits=m['pc_commits'];b.control.calls=m['control_calls']
        b.math.calls,b.math.rounds,b.math.gates=m['math_counts']
        b.memory.reads,b.memory.writes=m['memory_counts']
    return b

def board(data,a,cp=None):
    ref,_=load_image(data,64)
    e,_=setup(ref,a)
    if cp:restore(ref,cp,hashlib.sha256(data).hexdigest())
    e.attach(ref)
    return ref

def verify(data,a,cp=None,steps=512):
    h=hashlib.sha256(data).hexdigest();ref=board(data,a,cp);b=board(data,a,cp)
    # Both boards have identical checkpoint state. Convert one to
    # synaptic storage, then discard its conventional RAM entirely.
    b.__class__=SynapticBoard
    b.install(PagedCrossbar(256))
    assert b.ram is None
    assert b.memory.export_bytes(0,64<<20)==bytes(ref.ram)
    PageCache(ref);PageCache(b)
    orig=b.c.run;checks=0
    def audited(op,x,y,ablate=False):
        nonlocal checks
        v=orig(op,x,y,ablate)
        if not ablate:assert v[0]==oracle(op,x,y);checks+=1
        return v
    b.c.run=audited
    rr=advance(ref,steps);nr=advance(b,steps)
    assert rr['delta']==nr['delta'] and rr['pc']==nr['pc'] and rr['stop']==nr['stop'] and rr['fault']==nr['fault']
    for k in ('reg','counter','priv','csrs'):
        if k=='reg':assert np.array_equal(ref.reg._state,b.reg._state)
        elif k in ('counter','priv'):assert np.array_equal(getattr(ref,k).state,getattr(b,k).state)
        else:
            assert all(np.array_equal(ref.csrs.state[x],b.csrs.state[x]) for x in ref.csrs.state)
            assert all(np.array_equal(ref.csrs.fixed[x],b.csrs.fixed[x]) for x in ref.csrs.fixed)
    assert bytes(ref.ram)==b.memory.export_bytes(0,64<<20)
    assert ref.serial==b.serial and ref.uart_regs==b.uart_regs and vars(ref.timer)==vars(b.timer)
    assert ref.reservation==b.reservation and ref.deadline==b.deadline
    assert ref.steps==b.steps and ref.traps==b.traps and ref.interrupts==b.interrupts
    evidence={}
    if getattr(a,'checkpoint',None):
        save_neural(b,a.checkpoint,h)
        q=load_neural(data,a,a.checkpoint)
        sx=neural_snapshot(b,h);sy=neural_snapshot(q,h)
        assert all(np.array_equal(sx[k],sy[k]) for k in sx)
        # Rebuild empty derived TLB on both instances before replay.
        b.translate=b.__class__.translate.__get__(b);b.phys_write=b.__class__.phys_write.__get__(b);b.special_step=b.__class__.special_step.__get__(b)
        PageCache(b);PageCache(q)
        rb=advance(b,128);rq=advance(q,128)
        assert rb['pc']==rq['pc'] and rb['delta']==rq['delta']
        sx=neural_snapshot(b,h);sy=neural_snapshot(q,h)
        assert all(np.array_equal(sx[k],sy[k]) for k in sx)
        evidence=dict(synaptic_checkpoint_exact=True,resume_replay_instructions=rb['delta'],resumed_retired=rb['retired'],
            checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),checkpoint_bytes=a.checkpoint.stat().st_size)
    return dict(reference=rr,synaptic=nr,full_ram_equal=True,register_csr_pc_equal=True,peripherals_equal=True,
        ram_reads=b.memory.reads,ram_writes=b.memory.writes,oracle_checks=checks,ram_cells=b.memory.cells,**evidence)

def main(a):
    t=time.perf_counter();disk=PagedCrossbar(64);rng=random.Random(310503)
    addresses=[0,63,64,65535,65536,disk.n-1]+[rng.randrange(disk.n) for _ in range(128)]
    expected={}
    for x in addresses:
        v=rng.getrandbits(32);disk.write(x,v);expected[x]=v;assert disk.read(x)==v
    for x,v in expected.items():assert disk.read(x)==v
    for f in (lambda:disk.read(0,True),lambda:disk.write(0,1,True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'disk.npz';disk.save(p);d=PagedCrossbar(64);d.restore(p)
        assert np.array_equal(d.weights,disk.weights);del d
        for x,v in expected.items():assert disk.read(x)==v
    data=a.image.read_bytes() if a.image else fixture()
    if a.image:
        # Store real kernel bytes as synapses, verify exact recovery.
        pad=data+b'\0'*((-len(data))%4);disk.import_bytes(0,pad)
        assert disk.export_bytes(0,len(pad))==pad
        if a.disk_checkpoint:
            disk.save(a.disk_checkpoint);restored=PagedCrossbar(64);restored.restore(a.disk_checkpoint);disk=restored
        # The firmware retrieves the actual kernel image from synaptic coefficients.
        recovered=disk.export_bytes(0,len(pad))[:len(data)]
        assert hashlib.sha256(recovered).digest()==hashlib.sha256(data).digest()
        data=recovered
    r=verify(data,a,a.input,a.steps)
    budget=123110+r['ram_cells']+disk.cells+VisualRegion.cells
    assert budget<=500000
    return dict(status='PASS',ram_bytes=64<<20,disk_bytes=16<<20,estimated_cells=budget,
        packed_memory_bytes=(64+16)*(1<<20)*8,logical_q4_coefficients=(64+16)*(1<<20)*16,
        disk_checks=len(addresses)+2*len(expected),disk_persistence=True,disk_ablation_cases=2,
        kernel_stored_on_synaptic_disk=bool(a.image),firmware_image_from_synaptic_disk=bool(a.image),
        kernel_disk_checkpoint_sha256=hashlib.sha256(a.disk_checkpoint.read_bytes()).hexdigest() if a.image and a.disk_checkpoint else None,
        real_linux_image=bool(a.image),linux_booted=False,
        checkpoint_import=bool(a.input),**r,seconds=round(time.perf_counter()-t,3))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--steps',type=int,default=128);p.add_argument('--output',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--disk-checkpoint',type=Path);a=p.parse_args()
    if not 1<=a.steps<=4096:p.error('bounded test')
    if a.input and not a.image:p.error('checkpoint requires image')
    r=main(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

