"""SC500K R02: dense, exact Q4 synaptic crossbar memory.

A row is activated by neural address selection, then a column by neural
read/hold/write gates. Sparse row routing skips mathematically silent
sources; the active row index is derived from spikes, not guest address.
Payload is packed Q4 coefficient pairs (1/7), not guest bytes.
"""
import argparse,hashlib,json,os,random,tempfile,time
from pathlib import Path
import numpy as np
from neural_hippocampus_r27 import Selector,VisualRegion,HippocampalCPU,votes
from neural_register_r03 import encode,decode,validate,fire
from neural_alu_r01 import Circuit,MASK
from neural_decoder_r02 import Decoder
from neural_rv32i_r01 import I,S
from neural_packed_r24 import accelerate

class Crossbar:
    def __init__(self,rows,cols=64):
        self.rows=rows;self.cols=cols;self.n=rows*cols;self.width=32
        self.row=Selector(rows);self.col=Selector(cols)
        # One byte per pair of Q4 coefficients; zero rail 7, one rail 1.
        self.weights=np.full((rows,cols,32),0x17,dtype=np.uint8)
        # Readout, read gating, hold/write, feedback groups shared across
        # selected rows; no per-bit persistent spike cell. Five bundles
        # per column bit, with two rails of five logical cells each.
        self.cells=self.row.cells+self.col.cells+50*cols*32+640
        self.reads=self.writes=0
    def select(self,a,ablate=False):
        if type(a) is not int or not 0<=a<self.n:raise ValueError('address')
        r=self.row.select(a//self.cols,ablate)
        c=self.col.select(a%self.cols,ablate)
        # Event routing: only a firing row can drive synapses.
        active=np.flatnonzero(votes(r)[:,1]==5)
        if len(active)!=1:raise RuntimeError('invalid row spikes')
        return int(active[0]),votes(c)
    @staticmethod
    def activate(w):
        currents=5*np.stack((w&15,w>>4),axis=-1).astype(np.int32)
        state=fire(currents);validate(state);return state
    def read(self,a,ablate=False):
        r,c=self.select(a,ablate)
        s=self.activate(self.weights[r])
        gated=fire(4*(votes(s)+c[:,1,None,None]))
        v=decode(fire(7*gated.sum(axis=(0,3),dtype=np.int32)))
        self.reads+=1;return v
    def write(self,a,v,ablate=False):
        r,c=self.select(a,ablate)
        old=self.activate(self.weights[r]);d=votes(encode(v&MASK,32))
        hold=fire(4*(votes(old)+c[:,0,None,None]))
        put=fire(4*(d[None,:,:]+c[:,1,None,None]))
        nxt=fire(7*(votes(hold)+votes(put)))
        q=validate(nxt)
        # Local plastic programming of the active row. All its columns
        # receive the gated update, including explicit hold paths.
        packed=np.where(q[:,:,1]==5,0x71,0x17).astype(np.uint8)
        self.weights[r]=packed;self.writes+=1
    program=write
    def reset(self):
        self.weights.fill(0x17)
    def save(self,path):
        path=Path(path);tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:
            np.savez_compressed(f,weights=self.weights);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    def restore(self,path):
        with np.load(path,allow_pickle=False) as z:
            w=z['weights']
            if w.shape!=self.weights.shape or w.dtype!=np.uint8:raise ValueError('shape/dtype')
            # Validate in bounded chunks before changing authoritative state.
            for row in w:
                if not np.all((row==0x17)|(row==0x71)):raise ValueError('invalid coefficient pair')
            self.weights=w.copy()

class DenseCPU(HippocampalCPU):
    GPU=0x10000000;DISK=0x20000000
    def fetch(self,pc):
        self.access(pc,4);return self.memory.read(pc//4)
    def load(self,a,n,signed=False):
        if n==4 and self.DISK<=a<self.DISK+self.disk.n*4 and a%4==0:return self.disk.read((a-self.DISK)//4)
        self.access(a,n);v=self.memory.read(a//4)
        if n!=4:
            v=self.alu('AND',self.alu('SRL',v,8*(a%4)),(1<<(n*8))-1)
            if signed:v=self.alu('SRA',self.alu('SLL',v,32-8*n),32-8*n)
        return v
    def store(self,a,n,v):
        if a==self.GPU and n==4:
            self.gpu.rectangle(v&255,(v>>8)&255,(v>>16)&1);return
        if n==4 and self.DISK<=a<self.DISK+self.disk.n*4 and a%4==0:
            self.disk.program((a-self.DISK)//4,v);return
        self.access(a,n)
        if n!=4:
            shift=8*(a%4);mask=((1<<(8*n))-1)<<shift
            old=self.memory.read(a//4)
            v=self.alu('OR',self.alu('AND',old,(~mask)&MASK),self.alu('AND',self.alu('SLL',v,shift),mask))
        self.memory.write(a//4,v)

def run(args):
    t=time.perf_counter();ram=Crossbar(8192);disk=Crossbar(4096);gpu=VisualRegion()
    budget=123110+ram.cells+disk.cells+gpu.cells
    assert budget<=500000,budget
    rng=random.Random(310502);ref={}
    # Verify both row and column selectors exhaustively, independently of
    # guest payload. Test random addresses and cross-row boundaries.
    for sel in (ram.row,ram.col,disk.row,disk.col):
        for i in range(sel.n):
            assert np.flatnonzero(votes(sel.select(i))[:,1]==5).tolist()==[i]
    addresses=[0,1,63,64,65,ram.n-2,ram.n-1]+[rng.randrange(ram.n) for _ in range(256)]
    for a in addresses:
        v=rng.getrandbits(32);ram.write(a,v);ref[a]=v
        assert ram.read(a)==v
    for a,v in ref.items():assert ram.read(a)==v
    for _ in range(128):
        a=rng.randrange(ram.n);assert ram.read(a)==ref.get(a,0)
    before=hashlib.sha256(ram.weights.tobytes()).hexdigest()
    for f in (lambda:ram.read(7,True),lambda:ram.write(7,42,True),lambda:disk.program(0,42,True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    assert hashlib.sha256(ram.weights.tobytes()).hexdigest()==before
    # Program boots from synaptic disk, accesses a high RAM address,
    # persists a word to the end of disk, and renders via neural MMIO.
    ram.reset()
    words=[0x123450b7,I(0x13,1,0,1,0x678),0x001ff137,I(0x13,2,0,2,-4),
           S(2,2,1,0),I(3,3,2,2,0),I(0x13,5,0,0,-1),S(0,2,5,1),I(3,6,2,2,0),
           0x100005b7,0x00018637,I(0x13,12,0,12,-386),S(2,11,12,0),
           0x00004637,I(0x13,12,0,12,-964),S(2,11,12,0),
           0x201006b7,I(0x13,13,0,13,-4),S(2,13,6,0),I(3,14,2,13,0),0x00100073]
    for i,w in enumerate(words):disk.program(i,w)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'synapses.npz';disk.save(p)
        fresh=Crossbar(4096);fresh.restore(p)
        for i,w in enumerate(words):assert fresh.read(i)==w;ram.write(i,fresh.read(i))
    cpu=DenseCPU(Circuit().compile(),Decoder().compile(),ram,fresh,gpu);accelerate(cpu,args)
    while not cpu.halted and cpu.steps<100:cpu.step()
    assert cpu.halted and cpu.steps==len(words) and cpu.ram is None
    assert cpu.reg[3]==0x12345678 and cpu.reg[6]==cpu.reg[14]==0x1234ff78
    assert fresh.read(fresh.n-1)==0x1234ff78 and ram.read((0x1ff000-4)//4)==0x1234ff78
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'synapses.npz';fresh.save(p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
        restored=Crossbar(4096);restored.restore(p);assert restored.read(restored.n-1)==0x1234ff78
        # Corrupt pair must fail closed, retaining old state.
        before=hashlib.sha256(restored.weights.tobytes()).hexdigest()
        bad=restored.weights.copy();bad[0,0,0]=0x77
        np.savez_compressed(p,weights=bad)
        try:restored.restore(p)
        except ValueError:pass
        else:raise AssertionError('invalid coefficient accepted')
        assert hashlib.sha256(restored.weights.tobytes()).hexdigest()==before
    frame=gpu.frame();assert frame==['........','.######.','.#....#.','.#....#.','.#....#.','.#....#.','.######.','........']
    return dict(status='PASS',estimated_full_core_cells=123110,ram_cells=ram.cells,disk_cells=disk.cells,
        visual_cells=gpu.cells,combined_estimated_cells=budget,ram_bytes=ram.n*4,disk_bytes=disk.n*4,
        packed_synapse_bytes=ram.weights.nbytes+disk.weights.nbytes,logical_q4_coefficients=2*(ram.weights.size+disk.weights.size),
        replicated_synapses_if_expanded=50*(ram.weights.size+disk.weights.size),
        selector_checks=ram.rows+ram.cols+disk.rows+disk.cols,ram_read_checks=len(addresses)+len(ref)+128,
        ablation_cases=3,corrupt_pair_rejected=True,guest_instructions=cpu.steps,guest_alu_calls=cpu.alu_calls,
        guest_high_ram=True,guest_disk_mmio=True,guest_graphics_mmio=True,no_guest_bytearray=True,frame=frame,
        disk_checkpoint_sha256=sha,seconds=round(time.perf_counter()-t,3),linux_booted=False)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--output',type=Path);a=p.parse_args()
    r=run(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

