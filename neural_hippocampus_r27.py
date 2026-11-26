"""SC500K R01: hippocampus-inspired memory and visual-region prototype.

All addressed reads traverse weighted selectors and convergent spike gates.
Persistent disk payload is Q4 synaptic coefficients, not a byte array.
This is engineered circuitry, not a trained biological connectome.
"""
import argparse, hashlib, json, os, random, struct, tempfile, time
from pathlib import Path
import numpy as np
from neural_register_r03 import encode, decode, validate, fire
from neural_alu_r01 import Circuit, MASK
from neural_decoder_r02 import Decoder
from neural_control_r04 import ControlCPU
from neural_rv32i_r01 import I,S
from neural_packed_r24 import accelerate

def votes(x):return x.sum(axis=-1,dtype=np.int32)
def AND(a,b):return fire(4*(votes(a)+votes(b)))
def OR(a,b):return fire(7*(votes(a)+votes(b)))

class Selector:
    def __init__(self,n):
        if n<4 or n&(n-1):raise ValueError('power-of-two address space')
        self.n=n;self.bits=n.bit_length()-1
        self.low=self.bits//2;self.high=self.bits-self.low
        self.weights=[self.matrix(self.low),self.matrix(self.high)]
        self.cells=10*(self.bits+(1<<self.low)+(1<<self.high)+n)
    @staticmethod
    def matrix(bits):
        w=np.zeros((1<<bits,2,bits,2),dtype=np.int8)
        gain=(33+5*bits-1)//(5*bits)
        for i in range(1<<bits):
            for j in range(bits):
                v=i>>j&1
                w[i,1,j,v]=gain;w[i,1,j,1-v]=-7;w[i,0,j,1-v]=7
        return w
    def select(self,address,ablate=False):
        if type(address) is not int or not 0<=address<self.n:raise ValueError('address')
        src=votes(encode(address,self.bits))
        a,b=[fire(np.einsum('rabi,bi->ra',w,s,dtype=np.int32)*int(not ablate))
             for w,s in zip(self.weights,(src[:self.low],src[self.low:]))]
        validate(a);validate(b)
        x=votes(a)[None,:,:];y=votes(b)[:,None,:]
        rails=fire((x+y)*np.array([7,4],dtype=np.int32)).reshape(self.n,2,5)
        v=validate(rails)
        if np.count_nonzero(v[:,1]==5)!=1:raise RuntimeError('invalid address selection')
        return rails

class RecurrentRAM:
    def __init__(self,n=256,width=32):
        self.selector=Selector(n);self.n=n;self.width=width
        self.state=np.repeat(encode(0,width)[None,...],n,axis=0)
        self.reads=self.writes=0
        self.cells=self.selector.cells+20*width+40*n*width
    def read(self,address,ablate=False):
        sel=votes(self.selector.select(address,ablate))[:,1]
        g=fire(4*(sel[:,None,None]+votes(self.state)))
        v=decode(fire(7*g.sum(axis=(0,3),dtype=np.int32)))
        self.reads+=1;return v
    def write(self,address,value,ablate=False):
        sel=votes(self.selector.select(address,ablate))
        d=votes(encode(value,self.width))
        hold=fire(4*(votes(self.state)+sel[:,0,None,None]))
        put=fire(4*(d[None,:,:]+sel[:,1,None,None]))
        new=fire(7*(votes(hold)+votes(put)));validate(new)
        self.state=new;self.writes+=1
    def reset(self):
        self.state=np.repeat(encode(0,self.width)[None,...],self.n,axis=0)

class SynapticDisk:
    def __init__(self,n=32,width=32):
        self.n=n;self.width=width;self.selector=Selector(n)
        self.weights=np.repeat(np.where(encode(0,width)[None,...]==1,7,1).astype(np.int8),n,axis=0)
        # One coefficient per synchronized 5-source bundle: conceptual
        # five replicated target neurons share it. No integer payload mirror.
        self.cells=self.selector.cells+20*width+40*n*width
        self.reads=self.writes=0
    def read(self,address,ablate=False):
        sel=votes(self.selector.select(address,ablate))[:,1]
        current=(self.weights*sel[:,None,None,None]).sum(axis=-1,dtype=np.int32)//5
        # Each of the five source spikes has its Q4 coefficient; repeated
        # coefficients mean this is the current per target neuron.
        g=fire(current)
        v=decode(fire(7*g.sum(axis=(0,3),dtype=np.int32)))
        self.reads+=1;return v
    def program(self,address,value,ablate=False):
        sel=votes(self.selector.select(address,ablate))
        old=fire(self.weights.sum(axis=-1,dtype=np.int32))
        validate(old)
        d=votes(encode(value,self.width))
        hold=fire(4*(votes(old)+sel[:,0,None,None]))
        put=fire(4*(d[None,:,:]+sel[:,1,None,None]))
        next_state=fire(7*(votes(hold)+votes(put)));validate(next_state)
        # Local plastic programming: the active rail potentiates to 7,
        # inactive rail depresses to 1. No address-indexed write.
        self.weights=np.where(next_state==1,7,1).astype(np.int8)
        self.writes+=1
    def save(self,path):
        path=Path(path);tmp=path.with_suffix('.tmp')
        with tmp.open('wb') as f:np.savez_compressed(f,weights=self.weights);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    def restore(self,path):
        with np.load(path,allow_pickle=False) as z:
            w=z['weights']
            if w.shape!=self.weights.shape or w.dtype!=np.int8 or not np.all((w==1)|(w==7)):
                raise ValueError('invalid Q4 checkpoint')
            if not np.all(w==w[...,:1]):raise ValueError('replica mismatch')
            validate(fire(w.sum(axis=-1,dtype=np.int32)))
            self.weights=w.copy()

class VisualRegion:
    """8 x 8 monochrome parallel raster: masks in, recurrent pixels out."""
    cells=10*(16+64*5+1)
    def __init__(self):self.state=np.broadcast_to(encode(0,1)[0],(8,8,2,5)).copy()
    def rectangle(self,rows,cols,color,ablate=False):
        a=votes(encode(rows,8))[:,None,:];b=votes(encode(cols,8))[None,:,:]
        sel=fire((a+b)*np.array([7,4],dtype=np.int32)*int(not ablate))
        v=validate(sel);data=votes(encode(color,1))[0]
        hold=fire(4*(votes(self.state)+v[:,:,0,None]))
        put=fire(4*(data+v[:,:,1,None]))
        nxt=fire(7*(votes(hold)+votes(put)));validate(nxt);self.state=nxt
    def frame(self):
        v=validate(self.state)
        return [''.join('#' if x==5 else '.' for x in row) for row in v[:,:,1]]

class HippocampalCPU(ControlCPU):
    def __init__(self,c,d,memory,disk,gpu):
        super().__init__(c,d,b'')
        self.ram=None # Discard base-class placeholder before execution.
        self.memory=memory;self.disk=disk;self.gpu=gpu
    def access(self,a,n):
        if n not in (1,2,4) or a<0 or a%n or a+n>self.memory.n*4:raise RuntimeError('neural RAM bus fault')
    def load(self,a,n,signed=False):
        if n==4 and 0x2000<=a<0x2000+self.disk.n*4 and a%4==0:return self.disk.read((a-0x2000)//4)
        self.access(a,n);v=self.memory.read(a//4)
        if n!=4:
            v=self.alu('AND',self.alu('SRL',v,8*(a%4)),(1<<(n*8))-1)
            if signed:v=self.alu('SRA',self.alu('SLL',v,32-8*n),32-8*n)
        return v
    def store(self,a,n,v):
        if a==0x1000 and n==4:
            self.gpu.rectangle(v&255,(v>>8)&255,(v>>16)&1);return
        if n==4 and 0x2000<=a<0x2000+self.disk.n*4 and a%4==0:
            self.disk.program((a-0x2000)//4,v);return
        self.access(a,n)
        if n!=4:
            shift=8*(a%4);mask=((1<<(8*n))-1)<<shift
            old=self.memory.read(a//4)
            v=self.alu('OR',self.alu('AND',old,(~mask)&MASK),self.alu('AND',self.alu('SLL',v,shift),mask))
        self.memory.write(a//4,v)

def run(args):
    t=time.perf_counter();ram=RecurrentRAM();disk=SynapticDisk();gpu=VisualRegion()
    # Full core budget estimate includes existing RV32IMA/CSR/MMU paths,
    # even though this fixture uses RV32I only.
    budget=123110+ram.cells+disk.cells+gpu.cells
    assert budget<=500000,budget
    rng=random.Random(310500);ref=[0]*ram.n
    for i in range(ram.n):assert np.argmax(votes(ram.selector.select(i))[:,1])==i
    for _ in range(128):
        i=rng.randrange(ram.n);v=rng.getrandbits(32);ram.write(i,v);ref[i]=v
        assert ram.read(i)==v
    for i,v in enumerate(ref):assert ram.read(i)==v
    original=ram.state.copy()
    for f in (lambda:ram.read(7,True),lambda:ram.write(7,42,True),lambda:disk.read(0,True),lambda:disk.program(0,42,True),lambda:gpu.rectangle(1,1,1,True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    assert np.array_equal(original,ram.state)
    ram.reset()
    words=[0x123450b7,I(0x13,1,0,1,0x678),I(0x13,2,0,0,512),S(2,2,1,0),
           I(3,3,0,2,0),I(3,4,4,2,0),I(0x13,5,0,0,-1),S(0,2,5,1),
           I(3,6,2,2,0),S(1,2,5,2),I(3,7,2,2,0),I(0x13,9,0,0,42),
           S(2,2,9,4),I(3,10,2,2,4),
           0x000015b7,0x00018637,I(0x13,12,0,12,-386),S(2,11,12,0),
           0x00004637,I(0x13,12,0,12,-964),S(2,11,12,0),
           0x000026b7,S(2,13,9,124),I(3,14,2,13,124),0x00100073]
    for i,w in enumerate(words):disk.program(i,w)
    with tempfile.TemporaryDirectory() as td:
        p=Path(td)/'synapses.npz';disk.save(p);sha=hashlib.sha256(p.read_bytes()).hexdigest()
        fresh=SynapticDisk();fresh.restore(p);assert np.array_equal(disk.weights,fresh.weights)
        for i,w in enumerate(words):assert fresh.read(i)==w;ram.write(i,fresh.read(i))
        assert fresh.read(31)==0
    c=Circuit().compile();d=Decoder().compile();cpu=HippocampalCPU(c,d,ram,fresh,gpu);accelerate(cpu,args)
    while not cpu.halted and cpu.steps<100:cpu.step()
    assert cpu.halted and cpu.steps==len(words) and cpu.ram is None
    assert [cpu.reg[i] for i in (3,4,6,7,10)]==[0x78,0x78,0x1234ff78,0xffffff78,42]
    assert ram.read(128)==0xffffff78 and ram.read(129)==42
    assert cpu.reg[14]==fresh.read(31)==42
    assert cpu.load(513,1,True)==0xffffffff and cpu.load(514,2,False)==65535
    if args.disk_checkpoint:
        fresh.save(args.disk_checkpoint);restored=SynapticDisk();restored.restore(args.disk_checkpoint);assert restored.read(31)==42
        sha=hashlib.sha256(args.disk_checkpoint.read_bytes()).hexdigest()
    frame=gpu.frame();assert frame==['........','.######.','.#....#.','.#....#.','.#....#.','.#....#.','.######.','........']
    result=dict(status='PASS',estimated_full_core_cells=123110,ram_cells=ram.cells,disk_cells=disk.cells,visual_cells=gpu.cells,combined_estimated_cells=budget,ram_bytes=1024,disk_bytes=128,framebuffer_bits=64,address_checks=256,ram_read_checks=384,ablation_cases=5,guest_instructions=cpu.steps,guest_alu_calls=cpu.alu_calls,guest_instruction_fetch_from_neural_ram=True,boot_program_from_synaptic_disk=True,guest_disk_mmio=True,guest_graphics_mmio=True,no_guest_bytearray=True,frame=frame,disk_checkpoint_sha256=sha,seconds=round(time.perf_counter()-t,3),linux_booted=False)
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--output',type=Path);p.add_argument('--disk-checkpoint',type=Path);a=p.parse_args()
    r=run(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

