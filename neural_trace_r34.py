"""R08: capture actual native graph spikes at instruction boundaries.

No synthetic coordinates. Trace covers the three explicit fixed feedforward
graphs only, not the implicit recurrent/crossbar circuits. A boundary is
one guest step, not an individual GENESIS simulation tick.
"""
import argparse,ctypes,hashlib,json,random,time
from pathlib import Path
from native_bundle import native_library
import numpy as np
import neural_packed_r24 as packed
from neural_io_r33 import restore,save,upgrade,state
from neural_paged_r29 import board,SynapticBoard,PagedCrossbar
from neural_devices_r30 import attach
from neural_framebuffer_r32 import upgrade as visual_upgrade
from neural_linux_r12 import fixture
from neural_tlb_r17 import PageCache
from neural_alu_r01 import OPS,oracle
from neural_math_r05 import oracle as moracle

class TracePacked(packed.Packed):
    def __init__(self,n,edges,level,ports,outputs,args):
        # Generate the same native source as R24, then add a read-only
        # representative-state export at the actual evaluation boundary.
        super().__init__(n,edges,level,ports,outputs,args)
        self.n=n;self.edges=np.array(edges,dtype=np.int32);self.level=np.array([level[i] for i in range(n)],dtype=np.int16)
        # Reconstruct the exact R24 generated filename from source candidates
        # by matching native output dimensions and representative count.
        # The base instance's CDLL filename is the authoritative source path.
        source=Path(self.lib._name).with_suffix('.c').read_text()
        needle=f'for(int j=0;j<{n//5};j++)count+=r[j];return 5*count;'
        assert source.count(needle)==1
        source=source.replace('u32 *out){','u32 *out,unsigned char *trace){')
        source=source.replace(needle,f'for(int j=0;j<{n//5};j++){{trace[j]=r[j];count+=r[j];}}return 5*count;')
        # Trace must also reflect failed/ablated evaluation, before output validation.
        marker='out[0]=0;'
        assert source.count(marker)==1
        source=source.replace(marker,f'for(int j=0;j<{n//5};j++)trace[j]=r[j];'+marker)
        import subprocess
        key=hashlib.sha256(source.encode()).hexdigest()[:20]
        root=args.build_dir;c=root/(key+'.c');o=root/(key+'.o');so=root/(key+'.so');c.write_text(source)
        so=native_library(c,so,args.cc,args.ld,opt=2)
        self.lib=ctypes.CDLL(str(so.resolve()));self.fn=self.lib.evaluate
        ptr=ctypes.POINTER(ctypes.c_uint32);self.fn.argtypes=[ptr,ctypes.c_int,ptr,ctypes.POINTER(ctypes.c_ubyte)];self.fn.restype=ctypes.c_int
        self.trace=(ctypes.c_ubyte*(n//5))();self.sink=None
    def run(self,word,ablate=False):
        for j in range(self.niw):self.ib[j]=(word>>(32*j))&0xffffffff
        count=self.fn(self.ib,int(not ablate),self.ob,self.trace)
        bits=np.ctypeslib.as_array(self.trace)
        if count>=0:assert count==int(bits.sum())*5
        if self.sink:self.sink(self,bits,ablate,count)
        if count<0:raise RuntimeError('invalid dual-rail readout')
        return sum(int(self.ob[j])<<(32*j) for j in range(self.now)),count

class Recorder:
    def __init__(self,b,a):
        self.b=b;self.graphs=[];self.events=[];self.boundary=0;self.enabled=False
        old=packed.Packed
        def factory(*x,**kw):
            p=TracePacked(*x,**kw);self.graphs.append(p);return p
        packed.Packed=factory
        try:self.old,_=packed.accelerate(b,a)
        finally:packed.Packed=old
        self.names=('alu','decoder','control')
        assert len(self.graphs)==3
        for g in self.graphs:g.sink=self.record
    def record(self,g,bits,ablate,count):
        if not self.enabled:return
        i=self.graphs.index(g)
        self.events.append(dict(boundary=self.boundary,ordinal=len(self.events),region=self.names[i],
            rail_bits=np.packbits(bits,bitorder='little').tobytes().hex(),spikes=int(bits.sum())*5,
            ablated=bool(ablate),valid=count>=0))
    def graph(self,path):
        arrays={}
        for name,g in zip(self.names,self.graphs):
            arrays[name+'_edges']=g.edges;arrays[name+'_levels']=g.level
        np.savez_compressed(path,**arrays)
        return dict(regions=[dict(name=n,neurons=g.n,edges=len(g.edges),representatives=g.n//5) for n,g in zip(self.names,self.graphs)],
            coordinates_available=False,coordinates_status='GENESIS spatial coordinates absent; do not invent anatomy',
            graph_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            scope='explicit ALU, decoder, control only; no RAM, disk, GPU, register or CSR activation trace')
    def step(self):
        self.boundary=self.b.boundaries+1
        self.enabled=True
        try:self.b.step()
        finally:self.enabled=False
    def sample(self,path,steps):
        rows=[];start=time.perf_counter();old=self.b.steps
        initial=dict(boundary=self.b.boundaries,retired=old,pc=hex(self.b.pc),uart_hex=bytes(self.b.serial).hex(),framebuffer_sha256=hashlib.sha256(self.b.scanout()).hexdigest())
        for i in range(steps):
            before=len(self.events);serial_before=len(self.b.serial);traps=self.b.traps
            self.step()
            rows.append(dict(boundary=self.boundary,retired=self.b.steps,pc=hex(self.b.pc),
                events=[before,len(self.events)],uart_hex=bytes(self.b.serial[serial_before:]).hex(),
                framebuffer_sha256=hashlib.sha256(self.b.scanout()).hexdigest()))
            if self.b.traps!=traps or self.b.poweroff:break
        elapsed=time.perf_counter()-start
        path.write_text(json.dumps(dict(initial=initial,boundaries=rows,events=self.events),separators=(',',':'))+'\n')
        return dict(retired=self.b.steps,delta=self.b.steps-old,boundaries=len(rows),events=len(self.events),
            measured_spikes=sum(x['spikes'] for x in self.events),seconds=elapsed,trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def test(a):
    data=fixture();b=board(data,a);b.__class__=SynapticBoard;b.install(PagedCrossbar(256));attach(b);visual_upgrade(b,a);upgrade(b,a)
    r=Recorder(b,a);rng=random.Random(310508);checks=0
    r.enabled=True
    for op in OPS:
        for i in range(12):
            x,y=rng.getrandbits(32),rng.getrandbits(32)
            assert b.c.run(op,x,y)==r.old[0](op,x,y);checks+=1
    try:b.c.run('ADD',1,2,True)
    except RuntimeError:pass
    else:raise AssertionError('ablation')
    e=r.events[-1];assert e['ablated'] and not e['valid']
    bits=np.unpackbits(np.frombuffer(bytes.fromhex(e['rail_bits']),dtype=np.uint8),bitorder='little')[:r.graphs[0].n//5]
    assert not any(bits[i] for i in range(len(bits)) if r.graphs[0].level[5*i]>0)
    r.enabled=False;r.events.clear()
    r.step();assert r.events
    count=len(r.events);time.sleep(.02);assert len(r.events)==count
    return dict(status='PASS',alu_differentials=checks,ablation=True,pause_no_events=True,
        measured_regions=list(r.names),coordinates_available=False)

def run(a):
    data=a.image.read_bytes()
    if a.cold:
        b=board(data,a);b.__class__=SynapticBoard;b.install(PagedCrossbar(256));attach(b);visual_upgrade(b,a);upgrade(b,a)
        assert b.steps==b.boundaries==0 and not b.serial
    else:b=restore(data,a,a.input)
    PageCache(b);r=Recorder(b,a)
    # Audit native results independently, without using the oracle as output.
    old=b.c.run;oldm=b.math.run;checks=0
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
    g=r.graph(a.graph);report=r.sample(a.trace,a.steps)
    report.update(g,oracle_checks=checks,linux_booted=False,cold_entry=bool(a.cold),image_sha256=hashlib.sha256(data).hexdigest())
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--cold',action='store_true');p.add_argument('--graph',type=Path);p.add_argument('--trace',type=Path);p.add_argument('--steps',type=int,default=16);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.steps<=128:p.error('bounded trace')
    if a.image and not(a.graph and a.trace and bool(a.input)!=bool(a.cold)):p.error('graph/trace paths and exactly one of --input or --cold required')
    r=run(a) if a.image else test(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

