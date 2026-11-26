"""R17: conservative page-walk cache; never skips guest instructions.

A hit reuses a mapping obtained by the R10 neural page walker. The existing
neural ALU still assembles the address. Context and access kind are part
of the key. Writes to any observed page-table page flush all entries.
Only single-hart, no DMA, synchronous board access is supported.
"""
import argparse,hashlib,json,time
from collections import OrderedDict
from pathlib import Path
import numpy as np
from neural_linux_r12 import load_image,fixture
from neural_board_r11 import RAM,Board
from neural_mmu_r10 import V,R,W,X,U,A,D,pte
from neural_priv_r08 import GuestTrap
import neural_resume_r13 as r13
import neural_recurrent_r15 as r15

class PageCache:
    def __init__(self,b,capacity=256):
        self.b=b;self.capacity=capacity;self.entries=OrderedDict()
        self.pages=set();self.hits=0;self.misses=0;self.flushes=0
        self.original=b.translate;self.read=b.phys_read;self.write=b.phys_write
        self.special=b.special_step
        b.translate=self.translate;b.phys_write=self.phys_write;b.special_step=self.special_step
    def flush(self):
        self.entries.clear();self.pages.clear();self.flushes+=1
    def phys_write(self,a,n,v):
        self.write(a,n,v)
        if any(p in self.pages for p in range(a>>12,((a+n-1)>>12)+1)):
            self.flush()
    def special_step(self,name,ins,pc):
        result=self.special(name,ins,pc)
        if name=='SFENCE.VMA' and result:self.flush()
        return result
    def translate(self,va,kind):
        b=self.b;satp=b.csrs.raw(0x180);mode=b.mode
        if mode==3 or not satp&0x80000000:return va
        key=(satp,mode,b.csrs.raw(0x300)&0xc0000,kind,va>>12)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key)
            return b.or_(self.entries[key],b.and_(va,0xfff))
        self.misses+=1;observed=set()
        def record(a,n,signed=False):
            observed.add(a>>12);return self.read(a,n,signed)
        b.phys_read=record
        try:pa=self.original(va,kind)
        finally:b.phys_read=self.read
        # A/D updates during this walk may flush older entries. Publish the
        # new entry only after the walk and its updates have completed.
        self.pages.update(observed)
        self.entries[key]=pa&0xfffff000
        if len(self.entries)>self.capacity:self.entries.popitem(last=False)
        # Bound metadata too. Clearing is always conservative.
        if len(self.pages)>self.capacity*4:self.flush()
        return pa
    def report(self):
        return dict(hits=self.hits,misses=self.misses,flushes=self.flushes,
                    hit_rate=self.hits/max(1,self.hits+self.misses))

def selftest(a):
    b,_=load_image(fixture(),16);e=r15.setup(b,a);e.attach(b)
    root=RAM+0x1000;table=RAM+0x2000;leaf=RAM+0x3000
    va=0x40000000
    b.phys_write(root+256*4,4,pte(table,V))
    b.phys_write(table,4,pte(leaf,V|R|W|X|U))
    b.put(0x180,0x80000000|(root>>12));b.mode=0
    c=PageCache(b);checks=0
    for kind in ('fetch','load','store'):
        for off in (0,4,0xffc,16,128):
            assert b.translate(va+off,kind)==leaf+off;checks+=1
    assert c.hits>=10 and b.phys_read(table,4)&(A|D)==A|D
    # Leaf remap via bus, including partial write invalidates.
    b.phys_write(table,4,pte(leaf+4096,V|R|U|A));assert not c.entries
    assert b.translate(va,'load')==leaf+4096;checks+=1
    try:b.translate(va,'store')
    except GuestTrap as ex:assert ex.cause==15;checks+=1
    else:raise AssertionError('store permission')
    b.phys_write(table,1,0);assert not c.entries
    try:b.translate(va,'load')
    except GuestTrap as ex:assert ex.cause==13;checks+=1
    else:raise AssertionError('stale mapping')
    b.phys_write(table,4,pte(leaf,V|R|U|A))
    assert b.translate(va,'load')==leaf;checks+=1
    b.mode=1
    try:b.translate(va,'load')
    except GuestTrap as ex:assert ex.cause==13;checks+=1
    else:raise AssertionError('SUM check')
    b.put(0x300,1<<18);assert b.translate(va,'load')==leaf;checks+=1
    b.put(0x300,0)
    try:b.translate(va,'load')
    except GuestTrap as ex:assert ex.cause==13;checks+=1
    else:raise AssertionError('context stale')
    b.special_step('SFENCE.VMA',0x12000073,b.pc);assert not c.entries;checks+=1
    b.mode=0;b.put(0x180,0);assert b.translate(va,'load')==va;checks+=1
    b.put(0x180,0x80000000|(root>>12))
    assert b.translate(va,'load')==leaf;checks+=1
    # Parent page remap invalidation and 4 MiB superpage.
    b.phys_write(root+256*4,4,pte(RAM,V|R|X|U|A))
    assert b.translate(va+0x2000,'fetch')==RAM+0x2000
    assert b.translate(va+0x2004,'fetch')==RAM+0x2004;checks+=2
    # Real paged guest run, reference versus cached, same complete
    # architectural state apart from diagnostic execution counters.
    h=hashlib.sha256(fixture()).hexdigest()
    def machine(cache):
        q,_=load_image(fixture(),16);r15.r14.accelerate(q);e.attach(q)
        q.phys_write(root+256*4,4,pte(RAM+0x400000,V|R|W|X|A|D))
        q.put(0x180,0x80000000|(root>>12));q.pc=va+76;q.mode=1
        if cache:PageCache(q)
        return q
    x=machine(False);y=machine(True)
    r13.advance(x,64);r13.advance(y,64)
    sx=r13.snapshot(x,h);sy=r13.snapshot(y,h)
    assert all(np.array_equal(sx[k],sy[k]) for k in sx if k!='meta')
    mx=json.loads(sx['meta'].tobytes());my=json.loads(sy['meta'].tobytes())
    for k in ('alu_calls','walks','pte_reads'):mx['counters'].pop(k);my['counters'].pop(k)
    # CSR read counters are unaffected by raw reads.
    assert mx==my,(mx,my)
    assert x.steps==y.steps==64
    return dict(status='PASS',mapping_checks=checks,architectural_state_equal=True,fixture_steps=64,cache=c.report())

def main():
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--budget',type=int,default=512);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.budget<=100000:p.error('bounded budget')
    if not a.image:r=selftest(a)
    else:
        if not a.input or not a.checkpoint:p.error('checkpoints required')
        data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b,_=load_image(data,64)
        e=r15.setup(b,a);r13.restore(b,a.input,h);e.attach(b);c=PageCache(b)
        r=r13.advance(b,a.budget);r13.save(b,a.checkpoint,h)
        r.update(image_sha256=h,checkpoint_bytes=a.checkpoint.stat().st_size,cache=c.report(),walks=b.walks,pte_reads=b.pte_reads)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()

