"""R10: bounded Sv32 page walker with neural address arithmetic.

Host memory bus and permission policy remain conventional. PTE extraction,
address arithmetic and A/D bit updates use the existing Q4 ALU. No TLB:
SFENCE.VMA is a valid no-op after privilege check. Only 32-bit physical RAM
and machine timer MMIO are supported in this fixture.
"""
import json,struct,time
from neural_alu_r01 import Circuit,MASK,ROOT
from neural_irq_r09 import IRQCPU,Timer
from neural_priv_r08 import PrivDecoder,GuestTrap
from neural_atomic_r06 import A_OPS,atomic_word
from neural_rv32i_r01 import CPU,I,S
from neural_csr_r07 import word
V,R,W,X,U,G,A,D=1,2,4,8,16,32,64,128
SATP=0x180
SFENCE=0x12000073
class MMUDecoder(PrivDecoder):
    patterns=PrivDecoder.patterns+(('SFENCE.VMA',0xfe007fff,SFENCE),('FENCE.I',MASK,0x0000100f))
class MMUCPU(IRQCPU):
    def __init__(self,c,d,image):
        super().__init__(c,d,image);self.walks=0;self.pte_reads=0;self.ad_updates=0;self.fences=0
        self._atomic_physical=False;self._atomic_addr=None
    def phys_read(self,a,n,signed=False):
        if a<0 or a+n>len(self.ram) or a%n:raise ValueError('physical access')
        return CPU.load(self,a,n,signed)
    def phys_write(self,a,n,v):
        if a<0 or a+n>len(self.ram) or a%n:raise ValueError('physical access')
        CPU.store(self,a,n,v)
        if self.reservation is not None and a<self.reservation+4 and self.reservation<a+n:self.reservation=None
    def translate(self,va,kind):
        cause={'fetch':12,'load':13,'store':15}[kind]
        access={'fetch':1,'load':5,'store':7}[kind]
        satp=self.csrs.raw(SATP)
        if self.mode==3 or not satp&0x80000000:return va
        self.walks+=1;status=self.csrs.raw(0x300)
        def fail():raise GuestTrap(cause,va)
        def badphys():raise GuestTrap(access,va)
        root=self.and_(satp,0x003fffff)
        if root>=1<<20:badphys() # physical bus fixture is 32 bit
        table=self.alu('SLL',root,12)
        for level in (1,0):
            vpn=self.and_(self.alu('SRL',va,12+10*level),0x3ff)
            pa=self.alu('ADD',table,self.alu('SLL',vpn,2))
            try:pte=self.phys_read(pa,4)
            except ValueError:badphys()
            self.pte_reads+=1
            flags=self.and_(pte,0xff)
            if not flags&V or (flags&W and not flags&R):fail()
            ppn=self.alu('SRL',pte,10)
            if flags&(R|X):
                if self.mode==0 and not flags&U:fail()
                if self.mode==1 and flags&U and (kind=='fetch' or not status&(1<<18)):fail()
                if kind=='fetch' and not flags&X:fail()
                if kind=='load' and not (flags&R or (status&(1<<19) and flags&X)):fail()
                if kind=='store' and not flags&W:fail()
                if level==1 and self.and_(ppn,0x3ff):fail()
                if ppn>=1<<20:badphys()
                # Assemble base and offset with the same neural ALU.
                if level==1:
                    base=self.alu('SLL',ppn,12);off=self.and_(va,0x3fffff)
                else:
                    base=self.alu('SLL',ppn,12);off=self.and_(va,0xfff)
                result=self.or_(base,off)
                required=A|(D if kind=='store' else 0)
                if flags&required!=required:
                    new=self.or_(pte,required)
                    try:self.phys_write(pa,4,new)
                    except ValueError:badphys()
                    self.ad_updates+=1
                return result
            if level==0 or flags&(U|A|D):fail()
            if ppn>=1<<20:badphys()
            table=self.alu('SLL',ppn,12)
        fail()
    def physical(self,va,n,kind):
        ac,mis={'fetch':(1,0),'load':(5,4),'store':(7,6)}[kind]
        if va%n:raise GuestTrap(mis,va)
        pa=va if self._atomic_physical else self.translate(va,kind)
        if pa+n>len(self.ram) and not (kind!='fetch' and Timer.contains(pa) and self.mode==3 and n==4):raise GuestTrap(ac,va)
        return pa
    def fetch(self,a):
        return self.phys_read(self.physical(a,4,'fetch'),4)
    def load(self,a,n,signed=False):
        pa=self.physical(a,n,'load')
        if Timer.contains(pa):return self.timer.read(pa)
        return self.phys_read(pa,n,signed)
    def store(self,a,n,v):
        pa=self.physical(a,n,'store')
        if Timer.contains(pa):self.timer.write(pa,v);self.reservation=None;self.sync_pending()
        else:self.phys_write(pa,n,v)
    def prepare_instruction_extension(self,name,a,b,ins):
        if name not in A_OPS:return super().prepare_instruction_extension(name,a,b,ins)
        # Translate the atomic address once; require normal RAM, not MMIO.
        pa=self.physical(a,4,'load' if name=='LR.W' else 'store')
        if pa+4>len(self.ram):raise GuestTrap(5 if name=='LR.W' else 7,a)
        self._atomic_addr=pa
        self._atomic_physical=True
        try:result,store=super().prepare_instruction_extension(name,pa,b,ins)
        finally:self._atomic_physical=False
        return result,(a,store[1],store[2]) if store else None
    def finish_instruction_extension(self,name,a,b,ins):
        if name in A_OPS:
            super().finish_instruction_extension(name,self._atomic_addr,b,ins)
            self._atomic_addr=None
        else:super().finish_instruction_extension(name,a,b,ins)
    def special_step(self,name,ins,pc):
        if name not in ('SFENCE.VMA','FENCE.I'):return super().special_step(name,ins,pc)
        if name=='SFENCE.VMA' and self.mode==0:raise GuestTrap(2,ins)
        self.pc=self.alu('ADD',pc,4);self.steps+=1;self.fences+=1;return True

def pte(pa,flags):return ((pa>>12)<<10)|flags
def main():
    c=Circuit().compile();d=MMUDecoder().compile()
    def cpu():
        m=MMUCPU(c,d,b'')
        m.phys_write(0x1000+256*4,4,pte(0x2000,V))
        m.phys_write(0x2000,4,pte(0x3000,V|R|X|U))
        m.phys_write(0x2004,4,pte(0x4000,V|R|W|U))
        m.put(SATP,0x80000001);return m
    va=0x40000000;checks=0;t=time.perf_counter()
    m=cpu();m.phys_write(0x3000,4,I(3,5,2,1,0));m.phys_write(0x3004,4,S(2,1,5,4))
    m.phys_write(0x4000,4,0x12345678);m.reg[1]=va+4096;m.pc=va;m.mode=0
    m.step();m.step()
    assert m.reg[5]==0x12345678 and m.phys_read(0x4004,4)==0x12345678 and m.pc==va+8
    assert m.phys_read(0x2000,4)&A and m.phys_read(0x2004,4)&(A|D)==A|D
    checks+=1
    # Page permissions, illegal PTE encodings, superpage alignment, SUM/MXR.
    cases=[(V|R|W|U,'fetch',0,0,12),(V|R|X,'load',0,0,13),
        (V|R|X|U,'store',0,0,15),(V|W|U,'load',0,0,13),
        (V|X|U,'load',0,0,13),(V|R|U,'load',1,0,13),
        (V|R|X|U,'fetch',1,1<<18,12),(0,'load',0,0,13)]
    for flags,kind,mode,status,cause in cases:
        q=cpu();q.phys_write(0x2000,4,pte(0x3000,flags));q.put(0x300,status);q.mode=mode
        try:q.translate(va,kind)
        except GuestTrap as e:assert (e.cause,e.tval)==(cause,va)
        else:raise AssertionError('permission failure missing')
        checks+=1
    q=cpu();q.phys_write(0x2000,4,pte(0x3000,V|X|U));q.put(0x300,1<<19);q.mode=0
    assert q.translate(va,'load')==0x3000;checks+=1
    q=cpu();q.put(0x300,1<<18);q.mode=1;assert q.translate(va,'load')==0x3000;checks+=1
    q=cpu();q.phys_write(0x1400,4,pte(0,V|R|X|U));q.mode=0
    assert q.translate(va+0x3456,'fetch')==0x3456;checks+=1
    q.phys_write(0x1400,4,pte(0x1000,V|R|X|U))
    try:q.translate(va,'fetch')
    except GuestTrap as e:assert e.cause==12
    else:raise AssertionError('misaligned superpage')
    checks+=1
    # Actual page fault enters M with virtual fault address, no retirement.
    q=cpu();q.phys_write(0x3000,4,I(3,5,2,1,0));q.reg[1]=va+8192
    q.put(0x305,0x800);q.pc=va;q.mode=0;q.step()
    assert q.pc==0x800 and q.mode==3 and q.csrs.raw(0x342)==13 and q.csrs.raw(0x343)==va+8192 and q.steps==0;checks+=1
    # A translated physical alias must share LR/SC reservation.
    q=cpu();q.phys_write(0x2008,4,pte(0x4000,V|R|W|U))
    for i,w in enumerate([atomic_word(2,5,1),atomic_word(3,6,2,3)]):q.phys_write(0x3000+4*i,4,w)
    q.reg[1]=va+4096;q.reg[2]=va+8192;q.reg[3]=99;q.pc=va;q.mode=0
    q.step();q.step();assert q.reg[6]==0 and q.phys_read(0x4000,4)==99;checks+=1
    q=cpu();q.phys_write(0,4,SFENCE);q.phys_write(4,4,0x0000100f)
    q.step();q.step();assert q.fences==2 and q.pc==8;checks+=1
    q=cpu();q.phys_write(0x3000,4,SFENCE);q.pc=va;q.mode=0;q.put(0x305,0x800);q.step()
    assert q.csrs.raw(0x342)==2 and q.pc==0x800;checks+=1
    q=cpu();assert q.translate(va,'fetch')==va;checks+=1
    total=c.n+d.n+m.reg.neurons+m.control.neurons+m.counter.neurons+m.math.neurons+m.csrs.neurons+m.priv.neurons
    assert total<=131072
    report=dict(status='PASS',scenarios=checks,guest_virtual_instructions=m.steps,guest_page_walks=m.walks,pte_reads=m.pte_reads,
        ad_updates=m.ad_updates,physical_alias_lr_sc=True,total_budget_cells=total,seconds=round(time.perf_counter()-t,3),
        linux_booted=False,limitations='Host permission policy and physical bus, ALU page-walk arithmetic; no TLB, PMP, MPRV/TVM, 34-bit physical bus, Linux-scale RAM or full architectural conformance. Fixed circuits remain separate.')
    out=ROOT/'mmu-r10';out.mkdir(exist_ok=True);(out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

