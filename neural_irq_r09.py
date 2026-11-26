"""R09: deterministic CLINT-like timer, interrupt delivery and time CSR.

Host implements peripheral counters, bus and interrupt scheduling. The
architectural pending/status/cause/EPC/PC state uses neural rails and the
existing neural ALU for interrupt masks and trap-vector arithmetic.
"""
import json, struct, time
from neural_alu_r01 import Circuit,MASK,ROOT
from neural_priv_r08 import PrivCPU,PrivDecoder,GuestTrap,SYS
from neural_csr_r07 import CSRBank,CSRFault,word
from neural_register_r03 import encode,decode,fire,validate
from neural_rv32i_r01 import I,S
MASK64=(1<<64)-1
MSIP=0x02000000
MTIMECMP=0x02004000
MTIME=0x0200bff8
TIME=0xc01
TIMEH=0xc81
PRIORITY=(11,3,7,9,1,5)
HW_MASK=(1<<11)|(1<<3)|(1<<7)|(1<<9)|(1<<5)
class Timer:
    def __init__(self):
        self.now=0;self.compare=MASK64;self.software=0;self.external=0;self.supervisor_timer=False
    def advance(self):self.now=(self.now+1)&MASK64
    def pending(self):
        return self.external|(self.software<<3)|((self.now>=self.compare)<<7)|(int(self.supervisor_timer)<<5)
    @staticmethod
    def contains(a):return a in (MSIP,MTIMECMP,MTIMECMP+4,MTIME,MTIME+4)
    def read(self,a):
        if a==MSIP:return self.software
        if a in (MTIMECMP,MTIMECMP+4):return (self.compare>>(32*(a!=MTIMECMP)))&MASK
        if a in (MTIME,MTIME+4):return (self.now>>(32*(a!=MTIME)))&MASK
        raise ValueError('bad timer address')
    def write(self,a,v):
        v&=MASK
        if a==MSIP:self.software=v&1;return
        for base,attr in ((MTIMECMP,'compare'),(MTIME,'now')):
            if a in (base,base+4):
                shift=32*(a!=base)
                setattr(self,attr,(getattr(self,attr)&(MASK64^(MASK<<shift)))|(v<<shift));return
        raise ValueError('bad timer address')
class TimerCSR(CSRBank):
    neurons=CSRBank.neurons+640
    def __init__(self,alu,timer):
        super().__init__(alu);self.timer=timer
    def check(self,addr,mode,write):
        if addr not in (TIME,TIMEH):return super().check(addr,mode,write)
        if write:raise CSRFault('read-only time')
        if mode not in (0,1,3):raise CSRFault('bad privilege')
        if mode!=3 and not self.raw(0x306)&2:raise CSRFault('mcounteren time')
        if mode==0 and not self.raw(0x106)&2:raise CSRFault('scounteren time')
    def read(self,addr):
        if addr not in (TIME,TIMEH):return super().read(addr)
        self.reads+=1
        rails=encode((self.timer.now>>(32*(addr==TIMEH)))&MASK,32)
        return decode(fire(7*rails.sum(axis=-1,dtype='int32')))
class IRQCPU(PrivCPU):
    def __init__(self,c,d,image):
        super().__init__(c,d,image)
        self.timer=Timer();self.csrs=TimerCSR(self.alu,self.timer)
        from neural_csr_r07 import MISA
        self.csrs.fixed[0x301]=encode(MISA|(1<<18)|(1<<20),32)
        self.interrupts=0;self.boundaries=0
    def sync_pending(self):
        old=self.csrs.raw(0x344);hardware=self.timer.pending()
        # External device writes hardware-owned bits, preserving writable SSIP.
        new=(old&~HW_MASK)|hardware
        if new!=old:self.csrs.commit((0x344,new))
    def load(self,a,n,signed=False):
        if Timer.contains(a):
            if n!=4 or self.mode!=3:raise GuestTrap(5,a)
            return self.timer.read(a)
        return super().load(a,n,signed)
    def store(self,a,n,v):
        if Timer.contains(a):
            if n!=4 or self.mode!=3:raise GuestTrap(7,a)
            self.timer.write(a,v);self.reservation=None;self.sync_pending();return
        super().store(a,n,v)
    def select_irq(self):
        pending=self.csrs.raw(0x344)
        if not pending:return None
        enabled=self.and_(pending,self.csrs.raw(0x304))
        if not enabled:return None
        delegation=self.csrs.raw(0x303);status=self.csrs.raw(0x300);mode=self.mode
        # Privilege and priority are architectural host dispatch policy.
        for cause in PRIORITY:
            if not enabled&(1<<cause):continue
            delegated=bool(delegation&(1<<cause))
            if delegated:
                if mode==3 or (mode==1 and not status&2):continue
                return cause,1
            if mode==3 and not status&8:continue
            return cause,3
        return None
    def take_irq(self,cause,target_mode):
        pc=self.pc;mode=self.mode;status=self.csrs.raw(0x300)
        if target_mode==1:
            new=self.replace(status,0x122,self.or_(self.bit(status,1,5),(mode&1)<<8))
            epc,cc,tv,vec=0x141,0x142,0x143,0x105
        else:
            new=self.replace(status,0x1888,self.or_(self.bit(status,3,7),mode<<11))
            epc,cc,tv,vec=0x341,0x342,0x343,0x305
        vector=self.csrs.raw(vec);target=self.and_(vector,0xfffffffc)
        if vector&3==1:target=self.alu('ADD',target,self.alu('SLL',cause,2))
        self.put(0x300,new);self.put(epc,pc);self.put(cc,self.or_(0x80000000,cause));self.put(tv,0)
        self.mode=target_mode;self.pc=target;self.reservation=None;self.interrupts+=1
    def step(self):
        self.timer.advance();self.boundaries+=1;self.sync_pending()
        irq=self.select_irq()
        if irq is not None:self.take_irq(*irq);return
        super().step()

def main():
    c=Circuit().compile();d=PrivDecoder().compile()
    def cpu(words=()):return IRQCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    def at(m,a,w):m.store(a,4,w)
    t=time.perf_counter();checks=0
    # Actual guest MMIO write to mtimecmp; timer IRQ takes precedence over
    # instruction fetch and handler disarms the device before returning.
    m=cpu([S(2,1,2,0),I(0x13,10,0,0,42)])
    m.reg[1]=MTIMECMP;m.reg[2]=2;m.reg[3]=MASK
    m.timer.compare=0xffffffff
    m.put(0x304,1<<7);m.put(0x300,8);m.put(0x305,0x100|1)
    at(m,0x11c,S(2,1,3,0));at(m,0x120,SYS['MRET'])
    m.step();assert m.pc==4 and m.timer.compare==2
    m.step();assert m.pc==0x11c and m.steps==1 and m.csrs.raw(0x341)==4
    assert m.csrs.raw(0x342)==0x80000007 and m.csrs.raw(0x343)==0 and m.csrs.raw(0x300)&0x1888==0x1880
    m.step();m.step();m.step()
    assert m.reg[10]==42 and m.pc==8 and m.interrupts==1 and m.steps==4;checks+=1
    # Delegated supervisor timer interrupts U even with SIE clear.
    n=cpu([I(0x13,8,0,0,7)]);n.mode=0;n.put(0x303,1<<5);n.put(0x304,1<<5);n.put(0x105,0x200|1)
    n.ram[0x214:0x218]=struct.pack('<I',SYS['SRET'])
    n.timer.supervisor_timer=True;n.step()
    assert n.mode==1 and n.pc==0x214 and n.csrs.raw(0x142)==0x80000005 and n.csrs.raw(0x141)==0
    n.timer.supervisor_timer=False;n.step();n.step()
    assert n.mode==0 and n.reg[8]==7 and n.interrupts==1;checks+=1
    # Delegated interrupt masked in M, and in S with SIE clear.
    for mode in (3,1):
        q=cpu([I(0x13,8,0,0,9)]);q.mode=mode;q.put(0x303,1<<5);q.put(0x304,1<<5);q.timer.supervisor_timer=True
        q.step();assert q.reg[8]==9 and q.interrupts==0;checks+=1
    # Higher privilege IRQ preempts S regardless of MIE.
    q=cpu();q.mode=1;q.put(0x304,1<<3);q.put(0x305,0x300);q.timer.software=1;q.step()
    assert q.mode==3 and q.pc==0x300 and q.csrs.raw(0x342)==0x80000003;checks+=1
    # Fixed machine priority: external before software before timer.
    q=cpu();q.put(0x300,8);q.put(0x304,0x888);q.put(0x305,0x300)
    q.timer.external=1<<11;q.timer.software=1;q.timer.compare=0;q.step()
    assert q.csrs.raw(0x342)==0x8000000b;checks+=1
    # Disabled interrupt remains pending, no retirement loss.
    q=cpu([I(0x13,8,0,0,1)]);q.timer.compare=0;q.step()
    assert q.reg[8]==1 and q.csrs.raw(0x344)&0x80 and not q.interrupts;checks+=1
    # MMIO 64-bit high/low writes and read-only time/timeh guest access.
    q=cpu([word(2,5,TIME,0),word(2,6,TIMEH,0)]);q.store(MTIME+4,4,0x1234);q.store(MTIME,4,0xfffffffe)
    q.step();q.step();assert q.reg[5]==MASK and q.reg[6]==0x1235
    assert q.load(MTIME,4)==0 and q.load(MTIME+4,4)==0x1235;checks+=1
    for mode,mc,sc,ok in ((0,0,0,False),(0,2,0,False),(0,2,2,True),(1,0,0,False),(1,2,0,True)):
        q=cpu([word(2,5,TIME,0)]);q.mode=mode;q.put(0x306,mc);q.put(0x106,sc);q.put(0x305,0x400);q.step()
        assert (q.steps==1)==ok
        if not ok:assert q.csrs.raw(0x342)==2 and q.pc==0x400
        checks+=1
    q=cpu([word(1,0,TIME,0)]);q.put(0x305,0x400);q.step();assert q.csrs.raw(0x342)==2;checks+=1
    # Hardware bits are read-only to guest CSR writes; SSIP persists.
    q=cpu();q.timer.compare=0;q.sync_pending();q.put(0x344,2);q.sync_pending()
    assert q.csrs.raw(0x344)==0x82;q.timer.compare=MASK64;q.sync_pending();assert q.csrs.raw(0x344)==2;checks+=1
    total=c.n+d.n+m.reg.neurons+m.control.neurons+m.counter.neurons+m.math.neurons+m.csrs.neurons+m.priv.neurons
    assert total<=131072
    report=dict(status='PASS',scenarios=checks,interrupt_causes=list(PRIORITY),
        guest_timer_roundtrip=True,guest_supervisor_roundtrip=True,guest_retired=m.steps,
        timer_boundaries=m.boundaries,csr_budget_cells=m.csrs.neurons,total_budget_cells=total,
        seconds=round(time.perf_counter()-t,3),linux_booted=False,
        limitations='Deterministic host timer increments per step boundary, not wall time. CLINT-like word MMIO, host IRQ priority/delegation, no PLIC, SBI, real WFI sleep, MMU, PMP, Sstc or full conformance. Supervisor timer/external lines are test injection only.')
    out=ROOT/'irq-r09';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

