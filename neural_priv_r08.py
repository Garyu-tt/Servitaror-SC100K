"""SC100K R08: synchronous architectural traps and M/S/U transitions.

Privilege mode and trap CSRs are dual-rail state. CSR bit transforms and
trap-vector address masking use the neural ALU. Host dispatches architectural
policy and sequences commits; interrupts, translation and counters are future.
"""
import json, struct, time
from neural_alu_r01 import Circuit,MASK,ROOT
from neural_csr_r07 import CSRCPU,CSRDecoder,CSRFault,MISA,word
from neural_atomic_r06 import A_OPS,atomic_word
from neural_register_r03 import encode,decode,fire,validate
from neural_rv32i_r01 import I,J,S
import numpy as np
SYS={'ECALL':0x00000073,'MRET':0x30200073,'SRET':0x10200073,'WFI':0x10500073}
class PrivDecoder(CSRDecoder):
    patterns=CSRDecoder.patterns+tuple((n,MASK,w) for n,w in SYS.items())
class GuestTrap(RuntimeError):
    def __init__(self,cause,tval=0):
        self.cause=cause;self.tval=tval&MASK
        super().__init__(f'guest exception {cause}: {self.tval:#x}')
class Mode:
    neurons=60
    def __init__(self):self.state=encode(3,2)
    def get(self):return decode(fire(7*self.state.sum(axis=-1,dtype=np.int32)))
    def set(self,v):
        if v not in (0,1,3):raise ValueError('bad privilege mode')
        q=fire(7*encode(v,2).sum(axis=-1,dtype=np.int32));validate(q);self.state=q
class PrivCPU(CSRCPU):
    def __init__(self,c,d,image):
        self.priv=Mode()
        super().__init__(c,d,image)
        self.csrs.fixed[0x301]=encode(MISA|(1<<18)|(1<<20),32)
        self.traps=0;self.returns=0;self.waits=0
    @property
    def mode(self):return self.priv.get()
    @mode.setter
    def mode(self,v):self.priv.set(v)
    def or_(self,a,b):return self.alu('OR',a,b)
    def and_(self,a,b):return self.alu('AND',a,b)
    def replace(self,status,mask,value):
        return self.or_(self.and_(status,MASK^mask),self.and_(value,mask))
    def bit(self,status,from_bit,to_bit):
        x=self.and_(status,1<<from_bit)
        if from_bit>to_bit:return self.alu('SRL',x,from_bit-to_bit)
        if to_bit>from_bit:return self.alu('SLL',x,to_bit-from_bit)
        return x
    def put(self,addr,value):self.csrs.commit(self.csrs.prepare(addr,value))
    def check_address(self,a,n,align_cause,access_cause):
        if a%n:raise GuestTrap(align_cause,a)
        if a<0 or a+n>len(self.ram):raise GuestTrap(access_cause,a)
    def fetch(self,a):
        self.check_address(a,4,0,1)
        return super().load(a,4)
    def load(self,a,n,signed=False):
        self.check_address(a,n,4,5)
        return super().load(a,n,signed)
    def store(self,a,n,v):
        self.check_address(a,n,6,7)
        super().store(a,n,v)
    def illegal(self,ins):raise GuestTrap(2,ins)
    def check_next_pc(self,nxt):
        if nxt&3:raise GuestTrap(0,nxt)
    def prepare_instruction_extension(self,name,a,b,ins):
        if name in A_OPS:
            self.check_address(a,4,4 if name=='LR.W' else 6,5 if name=='LR.W' else 7)
        try:return super().prepare_instruction_extension(name,a,b,ins)
        except CSRFault:raise GuestTrap(2,ins) from None
    def special_step(self,name,ins,pc):
        if name=='ECALL':raise GuestTrap({0:8,1:9,3:11}[self.mode])
        if name=='EBREAK':raise GuestTrap(3,pc)
        if name not in ('MRET','SRET','WFI'):return False
        mode=self.mode;status=self.csrs.raw(0x300)
        if name=='WFI':
            # Permitted hint behavior; no low power/wakeup simulation.
            if mode==0:raise GuestTrap(2,ins)
            self.pc=self.alu('ADD',pc,4);self.steps+=1;self.waits+=1;return True
        if name=='MRET':
            if mode!=3:raise GuestTrap(2,ins)
            target_mode=(status>>11)&3
            new=self.replace(status,0x1888,self.or_(self.bit(status,7,3),0x80))
            target=self.csrs.raw(0x341)
        else:
            if mode==0:raise GuestTrap(2,ins)
            target_mode=(status>>8)&1
            new=self.replace(status,0x122,self.or_(self.bit(status,5,1),0x20))
            target=self.csrs.raw(0x141)
        self.check_next_pc(target)
        self.put(0x300,new);self.mode=target_mode;self.pc=target
        self.reservation=None;self.steps+=1;self.returns+=1
        return True
    def take_trap(self,cause,tval,pc):
        mode=self.mode;status=self.csrs.raw(0x300)
        delegated=mode!=3 and bool(self.and_(self.csrs.raw(0x302),1<<cause))
        if delegated:
            new=self.replace(status,0x122,self.or_(self.bit(status,1,5),(mode&1)<<8))
            epc,cc,tv,vec,target_mode=0x141,0x142,0x143,0x105,1
        else:
            new=self.replace(status,0x1888,self.or_(self.bit(status,3,7),mode<<11))
            epc,cc,tv,vec,target_mode=0x341,0x342,0x343,0x305,3
        target=self.and_(self.csrs.raw(vec),0xfffffffc)
        self.put(0x300,new);self.put(epc,pc);self.put(cc,cause);self.put(tv,tval)
        self.mode=target_mode;self.pc=target;self.reservation=None;self.traps+=1
    def step(self):
        pc=self.pc
        try:super().step()
        except GuestTrap as trap:
            self.take_trap(trap.cause,trap.tval,pc)

def main():
    c=Circuit().compile();d=PrivDecoder().compile()
    def cpu(words):return PrivCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    def at(m,a,w):m.store(a,4,w)
    t=time.perf_counter();checks=0
    # M -> U, U ECALL delegated to S, handler advances sepc, SRET to U.
    m=cpu([SYS['MRET']]);m.put(0x341,0x100);m.put(0x300,0x80|2)
    m.put(0x302,1<<8);m.put(0x105,0x200)
    at(m,0x100,SYS['ECALL']);at(m,0x104,I(0x13,10,0,0,42))
    for i,w in enumerate([word(2,5,0x141,0),I(0x13,5,0,5,4),word(1,0,0x141,5),SYS['SRET']]):at(m,0x200+4*i,w)
    m.step();assert m.mode==0 and m.pc==0x100 and m.csrs.raw(0x300)&0x88==0x88
    m.step();assert m.mode==1 and m.pc==0x200 and m.csrs.raw(0x141)==0x100 and m.csrs.raw(0x142)==8 and m.csrs.raw(0x143)==0
    assert m.csrs.raw(0x300)&0x122==0x20
    for _ in range(4):m.step()
    assert m.mode==0 and m.pc==0x104 and m.csrs.raw(0x300)&0x122==0x22
    m.step();assert m.reg[10]==42 and m.traps==1 and m.returns==2;checks+=1
    # M-mode ecall never delegates. Vectored mode still uses BASE for sync.
    n=cpu([SYS['ECALL']]);n.put(0x305,0x301);n.put(0x300,8)
    n.step();assert n.pc==0x300 and n.mode==3 and n.csrs.raw(0x342)==11 and n.csrs.raw(0x341)==0
    assert n.csrs.raw(0x300)&0x1888==0x1880;checks+=1
    # S-mode ecall enters M unless delegated; MRET restores S.
    n=cpu([SYS['ECALL']]);n.mode=1;n.put(0x305,0x300);at(n,0x300,SYS['MRET'])
    n.step();assert n.mode==3 and n.csrs.raw(0x342)==9 and n.csrs.raw(0x300)&0x1800==0x800
    n.put(0x341,4);n.step();assert n.mode==1 and n.pc==4;checks+=1
    cases=[
        (0xffffffff,3,2,0xffffffff,0),(SYS['MRET'],1,2,SYS['MRET'],0),
        (SYS['SRET'],0,2,SYS['SRET'],0),(word(2,5,0x340,0),0,2,word(2,5,0x340,0),0),
        (J(5,2),3,0,2,0),(I(3,5,2,1,0),3,4,1,1),
        (I(3,5,2,1,0),3,5,65536,65536),(S(2,1,2,0),3,6,1,1),
        (S(2,1,2,0),3,7,65536,65536),(0x00100073,3,3,0,0),
        (atomic_word(2,5,1),3,4,1,1),(atomic_word(0,5,1,2),3,6,1,1)]
    for ins,mode,cause,tval,addr in cases:
        f=cpu([ins]);f.mode=mode;f.reg[1]=addr;f.reg[2]=99;f.put(0x305,0x400)
        before=bytes(f.ram);f.step()
        assert f.mode==3 and f.pc==0x400 and f.csrs.raw(0x342)==cause and f.csrs.raw(0x343)==tval,(hex(ins),cause,f.csrs.raw(0x342))
        assert f.csrs.raw(0x341)==0 and f.reg[5]==0 and f.steps==0 and bytes(f.ram)==before
        checks+=1
    f=cpu([]);f.pc=65536;f.put(0x305,0x400);f.step()
    assert f.csrs.raw(0x342)==1 and f.csrs.raw(0x343)==65536;checks+=1
    f=cpu([SYS['WFI']]);f.step();assert f.pc==4 and f.waits==1;checks+=1
    # A host neural failure is not misrepresented as a guest illegal instruction.
    f=cpu([SYS['ECALL']]);f.decoder.decode=lambda *_:(_ for _ in ()).throw(RuntimeError('invalid neural rails'))
    try:f.step()
    except RuntimeError as e:assert str(e)=='invalid neural rails'
    else:raise AssertionError('host fault swallowed')
    assert f.traps==0 and f.pc==0;checks+=1
    total=c.n+d.n+m.reg.neurons+m.control.neurons+m.counter.neurons+m.math.neurons+m.csrs.neurons+m.priv.neurons
    assert total<=131072
    report=dict(status='PASS',scenario_checks=checks,implemented_modes=['M','S','U'],
        new_system_patterns=list(SYS),guest_retired_in_roundtrip=m.steps,
        roundtrip_traps=m.traps,roundtrip_returns=m.returns,roundtrip_result=m.reg[10],
        privilege_cells=m.priv.neurons,decoder_cells=d.n,total_budget_cells=total,
        seconds=round(time.perf_counter()-t,3),linux_booted=False,
        limitations='Synchronous exception subset; host dispatches policy, no interrupt delivery, MMU, PMP, counters, TVM/TW/TSR/MPRV, full privilege conformance or continuous integrated GENESIS.')
    out=ROOT/'priv-r08';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

