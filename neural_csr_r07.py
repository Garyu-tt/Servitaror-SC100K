"""SC100K R07: Zicsr and clocked neural CSR storage.

Host routes CSR addresses and checks access/WARL policy. Mutable CSR values
live solely in dual-rail spike arrays; bit updates use the existing Q4 ALU.
Privilege transitions, interrupts and page translation are not implemented.
"""
import json, random, struct, time
import numpy as np
from neural_alu_r01 import Circuit, MASK, ROOT
from neural_atomic_r06 import ADecoder, ACPU
from neural_register_r03 import encode, decode, fire, validate
from neural_rv32i_r01 import I

CSR_OPS={1:'CSRRW',2:'CSRRS',3:'CSRRC',5:'CSRRWI',6:'CSRRSI',7:'CSRRCI'}
class CSRDecoder(ADecoder):
    patterns=ADecoder.patterns+tuple((name,0x707f,0x73|(f<<12)) for f,name in CSR_OPS.items())
    # FENCE.I is intentionally not claimed in this revision.

# Preliminary register contract: masks store supported bits but do not
# implement the architectural effects of interrupt, delegation or MMU CSRs.
MSTATUS_MASK=0x000c19aa # MIE/SIE, MPIE/SPIE, SPP, MPP, SUM, MXR
SSTATUS_MASK=0x000c0122
IRQ_MASK=0x00000aaa
MISA=(1<<30)|(1<<0)|(1<<8)|(1<<12)
MASKS={
    0x300:MSTATUS_MASK,0x302:0x0000b3ff,0x303:0x222,0x304:IRQ_MASK,
    0x305:MASK,0x340:MASK,0x341:0xfffffffc,0x342:MASK,0x343:MASK,
    0x344:0x002, # only SSIP software-writable in this fixture
    0x105:MASK,0x140:MASK,0x141:0xfffffffc,0x142:MASK,0x143:MASK,0x180:MASK,
    0x306:0x7,0x106:0x7,
}
FIXED={0x301:MISA,0xf11:0,0xf12:0,0xf13:0,0xf14:0}
ALIASES={0x100:0x300,0x104:0x304,0x144:0x344}
class CSRFault(RuntimeError):pass

class CSRBank:
    # Each word has 320 persistent cells. Input, threshold feedback and
    # readout lanes are shared across sequential CSR accesses.
    neurons=320*(len(MASKS)+len(FIXED)+3)
    def __init__(self,alu):
        self.alu=alu
        self.state={a:encode(0,32) for a in MASKS}
        self.fixed={a:encode(v,32) for a,v in FIXED.items()}
        self.reads=0;self.writes=0
    def check(self,addr,mode,write):
        if mode not in (0,1,3) or addr not in MASKS and addr not in FIXED and addr not in ALIASES:
            raise CSRFault('unsupported CSR')
        if mode<((addr>>8)&3):raise CSRFault('CSR privilege violation')
        if write and ((addr>>10)&3)==3:raise CSRFault('read-only CSR')
    def raw(self,addr):
        rails=self.state[addr] if addr in self.state else self.fixed[addr]
        return decode(fire(7*rails.sum(axis=-1,dtype=np.int32)))
    def mask(self,addr):
        if addr==0x100:return SSTATUS_MASK
        if addr in (0x104,0x144):
            return self.alu('AND',self.raw(0x303),IRQ_MASK)
        return MASK
    def read(self,addr):
        raw=self.raw(ALIASES.get(addr,addr))
        if addr in ALIASES:raw=self.alu('AND',raw,self.mask(addr))
        self.reads+=1
        return raw
    def prepare(self,addr,value):
        physical=ALIASES.get(addr,addr)
        if physical in FIXED:return None # fixed WARL misa
        wmask=MASKS[physical]
        if addr in ALIASES:wmask=self.alu('AND',wmask,self.mask(addr))
        old=self.raw(physical)
        value=self.alu('OR',self.alu('AND',old,self.alu('XOR',wmask,MASK)),self.alu('AND',value,wmask))
        # WARL policy checks are host control, bit transforms are neural.
        if physical==0x300 and ((value>>11)&3)==2:
            value=self.alu('AND',value,MASK^(3<<11))
        if physical in (0x105,0x305) and (value&3)>1:
            value=self.alu('AND',value,0xfffffffc)
        return physical,value
    def commit(self,pending,ablate=False):
        if pending is None:return
        addr,value=pending
        candidate=fire(7*encode(value&MASK,32).sum(axis=-1,dtype=np.int32)*(not ablate))
        validate(candidate)
        self.state[addr]=candidate;self.writes+=1

class CSRCPU(ACPU):
    def __init__(self,circuit,decoder,image):
        super().__init__(circuit,decoder,image)
        self.csrs=CSRBank(self.alu);self.mode=3;self.csr_calls=0
        self._csr_pending=None
    def prepare_instruction_extension(self,name,a,b,ins):
        if name not in CSR_OPS.values():
            return super().prepare_instruction_extension(name,a,b,ins)
        self._csr_pending=None
        addr=ins>>20;rs=(ins>>15)&31;rd=(ins>>7)&31
        imm=name.endswith('I');source=rs if imm else a
        kind=name[:-1] if imm else name
        write=kind=='CSRRW' or rs!=0
        read=kind!='CSRRW' or rd!=0
        self.csrs.check(addr,self.mode,write)
        old=self.csrs.read(addr) if read else 0
        if write:
            if kind=='CSRRW':new=source
            elif kind=='CSRRS':new=self.alu('OR',old,source)
            else:new=self.alu('AND',old,self.alu('XOR',source,MASK))
            self._csr_pending=self.csrs.prepare(addr,new)
        return old,None
    def finish_instruction_extension(self,name,a,b,ins):
        if name in CSR_OPS.values():
            self.csrs.commit(self._csr_pending);self._csr_pending=None;self.csr_calls+=1
        else:super().finish_instruction_extension(name,a,b,ins)

def word(f,rd,addr,src):return (addr<<20)|(src<<15)|(f<<12)|(rd<<7)|0x73

def main():
    c=Circuit().compile();d=CSRDecoder().compile()
    def cpu(words):return CSRCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    rng=random.Random(3113);checks=0
    for f,name in CSR_OPS.items():
        for _ in range(32):
            assert d.decode(word(f,rng.randrange(32),rng.randrange(4096),rng.randrange(32)))==name;checks+=1
    for f in (0,4):assert d.decode(word(f,1,0x340,1)) is None;checks+=1
    # Full guest execution: register and immediate read/modify/write.
    m=cpu([I(0x13,1,0,0,0x55),word(1,2,0x340,1),word(6,3,0x340,2),
           word(7,4,0x340,1),word(5,5,0x340,7),word(2,6,0x340,0),
           word(3,7,0x340,1),word(1,0,0x340,1),word(2,8,0x301,0),
           word(2,9,0xf14,0),0x00100073])
    t=time.perf_counter()
    while not m.halted and m.steps<100:m.step()
    guest_s=time.perf_counter()-t
    assert m.steps==11 and m.csr_calls==9
    assert [m.reg[i] for i in range(2,10)]==[0,0x55,0x57,0x56,7,7,MISA,0]
    assert m.csrs.read(0x340)==0x55
    # CSRRW x0 suppresses the architecturally visible CSR read.
    n=cpu([word(1,0,0x340,0),word(2,2,0x340,0),word(6,3,0x340,0)])
    n.step();assert n.csrs.reads==0 and n.csrs.writes==1
    n.step();n.step();assert n.csrs.reads==2 and n.csrs.writes==1
    # A nonzero source register containing zero still requests a write.
    for ins,mode in [(word(2,1,0xf14,2),3),(word(1,0,0xf14,0),3),
                     (word(2,1,0x340,0),1),(word(2,1,0x140,0),0),
                     (word(2,1,0x999,0),3)]:
        f=cpu([ins]);f.mode=mode;before={a:v.copy() for a,v in f.csrs.state.items()}
        try:f.step()
        except CSRFault:pass
        else:raise AssertionError('CSR fault not raised')
        assert f.pc==0 and f.steps==0 and f.reg[1]==0
        assert all(np.array_equal(v,f.csrs.state[a]) for a,v in before.items())
    # Supervisor aliases preserve machine-only fields; masks enforce WARL.
    bank=m.csrs
    bank.commit(bank.prepare(0x300,MASK));assert bank.read(0x300)==MSTATUS_MASK
    bank.commit(bank.prepare(0x100,0));assert bank.read(0x300)==MSTATUS_MASK&~SSTATUS_MASK
    bank.commit(bank.prepare(0x341,0x123));assert bank.read(0x341)==0x120
    bank.commit(bank.prepare(0x305,0x123));assert bank.read(0x305)==0x120
    bank.commit(bank.prepare(0x303,0x222));bank.commit(bank.prepare(0x104,MASK))
    assert bank.read(0x304)==0x222 and bank.read(0x104)==0x222
    old=bank.state[0x340].copy()
    try:bank.commit((0x340,123),ablate=True)
    except RuntimeError:pass
    else:raise AssertionError('ablation failed')
    assert np.array_equal(old,bank.state[0x340])
    # Randomized storage and readback in a full-width scratch CSR.
    for _ in range(64):
        v=rng.getrandbits(32);bank.commit(bank.prepare(0x340,v));assert bank.read(0x340)==v
    total=c.n+d.n+m.reg.neurons+m.control.neurons+m.counter.neurons+m.math.neurons+bank.neurons
    assert total<=131072
    report=dict(status='PASS',decode_vectors=checks,csr_ops=list(CSR_OPS.values()),
        mutable_csrs=len(MASKS),fixed_csrs=len(FIXED),aliases=len(ALIASES),
        random_storage_checks=64,access_fault_cases=5,ablation_rejected=True,
        guest_instructions=m.steps,guest_csr_ops=m.csr_calls,guest_seconds=round(guest_s,3),
        csr_budget_cells=bank.neurons,decoder_neurons=d.n,total_budget_cells=total,
        linux_booted=False,limitations='Preliminary CSR masks and storage; host address routing, access policy, no privilege transitions, architectural traps, counters, interrupts, MMU or complete privileged conformance. MISA advertises RV32IMA only; S/U bits remain clear until privilege support exists.')
    out=ROOT/'csr-r07';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

