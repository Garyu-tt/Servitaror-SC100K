"""SC100K R06: single-hart RV32A word operations.

The host RAM bus serializes requests and maintains reservation metadata.
AMO arithmetic/comparison and min/max selection use existing neural gates.
No multi-hart coherence or privileged traps yet.
"""
import json, random, struct, time
from neural_alu_r01 import Circuit, MASK, ROOT
from neural_math_r05 import MDecoder, MCPU
from neural_register_r03 import encode, decode
from neural_rv32i_r01 import I,S

A_CODES={0x02:'LR.W',0x03:'SC.W',0x01:'AMOSWAP.W',0x00:'AMOADD.W',
         0x04:'AMOXOR.W',0x0c:'AMOAND.W',0x08:'AMOOR.W',
         0x10:'AMOMIN.W',0x14:'AMOMAX.W',0x18:'AMOMINU.W',0x1c:'AMOMAXU.W'}
A_OPS=tuple(A_CODES.values())
class ADecoder(MDecoder):
    patterns=MDecoder.patterns+tuple(
        (name,0xf9f0707f if code==2 else 0xf800707f,(code<<27)|0x202f)
        for code,name in A_CODES.items())

def atomic_word(code,rd,rs1,rs2=0,aqrl=0):
    return (code<<27)|(aqrl<<25)|(rs2<<20)|(rs1<<15)|0x2000|(rd<<7)|0x2f

class ACPU(MCPU):
    def __init__(self,circuit,decoder,image):
        super().__init__(circuit,decoder,image)
        self.reservation=None
        self.atomic_calls=0;self.atomic_selects=0
    def store(self,a,n,v):
        super().store(a,n,v)
        # Conservative single-hart policy: any overlapping local write
        # invalidates the reservation, including subword stores.
        if self.reservation is not None and a<self.reservation+4 and self.reservation<a+n:
            self.reservation=None
    def prepare_extension(self,name,a,b):
        if name not in A_OPS:return super().prepare_extension(name,a,b)
        self.access(a,4)
        if name=='LR.W':return self.load(a,4),None
        if name=='SC.W':
            # Reservation metadata belongs to the serialized memory bus.
            ok=self.reservation==a
            return (0,(a,4,b)) if ok else (1,None)
        old=self.load(a,4)
        direct={'AMOADD.W':'ADD','AMOXOR.W':'XOR','AMOAND.W':'AND','AMOOR.W':'OR'}
        if name=='AMOSWAP.W':new=b
        elif name in direct:new=self.alu(direct[name],old,b)
        else:
            less=self.alu('SLTU' if name.endswith('U.W') else 'SLT',old,b)
            # M-lane's existing Q4 mux computes selection; no host min/max.
            self.math.ablate=False
            left,right=encode(old,32),encode(b,32)
            if name in ('AMOMIN.W','AMOMINU.W'):left,right=right,left
            new=decode(self.math.mux(encode(less,1)[0],left,right))
            self.atomic_selects+=1
        return old,(a,4,new)
    def finish_extension(self,name,a,b):
        if name=='LR.W':self.reservation=a
        elif name=='SC.W':self.reservation=None
        if name in A_OPS:self.atomic_calls+=1

def oracle(name,a,b):
    s=lambda x:x if x<1<<31 else x-(1<<32)
    return {'AMOSWAP.W':lambda:b,'AMOADD.W':lambda:(a+b)&MASK,
        'AMOXOR.W':lambda:a^b,'AMOAND.W':lambda:a&b,'AMOOR.W':lambda:a|b,
        'AMOMIN.W':lambda:min(a,b,key=s),'AMOMAX.W':lambda:max(a,b,key=s),
        'AMOMINU.W':lambda:min(a,b),'AMOMAXU.W':lambda:max(a,b)}[name]()

def main():
    d=ADecoder().compile();c=Circuit().compile()
    rng=random.Random(3112);decode_checks=0
    for code,name in A_CODES.items():
        for aqrl in range(4):
            for _ in range(8):
                w=atomic_word(code,rng.randrange(32),rng.randrange(32),0 if code==2 else rng.randrange(32),aqrl)
                assert d.decode(w)==name;decode_checks+=1
                if code==2:
                    assert d.decode(w|(1<<20)) is None;decode_checks+=1
    assert d.decode(atomic_word(0x1f,1,2,3)) is None
    assert d.decode(atomic_word(0,1,2,3)^0x3000) is None
    def cpu(words):return ACPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    m=cpu([0x00100073]);addr=1024
    pairs=[(0,0),(MASK,1),(0x80000000,0x7fffffff),(0x7fffffff,MASK),(0x80000000,MASK),(3,7)]
    pairs += [(rng.getrandbits(32),rng.getrandbits(32)) for _ in range(8)]
    t=time.perf_counter();amo_checks=0
    for name in A_OPS[2:]:
        for old,b in pairs:
            m.store(addr,4,old)
            val,store=m.prepare_extension(name,addr,b)
            assert m.load(addr,4)==old and val==old
            m.store(*store);m.finish_extension(name,addr,b)
            assert m.load(addr,4)==oracle(name,old,b)
            amo_checks+=1
    # guest: LR/SC success; repeated SC fails; overlapping SB invalidates;
    # all nine AMO variants execute through decoder and registers.
    words=[I(0x13,1,0,0,1024),I(0x13,2,0,0,7),
        atomic_word(2,3,1),atomic_word(3,4,1,2),atomic_word(3,5,1,2),
        atomic_word(2,6,1),S(0,1,2,1),atomic_word(3,7,1,2)]
    words.extend(atomic_word(code,8+i,1,2,i%4) for i,code in enumerate(list(A_CODES)[2:]))
    words.append(0x00100073)
    g=cpu(words);g.store(addr,4,11)
    start=time.perf_counter()
    while not g.halted and g.steps<100:g.step()
    guest_s=time.perf_counter()-start
    assert g.halted and g.steps==18 and g.atomic_calls==14
    assert [g.reg[i] for i in range(3,8)]==[11,0,1,7,1]
    expected=0x707
    for i,name in enumerate(A_OPS[2:]):
        assert g.reg[8+i]==expected
        expected=oracle(name,expected,7)
    assert g.load(addr,4)==expected
    # Wrong address SC clears reservation, disjoint stores preserve it.
    h=cpu([atomic_word(2,3,1),S(2,1,2,4),atomic_word(3,4,1,2),atomic_word(2,5,1),atomic_word(3,6,2,1),atomic_word(3,7,1,2)])
    h.reg[1]=addr;h.reg[2]=addr+4
    for _ in range(6):h.step()
    assert h.reg[4]==0 and h.reg[6]==h.reg[7]==1 and h.reservation is None
    # Misalignment is detected before side effects.
    f=cpu([atomic_word(0,3,1,2)]);f.reg[1]=addr+1;f.reg[2]=99;before=bytes(f.ram)
    try:f.step()
    except RuntimeError as e:assert 'misaligned' in str(e)
    else:raise AssertionError('misaligned AMO accepted')
    assert bytes(f.ram)==before and f.pc==0 and f.reg[3]==0 and f.steps==0
    total=c.n+d.n+g.reg.neurons+g.control.neurons+g.counter.neurons+g.math.neurons
    assert total<=131072
    report=dict(status='PASS',atomic_operations=list(A_OPS),decode_vectors=decode_checks+2,
        amo_vectors=amo_checks,guest_instructions=g.steps,guest_atomic_ops=g.atomic_calls,
        guest_seconds=round(guest_s,3),neural_minmax_selects=g.atomic_selects,
        lr_sc_success_failure=True,overlap_invalidation=True,disjoint_store_preserves=True,
        misalignment_no_side_effect=True,decoder_neurons=d.n,total_budget_cells=total,
        linux_booted=False,limitations='Single hart serialized RAM; aq/rl decoded, no separate ordering effect on synchronous bus. Host reservation metadata and bus atomicity; no multi-hart, DMA, MMIO, privileged traps or continuous integrated GENESIS.')
    out=ROOT/'atomic-r06';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

