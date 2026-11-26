"""SC100K R03: clocked dual-rail recurrent register bank.

The sole register state is 32 x 32 x 2 x 5 spike cells. Fixed Q4 synaptic
bundles select the register, gate reads, and compute next-state feedback.
Host schedules phases and carries spike state across ticks; no integer shadow
registers or host address-indexed register reads/writes. Not a learned circuit.
"""
import json, random, struct, time
import numpy as np
from neural_alu_r01 import Circuit, MASK, ROOT
from neural_decoder_r02 import Decoder, NeuralCPU
from neural_rv32i_r01 import I,R,S,B,J

def fire(current):
    # Five identical postsynaptic cells per rail; GENESIS R02 threshold.
    return np.repeat((np.clip(current//3,-32,31)>=11)[...,None],5,axis=-1).astype(np.int8)

def encode(value,width):
    bits=np.array([(value>>i)&1 for i in range(width)],dtype=np.int8)
    return np.repeat(np.stack((1-bits,bits),axis=-1)[...,None],5,axis=-1)

def validate(rails):
    votes=rails.sum(axis=-1,dtype=np.int32)
    if not np.all(((votes[...,0]==5)&(votes[...,1]==0))|((votes[...,0]==0)&(votes[...,1]==5))):
        raise RuntimeError('invalid dual-rail register state')
    return votes

def decode(rails):
    votes=validate(rails)
    return sum(int(v==5)<<i for i,v in enumerate(votes[:,1]))

class RegisterBank:
    # Logical cells: address 50, selectors 320, data 320, state 10240,
    # read gates 10240, read output 320, hold/write gates 20480,
    # constant write-inhibit rails 10. State cells are reused on feedback.
    neurons=41980
    def __init__(self):
        self._state=np.repeat(encode(0,32)[None,...],32,axis=0)
        self.reads=0;self.writes=0
        # Static synaptic wiring, not a per-request lookup.
        self._w=np.zeros((32,2,5,2),dtype=np.int8)
        for r in range(32):
            for i in range(5):
                bit=(r>>i)&1
                self._w[r,1,i,bit]=2
                self._w[r,1,i,1-bit]=-7
                self._w[r,0,i,1-bit]=7
    @staticmethod
    def _address(i):
        if type(i) is not int or not 0<=i<32:raise ValueError('bad register address')
    def select(self,i,ablate=False):
        self._address(i)
        src=encode(i,5).sum(axis=-1,dtype=np.int32)
        currents=np.einsum('rabi,bi->ra',self._w,src,dtype=np.int32)
        rails=fire(currents*(not ablate))
        votes=validate(rails)
        if int((votes[:,1]==5).sum())!=1:raise RuntimeError('not one-hot register selection')
        return rails
    def read(self,i,*,ablate=False):
        selected=self.select(i).sum(axis=-1,dtype=np.int32)[:,1]
        # AND on selected and state rail; unselected cells emit neither rail.
        gated=fire((4*(selected[:,None,None]+self._state.sum(axis=-1,dtype=np.int32)))*(not ablate))
        # OR over all register rows through convergent Q4 synapses.
        out=fire(7*gated.sum(axis=(0,3),dtype=np.int32))
        result=decode(out)
        self.reads+=1
        return result
    def write(self,i,value,*,ablate=False):
        sel=self.select(i).copy()
        # Constant wired hold rail for x0; no host conditional on address.
        sel[0]=encode(0,1)[0]
        votes=sel.sum(axis=-1,dtype=np.int32)
        data=encode(value&MASK,32).sum(axis=-1,dtype=np.int32)
        old=self._state.sum(axis=-1,dtype=np.int32)
        hold=fire(4*(old+votes[:,0,None,None])*(not ablate))
        put=fire(4*(data[None,:,:]+votes[:,1,None,None])*(not ablate))
        candidate=fire(7*(hold.sum(axis=-1,dtype=np.int32)+put.sum(axis=-1,dtype=np.int32)))
        validate(candidate)
        # Recurrent spike feedback is the sole persistent register state.
        self._state=candidate
        self.writes+=1
    def __getitem__(self,i):return self.read(i)
    def __setitem__(self,i,v):self.write(i,v)
class RegisterCPU(NeuralCPU):
    def __init__(self,circuit,decoder,image):
        super().__init__(circuit,decoder,image)
        self.reg=RegisterBank()
    def rd(self,i,v):
        self.reg.write(i,v&MASK)

def main():
    bank=RegisterBank();rng=random.Random(3109);ref=[0]*32
    t=time.perf_counter(); checks=0
    for i in range(512):
        reg=rng.randrange(32);value=rng.getrandbits(32)
        bank.write(reg,value)
        if reg:ref[reg]=value
        assert bank.read(reg)==ref[reg];checks+=1
        if i%16==0:
            for j in range(32):
                assert bank.read(j)==ref[j];checks+=1
    # Hold a full register file across idle intervals (no clock edges).
    for j in range(32):
        assert bank.read(j)==ref[j];checks+=1
    bank.write(0,MASK);assert bank.read(0)==0
    old=bank._state.copy()
    for f in (lambda:bank.select(3,True),lambda:bank.read(3,ablate=True),lambda:bank.write(3,42,ablate=True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation did not fail')
    assert np.array_equal(old,bank._state)
    elapsed=time.perf_counter()-t
    words=[I(0x13,1,0,0,1),I(0x13,2,0,0,17),I(0x13,3,0,0,0),
        R(3,0,3,1),I(0x13,1,0,1,1),B(1,1,2,-8),
        I(0x13,4,0,0,1024),S(2,4,3,0),I(0x03,5,2,4,0),
        J(6,20),I(0x13,8,0,0,-1),S(0,4,8,4),I(0x03,9,0,4,4),
        J(0,12),I(0x13,7,0,5,1),I(0x67,0,0,6,0),
        B(1,3,5,8),I(0x13,0,0,0,123),0x00100073]
    c=Circuit().compile();d=Decoder().compile()
    cpu=RegisterCPU(c,d,b''.join(struct.pack('<I',w) for w in words))
    start=time.perf_counter()
    while not cpu.halted and cpu.steps<1000:cpu.step()
    seconds=time.perf_counter()-start
    reads=cpu.reg.reads;writes=cpu.reg.writes
    assert cpu.halted and cpu.steps==cpu.decode_calls==64 and cpu.reg[3]==cpu.reg[5]==136 and cpu.reg[7]==137 and cpu.reg[9]==MASK and cpu.reg[0]==0
    assert cpu.load(1024,4)==136
    assert c.n+d.n+bank.neurons<=131072
    report=dict(status='PASS',registers=32,register_bits=1024,register_neurons=bank.neurons,
        total_circuit_neurons=c.n+d.n+bank.neurons,random_writes=512,read_checks=checks,
        errors=0,ablation_rejected=True,x0_immutable=True,register_test_seconds=round(elapsed,3),
        guest_instructions=cpu.steps,guest_ips=round(cpu.steps/seconds,2),neural_reads=reads,
        neural_writes=writes,neural_decode_calls=cpu.decode_calls,neural_alu_calls=cpu.alu_calls,
        linux_booted=False,limitations='Clocked recurrent spike-state bank, vectorized fixed Q4 bundle equations; host retains spike arrays and schedules phases, extracts fields, dispatches control and implements RAM. No autonomous recurrent GENESIS integration, ISA conformance or Linux yet.')
    out=ROOT/'register-r03';out.mkdir(exist_ok=True)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

