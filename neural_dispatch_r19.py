"""R19: OP-IMM dispatch and fail-closed regression."""
import json,struct,random
from neural_alu_r01 import Circuit,OPS,MASK,oracle
from neural_decoder_r02 import Decoder,NeuralCPU
from neural_control_r04 import ControlCPU
from neural_rv32i_r01 import CPU,I
from neural_linux_r12 import fixture,load_image
from neural_recurrent_r15 import setup
from pathlib import Path
import argparse

def tests(args):
    c=Circuit().compile();d=Decoder().compile()
    vectors=[(0,1),(1,0),(0xffffffff,0),(0x80000000,-1),(0xffffffff,-1),(0,0),(0x7fffffff,-2048),(0x80000000,2047)]
    r=random.Random(31019)
    vectors += [(r.getrandbits(32),r.randrange(-2048,2048)) for _ in range(24)]
    checks=0
    for cls in (CPU,NeuralCPU,ControlCPU):
        for f,op in enumerate(('ADD','SLL','SLT','SLTU','XOR','SRL','OR','AND')):
            for x,imm in vectors:
                if f in (1,5):imm&=31
                image=struct.pack('<I',I(0x13,2,f,1,imm))
                b=cls(c,image) if cls is CPU else cls(c,d,image)
                b.reg[1]=x;b.step()
                assert b.reg[2]==oracle(op,x,imm&MASK),(cls.__name__,op,x,imm,b.reg[2])
                checks+=1
    b,_=load_image(fixture(),16);e=setup(b,args);e.attach(b)
    for x,imm in vectors:
        b.phys_write(0x80000000,4,I(0x13,2,3,1,imm))
        b.pc=0x80000000;b.mode=3;b.reg[1]=x;b.step()
        assert b.reg[2]==oracle('SLTU',x,imm&MASK);checks+=1
    for fn in (lambda:c.run('SLTI',0,1),lambda:b.c.run('SLTI',0,1),lambda:b.alu('SLTI',0,1)):
        try:fn()
        except ValueError:pass
        else:raise AssertionError('unrecognized ALU operation accepted')
    print(json.dumps(dict(status='PASS',op_imm_checks=checks,unknown_op_rejected=3)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');tests(p.parse_args())

