#!/usr/bin/env python3
"""RV32I instruction front end for the Cogitator Q4 ALU.
Host owns instruction decode, register file, RAM and device/exception control.
Arithmetic, bitwise operations, branch predicates and effective addresses use
only the spiking ALU. Not a Linux-capable core yet.
"""
import json,time,struct
from pathlib import Path
from neural_alu_r01 import Circuit,MASK,ROOT,OPS
def sext(v,bits):
    return v-(1<<bits) if v&(1<<(bits-1)) else v
class CPU:
    def __init__(self,circuit,image):
        self.c=circuit; self.reg=[0]*32;self.pc=0
        self.ram=bytearray(65536);self.ram[:len(image)]=image
        self.steps=0;self.alu_calls=0;self.halted=False
    def alu(self,op,a,b):
        if op not in OPS:raise ValueError(f"unknown ALU operation: {op}")
        self.alu_calls+=1
        return self.c.run(op,a&MASK,b&MASK)[0]
    def rd(self,i,v):
        if i:self.reg[i]=v&MASK
    def access(self,a,n):
        if a%n or a+n>len(self.ram):raise RuntimeError('misaligned/out of bounds memory')
    def load(self,a,n,signed=False):
        self.access(a,n);return int.from_bytes(self.ram[a:a+n],'little',signed=signed)&MASK
    def store(self,a,n,v):
        self.access(a,n);self.ram[a:a+n]=(v&((1<<(8*n))-1)).to_bytes(n,'little')
    def step(self):
        if self.halted:raise RuntimeError('already halted')
        ins=self.load(self.pc,4);op=ins&127
        rd=(ins>>7)&31;f3=(ins>>12)&7;r1=(ins>>15)&31;r2=(ins>>20)&31;f7=ins>>25
        a=self.reg[r1];b=self.reg[r2];imm=sext(ins>>20,12)
        pc=self.pc;nxt=self.alu('ADD',pc,4)
        ops={0:'ADD',1:'SLL',2:'SLT',3:'SLTU',4:'XOR',5:'SRL',6:'OR',7:'AND'}
        if op==0x33:
            if f7 not in (0,32) or f7==32 and f3 not in (0,5):raise RuntimeError('illegal OP')
            self.rd(rd,self.alu({0:'SUB',5:'SRA'}.get(f3,ops[f3]) if f7==32 else ops[f3],a,b))
        elif op==0x13:
            if f3 in (1,5):
                if f7 not in ((0,) if f3==1 else (0,32)):raise RuntimeError('illegal shift')
                self.rd(rd,self.alu('SRA' if f7==32 else ops[f3],a,r2))
            else:self.rd(rd,self.alu(ops[f3],a,imm))
        elif op==0x03:
            n,sgn={0:(1,True),1:(2,True),2:(4,False),4:(1,False),5:(2,False)}[f3]
            self.rd(rd,self.load(self.alu('ADD',a,imm),n,sgn))
        elif op==0x23:
            off=sext(((ins>>25)<<5)|rd,12)
            self.store(self.alu('ADD',a,off),{0:1,1:2,2:4}[f3],b)
        elif op==0x63:
            off=sext(((ins>>31)<<12)|(((ins>>7)&1)<<11)|(((ins>>25)&63)<<5)|(((ins>>8)&15)<<1),13)
            if f3 in (0,1):take=(self.alu('XOR',a,b)==0)==(f3==0)
            elif f3 in (4,5):take=bool(self.alu('SLT',a,b))==(f3==4)
            elif f3 in (6,7):take=bool(self.alu('SLTU',a,b))==(f3==6)
            else:raise RuntimeError('illegal branch')
            if take:nxt=self.alu('ADD',pc,off)
        elif op==0x37:self.rd(rd,ins&0xfffff000)
        elif op==0x17:self.rd(rd,self.alu('ADD',pc,ins&0xfffff000))
        elif op==0x6f:
            off=sext(((ins>>31)<<20)|(((ins>>12)&255)<<12)|(((ins>>20)&1)<<11)|(((ins>>21)&1023)<<1),21)
            self.rd(rd,nxt);nxt=self.alu('ADD',pc,off)
        elif op==0x67 and f3==0:
            target=self.alu('AND',self.alu('ADD',a,imm),0xfffffffe)
            self.rd(rd,nxt);nxt=target
        elif op==0x0f and f3==0:pass # single ordered hart, fence is a no-op
        elif ins==0x00100073:self.halted=True # debug harness EBREAK
        else:raise RuntimeError(f'unsupported instruction {ins:08x} at {pc:08x}')
        self.pc=nxt;self.steps+=1
        assert self.reg[0]==0
def I(op,rd,f3,rs,imm):return ((imm&4095)<<20)|(rs<<15)|(f3<<12)|(rd<<7)|op
def R(rd,f3,a,b,f7=0):return (f7<<25)|(b<<20)|(a<<15)|(f3<<12)|(rd<<7)|0x33
def S(f3,a,b,imm):return (((imm>>5)&127)<<25)|(b<<20)|(a<<15)|(f3<<12)|((imm&31)<<7)|0x23
def B(f3,a,b,imm):
    return (((imm>>12)&1)<<31)|(((imm>>5)&63)<<25)|(b<<20)|(a<<15)|(f3<<12)|(((imm>>1)&15)<<8)|(((imm>>11)&1)<<7)|0x63
def J(rd,imm):return (((imm>>20)&1)<<31)|(((imm>>1)&1023)<<21)|(((imm>>11)&1)<<20)|(((imm>>12)&255)<<12)|(rd<<7)|0x6f
def main():
    # sum 1..16, save to RAM, load back, call subroutine, sign-extended load,
    # verify link registers, zero register and an error-branch not taken.
    words=[
        I(0x13,1,0,0,1),I(0x13,2,0,0,17),I(0x13,3,0,0,0),
        R(3,0,3,1),I(0x13,1,0,1,1),B(1,1,2,-8),
        I(0x13,4,0,0,1024),S(2,4,3,0),I(0x03,5,2,4,0),
        J(6,20),I(0x13,8,0,0,-1),S(0,4,8,4),I(0x03,9,0,4,4),
        J(0,12),I(0x13,7,0,5,1),I(0x67,0,0,6,0),
        B(1,3,5,8),I(0x13,0,0,0,123),0x00100073]
    image=b''.join(struct.pack('<I',w) for w in words)
    c=Circuit().compile();cpu=CPU(c,image)
    t=time.perf_counter()
    while not cpu.halted and cpu.steps<1000:cpu.step()
    elapsed=time.perf_counter()-t
    assert cpu.halted and cpu.steps==64,(cpu.steps,cpu.reg)
    assert cpu.reg[3]==cpu.reg[5]==136 and cpu.reg[7]==137
    assert cpu.reg[6]==40 and cpu.reg[9]==MASK and cpu.load(1024,4)==136
    out=ROOT/'rv32i-r01';out.mkdir(exist_ok=True)
    (out/'sum16.bin').write_bytes(image)
    report=dict(status='PASS',instructions=cpu.steps,alu_calls=cpu.alu_calls,
                seconds=round(elapsed,3),instructions_per_second=round(cpu.steps/elapsed,2),
                sum_register=cpu.reg[3],memory_word=cpu.load(1024,4),subroutine_result=cpu.reg[7],
                signed_byte=hex(cpu.reg[9]),host_services=['decode','register storage','memory','control'],
                neural_services=['ALU','PC arithmetic','effective addresses','branch comparison'],
                linux_booted=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

