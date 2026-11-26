"""R11: RV32 boot board, FDT and minimal SBI firmware boundary.

Host provides RAM/UART/timer and a clearly labelled SBI firmware service.
Guest instructions continue through the Q4 decoder, register bank and ALU.
This is a board smoke test, not a Linux boot.
"""
import json,struct,time,hashlib
from neural_alu_r01 import Circuit,MASK,ROOT
from neural_mmu_r10 import MMUCPU,MMUDecoder
from neural_irq_r09 import Timer
from neural_priv_r08 import GuestTrap
from neural_atomic_r06 import A_OPS
from neural_rv32i_r01 import I,S
RAM=0x80000000
RAM_SIZE=32*1024*1024
UART=0x10000000
FDT_ADDR=RAM+0x100000
def be(*v):return struct.pack('>'+'I'*len(v),*v)
def string(s):return s.encode()+b'\0'
def pad(b):return b+b'\0'*((-len(b))%4)
def dtb(size=RAM_SIZE):
    strings=bytearray(); offsets={}; body=bytearray()
    def begin(n):body.extend(be(1)+pad(string(n)))
    def end():body.extend(be(2))
    def prop(n,v):
        if n not in offsets:offsets[n]=len(strings);strings.extend(string(n))
        body.extend(be(3,len(v),offsets[n])+pad(v))
    begin('');prop('#address-cells',be(2));prop('#size-cells',be(2));prop('compatible',string('sc100k,virt'));prop('model',string('SC100K R11'))
    begin('chosen');prop('stdout-path',string('/soc/serial@10000000'));prop('bootargs',string('console=ttyS0 earlycon=uart8250,mmio,0x10000000'));end()
    begin('memory@80000000');prop('device_type',string('memory'));prop('reg',be(0,RAM,0,size));end()
    begin('cpus');prop('#address-cells',be(1));prop('#size-cells',be(0));prop('timebase-frequency',be(1000000))
    # Legacy ISA spelling: Linux 5.15 scans each letter (including Z names).
    # Zicsr and Zifencei remain implemented in the CPU.
    begin('cpu@0');prop('device_type',string('cpu'));prop('reg',be(0));prop('status',string('okay'));prop('compatible',string('riscv'));prop('riscv,isa',string('rv32ima'));prop('mmu-type',string('riscv,sv32'))
    begin('interrupt-controller');prop('#interrupt-cells',be(1));prop('interrupt-controller',b'');prop('compatible',string('riscv,cpu-intc'));prop('phandle',be(1));end();end();end()
    begin('soc');prop('#address-cells',be(2));prop('#size-cells',be(2));prop('compatible',string('simple-bus'));prop('ranges',b'')
    begin('serial@10000000');prop('compatible',string('ns16550a'));prop('reg',be(0,UART,0,8));prop('clock-frequency',be(3686400));prop('reg-shift',be(0));prop('reg-io-width',be(1));end();end();end();body.extend(be(9))
    reserve=b'\0'*16;off=40+len(reserve);so=off+len(body)
    return be(0xd00dfeed,so+len(strings),off,so,40,17,16,0,len(strings),len(body))+reserve+body+strings
class Board(MMUCPU):
    def __init__(self,c,d,image=b'',size=RAM_SIZE):
        super().__init__(c,d,b'')
        if size<2*1024*1024 or len(image)>size:raise ValueError('bad RAM/image size')
        self.ram=bytearray(size);self.ram[:len(image)]=image
        self.serial=bytearray();self.uart_regs=bytearray(8);self.sbi_calls=0
        self.deadline=None;self.poweroff=False;self.pc=RAM
    def in_ram(self,a,n):return RAM<=a and a+n<=RAM+len(self.ram) and a%n==0
    def phys_read(self,a,n,signed=False):
        if not self.in_ram(a,n):raise ValueError('physical access')
        off=a-RAM
        return int.from_bytes(self.ram[off:off+n],'little',signed=signed)&MASK
    def phys_write(self,a,n,v):
        if not self.in_ram(a,n):raise ValueError('physical access')
        off=a-RAM;self.ram[off:off+n]=(v&((1<<(8*n))-1)).to_bytes(n,'little')
        if self.reservation is not None and a<self.reservation+4 and self.reservation<a+n:self.reservation=None
    def physical(self,va,n,kind):
        ac,mis={'fetch':(1,0),'load':(5,4),'store':(7,6)}[kind]
        if va%n:raise GuestTrap(mis,va)
        pa=va if self._atomic_physical else self.translate(va,kind)
        if self.in_ram(pa,n):return pa
        if kind!='fetch' and ((UART<=pa<UART+8 and n==1 and self.mode!=0) or (Timer.contains(pa) and n==4 and self.mode==3)):return pa
        raise GuestTrap(ac,va)
    def load(self,a,n,signed=False):
        pa=self.physical(a,n,'load')
        if UART<=pa<UART+8:
            i=pa-UART
            return 0x60 if i==5 else 1 if i==2 else self.uart_regs[i]
        if Timer.contains(pa):return self.timer.read(pa)
        return self.phys_read(pa,n,signed)
    def store(self,a,n,v):
        pa=self.physical(a,n,'store')
        if UART<=pa<UART+8:
            i=pa-UART
            if i==0 and not self.uart_regs[3]&0x80:self.serial.append(v&255)
            else:self.uart_regs[i]=v&255
            return
        if Timer.contains(pa):self.timer.write(pa,v);self.reservation=None;self.sync_pending();return
        self.phys_write(pa,n,v)
    def access(self,a,n):
        if not self.in_ram(a,n):raise GuestTrap(7,a)
    def check_address(self,a,n,align_cause,access_cause):
        if a%n:raise GuestTrap(align_cause,a)
        if not self.in_ram(a,n):raise GuestTrap(access_cause,a)
    def prepare_instruction_extension(self,name,a,b,ins):
        if name not in A_OPS:return super().prepare_instruction_extension(name,a,b,ins)
        # Reuse ACPU reservation semantics with translated high RAM address.
        from neural_atomic_r06 import ACPU
        pa=self.physical(a,4,'load' if name=='LR.W' else 'store')
        if not self.in_ram(pa,4):raise GuestTrap(5 if name=='LR.W' else 7,a)
        self._atomic_addr=pa;self._atomic_physical=True
        try:result,store=ACPU.prepare_extension(self,name,pa,b)
        finally:self._atomic_physical=False
        return result,(a,store[1],store[2]) if store else None
    def boot(self,entry=RAM,fdt=FDT_ADDR):
        blob=dtb(len(self.ram))
        if not self.in_ram(fdt,1) or fdt+len(blob)>RAM+len(self.ram):raise ValueError('FDT outside RAM')
        off=fdt-RAM;self.ram[off:off+len(blob)]=blob
        self.reg[10]=0;self.reg[11]=fdt;self.mode=1;self.pc=entry
        self.put(0x303,0x222);self.put(0x306,2)
        return blob
    def sbi(self):
        eid=self.reg[17];fid=self.reg[16];a0=self.reg[10];a1=self.reg[11]
        err,val=0,0
        if eid==0x10:
            vals={0:2,1:0x5343,2:11,4:0,5:0,6:0}
            if fid==3:val=int(a0 in (0x10,0x54494d45,0x53525354))
            elif fid in vals:val=vals[fid]
            else:err=-2
        elif eid==0x54494d45 and fid==0:
            self.deadline=(a1<<32)|a0;self.timer.supervisor_timer=False;self.sync_pending()
        elif eid==0x53525354 and fid==0 and a0==0:
            self.poweroff=True
        elif eid==1: # legacy console putchar
            self.serial.append(a0&255);self.sbi_calls+=1;return
        else:err=-2
        self.reg[10]=err&MASK;self.reg[11]=val;self.sbi_calls+=1
    def special_step(self,name,ins,pc):
        if name=='ECALL' and self.mode==1 and not self.csrs.raw(0x302)&(1<<9):
            self.sbi();self.pc=self.alu('ADD',pc,4);self.steps+=1;return True
        return super().special_step(name,ins,pc)
    def sync_pending(self):
        if self.deadline is not None and self.timer.now>=self.deadline:self.timer.supervisor_timer=True
        super().sync_pending()
    def step(self):
        if self.poweroff:raise RuntimeError('board powered off')
        super().step()
def lui(rd,imm):return (imm&0xfffff000)|(rd<<7)|0x37
def main():
    c=Circuit().compile();d=MMUDecoder().compile()
    # Actual S-mode guest checks FDT magic (little endian load), prints
    # UART bytes, queries SBI version, requests timer and shutdown.
    words=[I(3,5,2,11,0),lui(6,UART),I(0x13,7,0,0,ord('O')),S(0,6,7,0),
           I(0x13,7,0,0,ord('K')),S(0,6,7,0),I(0x13,7,0,0,10),S(0,6,7,0),
           I(0x13,17,0,0,0x10),I(0x13,16,0,0,0),0x73,
           I(0x13,8,0,11,0),lui(17,0x54495000),I(0x13,17,0,17,-0x2bb),
           I(0x13,10,0,0,30),I(0x13,11,0,0,0),0x73,
           lui(17,0x53525000),I(0x13,17,0,17,0x354),I(0x13,10,0,0,0),0x73]
    m=Board(c,d,b''.join(struct.pack('<I',w) for w in words));blob=m.boot();t=time.perf_counter()
    while not m.poweroff and m.steps<100:m.step()
    assert m.poweroff and m.steps==len(words) and m.sbi_calls==3
    assert m.reg[5]==0xedfe0dd0 and m.reg[8]==2 and m.serial==b'OK\n' and m.deadline==30
    assert m.phys_read(FDT_ADDR,4)==0xedfe0dd0
    # High RAM atomics and translated high physical addresses.
    from neural_atomic_r06 import atomic_word
    from neural_mmu_r10 import pte,V,R,W,U
    q=Board(c,d,struct.pack('<II',atomic_word(2,5,1),atomic_word(3,6,1,2)))
    q.reg[1]=RAM+0x4000;q.reg[2]=99;q.phys_write(RAM+0x4000,4,7)
    q.step();q.step();assert q.reg[5]==7 and q.reg[6]==0 and q.phys_read(RAM+0x4000,4)==99
    q.phys_write(RAM+0x1000+256*4,4,pte(RAM+0x2000,V))
    q.phys_write(RAM+0x2000,4,pte(RAM+0x4000,V|R|W|U))
    q.put(0x180,0x80000000|((RAM+0x1000)>>12));q.mode=0
    assert q.load(0x40000000,4)==99
    q.store(0x40000000,4,123);assert q.phys_read(RAM+0x4000,4)==123
    # Check FDT header and bounds; parser validates all structural tokens.
    magic,total,off,so,rs,ver,last,boot,ss,sz=struct.unpack('>10I',blob[:40])
    assert magic==0xd00dfeed and total==len(blob) and so+ss==total and off+sz==so and rs==40
    pos=off;depth=0
    while pos<so:
        tag=struct.unpack_from('>I',blob,pos)[0];pos+=4
        if tag==1:
            stop=blob.index(0,pos);pos=(stop+4)&~3;depth+=1
        elif tag==2:depth-=1;assert depth>=0
        elif tag==3:
            n,idx=struct.unpack_from('>II',blob,pos);pos+=8
            assert idx<ss and pos+n<=so and depth>0;pos+=(n+3)&~3
        elif tag==9:assert depth==0 and pos==so;break
        else:raise AssertionError(tag)
    out=ROOT/'board-r11';out.mkdir(exist_ok=True);(out/'board.dtb').write_bytes(blob)
    report=dict(status='PASS',ram_bytes=len(m.ram),guest_steps=m.steps,sbi_calls=m.sbi_calls,
        serial=m.serial.decode(),fdt_bytes=len(blob),fdt_sha256=hashlib.sha256(blob).hexdigest(),
        high_ram_atomics=True,high_ram_sv32=True,guest_poweroff=True,seconds=round(time.perf_counter()-t,3),
        linux_booted=False,limitations='Host SBI interception, limited BASE/TIME/SRST and legacy putchar. No external IRQ UART RX, PLIC, actual guest firmware or Linux image. FDT is structurally checked, not validated by dtc or Linux.')
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()

