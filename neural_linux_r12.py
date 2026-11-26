"""R12: bounded Linux Image entry runner; no alternate CPU backend.

Reads RISC-V Linux Image header, loads kernel and zero-fill into high RAM,
passes FDT and runs the existing neural board. Stops on bounded instruction
budget or first unexpected trap, preserving diagnostic state.
"""
import argparse,hashlib,json,struct,time
from pathlib import Path
from neural_alu_r01 import Circuit,ROOT
from neural_board_r11 import Board,RAM,dtb
from neural_mmu_r10 import MMUDecoder
MAGIC1=b'RISCV\0\0\0'
MAGIC2=b'RSC\x05'
def image_header(data):
    if len(data)<64 or data[48:56]!=MAGIC1 or data[56:60]!=MAGIC2:
        raise ValueError('not a supported RISC-V Linux Image header')
    offset,size,flags,version=struct.unpack_from('<QQQI',data,8)
    if flags!=0 or version>>16 or not size or size<len(data):
        raise ValueError('unsupported header flags, version or size')
    if offset%(4*1024*1024):raise ValueError('RV32 image alignment')
    return dict(offset=offset,image_size=size,flags=flags,version=version)
def load_image(data,ram_mb=64):
    h=image_header(data);size=ram_mb*1024*1024
    fdt=RAM+size-65536
    if h['offset']+h['image_size']>size-65536:raise ValueError('kernel overlaps FDT or exceeds RAM')
    c=Circuit().compile();d=MMUDecoder().compile();b=Board(c,d,size=size)
    start=h['offset'];b.ram[start:start+len(data)]=data
    b.boot(RAM+start,fdt)
    b.put(0x180,0);b.put(0x302,0xb1ff)
    return b,h
def run(data,budget=512,ram_mb=64):
    b,h=load_image(data,ram_mb)
    start=time.perf_counter();trace=[];entry_trace=[];why='budget';fault=None
    for _ in range(budget):
        pc=b.pc;mode=b.mode
        # Diagnostic fetch only from physical RAM in bare mode. No extra
        # translation or A/D side effects for paged guest execution.
        ins=b.phys_read(pc,4) if not b.csrs.raw(0x180)&0x80000000 and b.in_ram(pc,4) else None
        trace.append(dict(pc=hex(pc),ins=hex(ins) if ins is not None else None,mode=mode))
        if len(entry_trace)<16:entry_trace.append(trace[-1])
        trace=trace[-32:]
        before=b.traps
        try:b.step()
        except RuntimeError as e:
            why='host_error';fault=str(e);break
        if b.traps!=before:
            why='guest_trap'
            target=b.mode
            fault=dict(cause=hex(b.csrs.raw(0x342 if target==3 else 0x142)),
                       tval=hex(b.csrs.raw(0x343 if target==3 else 0x143)),
                       epc=hex(b.csrs.raw(0x341 if target==3 else 0x141)),
                       target=hex(b.pc),mode=target)
            break
        if b.poweroff:why='poweroff';break
    return dict(status='ENTRY_EXECUTED' if b.steps else 'NO_RETIREMENT',
        image_sha256=hashlib.sha256(data).hexdigest(),header=h,ram_bytes=len(b.ram),
        retired=b.steps,alu_calls=b.alu_calls,decoder_calls=b.decode_calls,
        stop=why,fault=fault,pc=hex(b.pc),mode=b.mode,serial=b.serial.decode(errors='replace'),
        seconds=round(time.perf_counter()-start,3),entry_trace=entry_trace,trace=trace,
        register_snapshot={f'x{i}':hex(b.reg[i]) for i in (10,11,13,14)},linux_booted=False,
        limitations='Entry probe only. Stops at first guest trap (including legitimate traps). No Linux shell claim; host SBI, peripheral and separate neural circuits.')
def ablation(data):
    b,_=load_image(data,16);original=b.c.run
    b.c.run=lambda op,a,v:original(op,a,v,ablate=True)
    pc=b.pc
    try:b.step()
    except RuntimeError as e:
        assert b.steps==0 and b.pc==pc and "invalid dual-rail" in str(e)
        return str(e)
    raise AssertionError("neural ablation did not stop execution")
def fixture():
    # Header-compatible executable fixture, NOT Linux. Jump over header,
    # write one byte through guest UART and loop.
    from neural_rv32i_r01 import J,I,S
    from neural_board_r11 import lui,UART
    words=[J(0,64),0]
    h=struct.pack('<IIQQQIIQ8s4sI',*words,0x400000,4096,0,2,0,0,MAGIC1,MAGIC2,0)
    return h+b''.join(struct.pack('<I',w) for w in [lui(6,UART),I(0x13,7,0,0,88),S(0,6,7,0),J(0,0)])
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path);p.add_argument('--budget',type=int,default=512);p.add_argument('--ram-mb',type=int,default=64);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.budget<=100000 or not 16<=a.ram_mb<=512:p.error('bounded budget/RAM required')
    if a.image:
        data=a.image.read_bytes();r=run(data,a.budget,a.ram_mb)
        r['fixture']=False
        r['ablation']=ablation(data)
    else:
        data=fixture();r=run(data,5,16);r['fixture']=True
        assert r['retired']==5 and r['serial']=='X' and r['stop']=='budget'
        for bad in (b'',data[:48]+b'BADMAGIC'+data[56:],data[:8]+struct.pack('<Q',123)+data[16:]):
            try:image_header(bad)
            except ValueError:pass
            else:raise AssertionError('malformed header accepted')
        r['status']='PASS'
    s=json.dumps(r,indent=2)
    if a.output:a.output.parent.mkdir(exist_ok=True,parents=True);a.output.write_text(s+'\n')
    print(s)
if __name__=='__main__':main()

