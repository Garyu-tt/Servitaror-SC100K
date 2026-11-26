"""R15: verify the legacy Linux ISA advertisement at cold boot.

Linux 5.15-era ISA scanner treats letters in Z-extension names as
single-letter extensions. Use the legacy IMA spelling for this board.
The implemented Zicsr/Zifencei instructions are still available.
"""
import argparse,hashlib,json,struct
from pathlib import Path
from neural_board_r11 import dtb,RAM
from neural_linux_r12 import fixture
from neural_uart_r36 import fresh,save
from neural_resume_r13 import advance
from neural_tlb_r17 import PageCache
from neural_alu_r01 import oracle
from neural_math_r05 import oracle as moracle

def isa(blob):
    magic,size,off,strings,*_=struct.unpack_from('>10I',blob)
    if magic!=0xd00dfeed or size>len(blob):raise ValueError('FDT header')
    i=off;found=[]
    while i<strings:
        tag=struct.unpack_from('>I',blob,i)[0];i+=4
        if tag==1:i=(blob.index(0,i)+4)&~3
        elif tag==2 or tag==4:continue
        elif tag==3:
            n,key=struct.unpack_from('>II',blob,i);i+=8
            name=blob[strings+key:blob.index(0,strings+key)]
            if name==b'riscv,isa':found.append(blob[i:i+n].rstrip(b'\0').decode())
            i=(i+n+3)&~3
        elif tag==9:break
        else:raise ValueError('FDT token')
    if len(found)!=1:raise ValueError('ISA property')
    return found[0]
def test():
    advertised=isa(dtb(64<<20))
    assert advertised=='rv32ima'
    # Reproduce the historical character scan, not guest execution.
    scan=lambda s:''.join(sorted(set(c for c in s[4:] if 'a'<=c<'x')))
    assert scan('rv32ima_zicsr_zifencei')=='acefimnrs'
    assert scan(advertised)=='aim'
    return dict(status='PASS',advertised=advertised,legacy_bad_scan=scan('rv32ima_zicsr_zifencei'),legacy_fixed_scan=scan(advertised))
def run(a):
    data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b=fresh(data,a)
    addr=b.reg[11];blob=b.memory.export_bytes(addr-RAM,4096);advertised=isa(blob)
    assert advertised=='rv32ima' and b.steps==b.boundaries==0 and not b.serial
    assert b.csrs.raw(0x301)&((1<<2)|(1<<3)|(1<<5))==0
    cache=PageCache(b);old=b.c.run;oldm=b.math.run;checks=0
    def ar(op,x,y,ablate=False):
        nonlocal checks
        v=old(op,x,y,ablate)
        if not ablate:assert v[0]==oracle(op,x,y);checks+=1
        return v
    def mr(op,x,y,ablate=False):
        nonlocal checks
        v=oldm(op,x,y,ablate)
        if not ablate:assert v==moracle(op,x,y);checks+=1
        return v
    b.c.run=ar;b.math.run=mr
    r=advance(b,a.steps);save(b,a.checkpoint,h)
    r.update(cold=True,advertised_isa=advertised,image_sha256=h,oracle_checks=checks,no_guest_bytearray=b.ram is None,
        checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),cache=cache.report())
    return r
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,default=Path('/tmp/sc500k-legacy'));p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--steps',type=int,default=512);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.steps<=50000 or (a.image and not a.checkpoint):p.error('bounded steps/checkpoint required')
    r=run(a) if a.image else test()
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

