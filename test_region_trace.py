import argparse,hashlib,json,time
from pathlib import Path
from neural_workstation_r40 import Workstation
from neural_uart_r36 import fresh,restore,state
from neural_linux_r12 import fixture
from neural_board_r11 import RAM,UART
from neural_devices_r30 import DISK,GPU
from neural_framebuffer_r32 import FB
from neural_io_r33 import PAL,KEY
from neural_rv32i_r01 import I,R,S
from neural_atomic_r06 import atomic_word
from region_diagnostics import REGIONS,EXPLICIT
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--cc',default='gcc');p.add_argument('--ld');p.add_argument('--output',type=Path)
a=p.parse_args();a.coordinates=None;a.fixture=True;a.interval=300
data=fixture();h=hashlib.sha256(data).hexdigest();b=fresh(data,a);w=Workstation(b,a,h)
# Set up a genuine guest program before enabling trace. These instructions
# cover memory-mapped devices, math, atomic, CSR and normal RAM.
code=RAM+0x4000;word=RAM+0x8000
for i,v in enumerate([word,42,DISK,FB,PAL,GPU,KEY,UART],1):b.reg[i]=v
words=[S(2,1,2,0),I(3,10,2,1,0),atomic_word(0,11,1,2),R(12,0,2,2,1),
       S(2,3,2,0),I(3,13,2,3,0),S(2,4,2,0),S(2,5,2,0),S(2,6,2,0),
       I(3,14,2,7,4),I(3,15,4,8,0),0x0000006f]
for i,v in enumerate(words):b.phys_write(code+4*i,4,v)
b.mode=3;b.pc=code
w.execute('key',dict(code=30,pressed=True));w.execute('serial',dict(text='Q'))
for _ in range(len(words)):w.step(True)
w.publish()
missing=[k for k in REGIONS if not w.snapshot['trace'][k]['count']]
assert not missing,missing
assert b.traps==0,(b.pc,b.traps,w.trap)
assert b.reg[12]==1764 and b.reg[13]==42 and b.reg[14]==286 and b.reg[15]==81
assert w.snapshot['trace']['ram']['latest']['route'] is not None
for k in REGIONS:
 t=w.snapshot['trace'][k];assert len(t['history'])<=48 and t['latest']['seq']==t['history'][-1]['seq']
before={k:v['count'] for k,v in w.snapshot['trace'].items()};w.publish()
assert before=={k:v['count'] for k,v in w.snapshot['trace'].items()}
w.persist()
assert a.checkpoint.exists()
r=dict(status='PASS',regions=len(REGIONS),events=before,retired=b.steps,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),idle_no_events=True)
print(json.dumps(r));a.output.write_text(json.dumps(r,indent=2)) if a.output else None

