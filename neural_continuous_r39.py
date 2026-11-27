"""Bounded continuous execution of the existing SC500K neural board.
One restore per run. No guest instruction substitution. Independent arithmetic
oracle remains enabled. Checkpoint on interval, normal stop or guest trap.
Host errors fail closed, retaining the last good checkpoint.
"""
import argparse,hashlib,json,time,os
from pathlib import Path
from neural_uart_r36 import restore,save,PageCache,oracle,moracle

def advance_chunk(b,budget):
    start=time.perf_counter();retired=b.steps;reason='budget';fault=None
    for _ in range(budget):
        before=b.traps
        try:b.step()
        except RuntimeError as e:reason='host_error';fault=str(e);break
        if b.traps!=before:
            reason='guest_trap';mode=b.mode
            fault={k:hex(b.csrs.raw(a)) for k,a in zip(('cause','tval','epc'),(0x342,0x343,0x341) if mode==3 else (0x142,0x143,0x141))}
            break
        if b.poweroff:reason='poweroff';break
    return dict(retired=b.steps,delta=b.steps-retired,pc=hex(b.pc),mode=b.mode,stop=reason,fault=fault,seconds=round(time.perf_counter()-start,3),alu_calls=b.alu_calls,serial=b.serial.decode(errors='replace'),linux_booted=False)

def run(a):
    data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest()
    b=restore(data,a,a.input)
    cache=PageCache(b);old=b.c.run;oldm=b.math.run
    checks=0
    def ar(op,x,y,ablate=False):
        nonlocal checks
        v=old(op,x,y,ablate)
        if not ablate:
            assert v[0]==oracle(op,x,y)
            checks+=1
        return v
    def mr(op,x,y,ablate=False):
        nonlocal checks
        v=oldm(op,x,y,ablate)
        if not ablate:
            assert v==moracle(op,x,y)
            checks+=1
        return v
    b.c.run=ar;b.math.run=mr
    start=b.steps;bound=b.boundaries;last=time.monotonic();elapsed=0
    before=(b.memory.reads,b.memory.writes,b.traps,b.interrupts,b.returns)
    initial_uart=len(b.serial);chunks=0
    assert b.ram is None
    while b.boundaries-bound<a.budget:
        r=advance_chunk(b,min(a.chunk,a.budget-(b.boundaries-bound)))
        chunks+=1;elapsed+=r['seconds']
        if r['stop']=='host_error':
            raise RuntimeError(r['fault'])
        done=r['stop']!='budget' or b.boundaries-bound>=a.budget
        if done or time.monotonic()-last>=a.interval:
            r.update(x13=hex(b.reg[13]),x14=hex(b.reg[14]))
            save(b,a.checkpoint,h)
            report=dict(r,delta=b.steps-start,boundary_delta=b.boundaries-bound,
                seconds=elapsed,oracle_checks=checks,chunks=chunks,
                ram_read_delta=b.memory.reads-before[0],ram_write_delta=b.memory.writes-before[1],
                new_traps=b.traps-before[2],new_interrupts=b.interrupts-before[3],
                new_returns=b.returns-before[4],new_uart=bytes(b.serial[initial_uart:]).decode(errors='replace'),
                cache=cache.report(),no_guest_bytearray=b.ram is None,
                checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest())
            tmp=a.output.with_suffix('.json.tmp')
            tmp.write_text(json.dumps(report,indent=2)+'\n')
            os.replace(tmp,a.output)
            last=time.monotonic()
        if done:return report
    raise RuntimeError('no progress')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('build-dir','image','input','checkpoint','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--cc',default='cc');p.add_argument('--ld')
    p.add_argument('--budget',type=int,default=60000);p.add_argument('--chunk',type=int,default=10000)
    p.add_argument('--interval',type=int,default=300)
    a=p.parse_args()
    if not 1<=a.chunk<=50000 or not 1<=a.budget<=1000000 or not 1<=a.interval<=300:p.error('bounded parameters')
    if a.input.resolve()==a.checkpoint.resolve() or a.checkpoint.exists() or a.output.exists():p.error('new checkpoint and output required')
    print(json.dumps(run(a),indent=2))

