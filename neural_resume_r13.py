"""R13: exact synchronized-rail quotient and resumable Linux entry.

A static compiler verifies that every five-cell rail has identical incoming
weights, then evaluates the same integer synaptic current and threshold
for one representative. No guest arithmetic shortcut, no result cache.
Only valid for the fixed, synchronous, uniform-replica graphs.
"""
import argparse,hashlib,json,random,time,os
from pathlib import Path
from collections import defaultdict
import numpy as np
from neural_alu_r01 import Circuit,OPS
from neural_linux_r12 import load_image,fixture
from neural_board_r11 import RAM
from neural_register_r03 import validate
from neural_decoder_r02 import Decoder

class Quotient:
    def __init__(self,n,edges,level,ports,outputs):
        assert n%5==0
        self.n=n;self.ports=ports;self.outputs=outputs
        incoming=[defaultdict(int) for _ in range(n)]
        for s,t,w in edges:incoming[t][s]+=w
        groups=[]
        for r in range(n//5):
            ref=dict(incoming[5*r])
            assert all(dict(incoming[5*r+j])==ref for j in range(5))
            grouped=defaultdict(int)
            for s,w in ref.items():grouped[s//5]+=w
            for source in grouped:
                weights=[ref.get(source*5+j,0) for j in range(5)]
                assert len(set(weights))==1
                assert level[5*source]<level[5*r]
            groups.append(dict(grouped))
        flat=[r//5 for p in ports for rail in p for r in rail]
        input_ids=set(flat)
        assert all((not q)==(r in input_ids) for r,q in enumerate(groups))
        assert all(tuple(rail)==tuple(range(rail[0],rail[0]+5)) and rail[0]%5==0 for p in list(ports)+list(outputs) for rail in p)
        lines=['def evaluate(bits,enabled):']
        for i,p in enumerate(ports):
            lines.extend([f' r{p[0][0]//5}=1-bits[{i}]',f' r{p[1][0]//5}=bits[{i}]'])
        for r in sorted(range(n//5),key=lambda r:level[5*r]):
            if not groups[r]:continue
            terms='+'.join(f'({w}*r{s})' for s,w in sorted(groups[r].items()))
            # clip(current//3,-32,31)>=11 iff current>=33.
            lines.append(f' r{r}=int(enabled and ({terms})>=33)')
        lines.append(' return ('+','.join(f'(r{p[0][0]//5},r{p[1][0]//5})' for p in outputs)+',),5*('+ '+'.join(f'r{r}' for r in range(n//5))+')')
        ns={};exec(compile('\n'.join(lines),'<verified-synaptic-quotient>','exec'),ns)
        self.evaluate=ns['evaluate']
        self.groups=len(groups)
    def run(self,bits,ablate=False):
        result,count=self.evaluate(bits,not ablate)
        if any((l,h) not in ((1,0),(0,1)) for l,h in result):raise RuntimeError('invalid dual-rail readout')
        return [h for l,h in result],count

def accelerate(b):
    c=b.c;d=b.decoder;ctrl=b.control
    names=list(c.inputs)
    alu=Quotient(c.n,c.edges,c.level,list(c.inputs.values()),c.outputs)
    decoder=Quotient(d.n,d.edges,d.level,d.bits,d.outputs)
    ports=[p for seq in ctrl.inputs.values() for p in seq]
    control=Quotient(ctrl.c.n,ctrl.c.edges,ctrl.c.level,ports,[ctrl.take]+ctrl.outputs)
    def ar(op,a,v,ablate=False):
        bits=[0 if n=='zero' else ((a>>int(n[1:]))&1) if n.startswith('a') else ((v>>int(n[1:]))&1) if n.startswith('b') else int(n==op) for n in names]
        out,count=alu.run(bits,ablate)
        return sum(x<<i for i,x in enumerate(out)),count
    def dr(word,ablate=False):
        out,_=decoder.run([(word>>i)&1 for i in range(32)],ablate)
        matches=[p[0] for p,v in zip(d.patterns,out) if v]
        if len(matches)>1:raise RuntimeError('ambiguous neural decode')
        return matches[0] if matches else None
    def cr(name,seq,target,xor=0,signed=0,unsigned=0,ablate=False):
        vals={'seq':seq,'target':target,'xor':xor,'signed':signed,'unsigned':unsigned}
        bits=[(vals.get(k,int(k==name))>>i)&1 for k,ps in ctrl.inputs.items() for i in range(len(ps))]
        out,_=control.run(bits,ablate);ctrl.calls+=1
        return sum(v<<i for i,v in enumerate(out[1:])),out[0]
    old=(c.run,d.decode,ctrl.route)
    c.run=ar;d.decode=dr;ctrl.route=cr
    return old,(alu.groups,decoder.groups,control.groups)

COUNTERS=('steps','alu_calls','decode_calls','traps','returns','waits','interrupts','boundaries','walks','pte_reads','ad_updates','fences','sbi_calls','atomic_calls','atomic_selects','csr_calls')
def snapshot(b,image_hash):
    meta=dict(schema=1,image_hash=image_hash,ram_bytes=len(b.ram),counters={k:getattr(b,k) for k in COUNTERS},
        timer=vars(b.timer).copy(),reservation=b.reservation,deadline=b.deadline,poweroff=b.poweroff,halted=b.halted,
        csr_addresses=sorted(b.csrs.state),fixed_addresses=sorted(b.csrs.fixed),
        reg_counts=[b.reg.reads,b.reg.writes],csr_counts=[b.csrs.reads,b.csrs.writes],pc_commits=b.counter.commits,
        control_calls=b.control.calls,math_counts=[b.math.calls,b.math.rounds,b.math.gates])
    return dict(meta=np.frombuffer(json.dumps(meta,sort_keys=True).encode(),dtype=np.uint8),
        ram=np.frombuffer(b.ram,dtype=np.uint8),regs=b.reg._state,pc=b.counter.state,mode=b.priv.state,
        csr=np.stack([b.csrs.state[a] for a in meta['csr_addresses']]),
        fixed=np.stack([b.csrs.fixed[a] for a in meta['fixed_addresses']]),
        uart=np.frombuffer(bytes(b.uart_regs),dtype=np.uint8),serial=np.frombuffer(bytes(b.serial),dtype=np.uint8))
def save(b,path,h):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**snapshot(b,h));f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)
def restore(b,path,h):
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['meta'].tobytes())
        if m['schema']!=1 or m['image_hash']!=h or m['ram_bytes']!=len(b.ram):raise ValueError('checkpoint identity mismatch')
        if m['csr_addresses']!=sorted(b.csrs.state) or m['fixed_addresses']!=sorted(b.csrs.fixed):raise ValueError('CSR layout mismatch')
        shapes={'ram':(len(b.ram),),'regs':(32,32,2,5),'pc':(32,2,5),'mode':(2,2,5),'csr':(len(b.csrs.state),32,2,5),'fixed':(len(b.csrs.fixed),32,2,5),'uart':(8,)}
        for k,shape in shapes.items():
            if z[k].shape!=shape:raise ValueError('bad checkpoint shape '+k)
        for k in ('regs','pc','mode','csr','fixed'):validate(z[k])
        if np.any(z['regs'][0,:,1]):raise ValueError('corrupt x0')
        b.ram[:]=z['ram'].tobytes();b.reg._state=z['regs'].copy();b.counter.state=z['pc'].copy();b.priv.state=z['mode'].copy()
        b.csrs.state={a:r.copy() for a,r in zip(m['csr_addresses'],z['csr'])};b.csrs.fixed={a:r.copy() for a,r in zip(m['fixed_addresses'],z['fixed'])}
        b.uart_regs[:]=z['uart'].tobytes();b.serial[:]=z['serial'].tobytes()
        for k,v in m['counters'].items():setattr(b,k,v)
        for k,v in m['timer'].items():setattr(b.timer,k,v)
        for k in ('reservation','deadline','poweroff','halted'):setattr(b,k,m[k])
        b.reg.reads,b.reg.writes=m['reg_counts'];b.csrs.reads,b.csrs.writes=m['csr_counts']
        b.counter.commits=m['pc_commits'];b.control.calls=m['control_calls']
        b.math.calls,b.math.rounds,b.math.gates=m['math_counts']
def advance(b,budget):
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
    return dict(retired=b.steps,delta=b.steps-retired,pc=hex(b.pc),mode=b.mode,stop=reason,fault=fault,seconds=round(time.perf_counter()-start,3),alu_calls=b.alu_calls,serial=b.serial.decode(errors='replace'),x13=hex(b.reg[13]),x14=hex(b.reg[14]),linux_booted=False)
def selftest(path):
    b,_=load_image(fixture(),16);old,groups=accelerate(b);rng=random.Random(31013);checks=0
    for op in OPS:
        for i in range(24):
            a,v=rng.getrandbits(32),rng.getrandbits(32)
            assert b.c.run(op,a,v)==old[0](op,a,v);checks+=1
    for _ in range(128):
        w=rng.getrandbits(32);assert b.decoder.decode(w)==old[1](w);checks+=1
    for name in ('BEQ','BNE','BLT','BGE','BLTU','BGEU','JAL','JALR','ADD'):
        for i in range(8):
            args=(name,rng.getrandbits(32),rng.getrandbits(32),i%2,rng.randrange(2),rng.randrange(2))
            assert b.control.route(*args)==old[2](*args);checks+=1
    for f in (lambda:b.c.run('ADD',3,4,True),lambda:b.decoder.decode(0x13,True),lambda:b.control.route('JAL',4,8,ablate=True)):
        try:f()
        except RuntimeError:pass
        else:raise AssertionError('ablation')
    h=hashlib.sha256(fixture()).hexdigest();advance(b,3);save(b,path,h);advance(b,8)
    expected={k:v.copy() for k,v in snapshot(b,h).items()}
    restore(b,path,h);advance(b,8)
    assert all(np.array_equal(v,snapshot(b,h)[k]) for k,v in expected.items())
    print(json.dumps(dict(status='PASS',equivalence_checks=checks,quotient_groups=groups,checkpoint_exact=True,ablation=True)))
def main():
    p=argparse.ArgumentParser();p.add_argument('--image',type=Path);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--resume',action='store_true');p.add_argument('--budget',type=int,default=512);p.add_argument('--output',type=Path);a=p.parse_args()
    if not 1<=a.budget<=100000:p.error('bounded budget')
    if not a.image:selftest(a.checkpoint);return
    data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b,_=load_image(data,64);_,groups=accelerate(b)
    if a.resume:restore(b,a.checkpoint,h)
    r=advance(b,a.budget);save(b,a.checkpoint,h)
    r.update(image_sha256=h,quotient_groups=groups,checkpoint_bytes=a.checkpoint.stat().st_size)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))
if __name__=='__main__':main()

