"""Read-only, image-bound Linux memory-init progress diagnostics.

Never changes guest execution, counters, registers, or checkpoint bytes.
The PFN interpretation applies only to the audited image and PC ranges.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from neural_register_r03 import validate
IMAGE='9feee94f9fbb3f9a761485fb982c78977594d9c0db8b0b411b5677dfb5efc325'
RANGES=((0xc0275600,0xc027575c),(0xc02530d0,0xc02531e8))
def read(path):
    with np.load(path,allow_pickle=False) as z:
        m=json.loads(z['meta'].tobytes())
        if m['image_hash']!=IMAGE or m['schema']!=2:raise ValueError('unsupported image or checkpoint')
        reg=z['regs'];pc=z['pc']
        if reg.shape!=(32,32,2,5) or pc.shape!=(32,2,5):raise ValueError('rail shape')
        if not np.all((reg==0)|(reg==1)) or not np.all((pc==0)|(pc==1)):raise ValueError('rail value')
        r=validate(reg);p=validate(pc)
        word=lambda a:sum(int(v==5)<<i for i,v in enumerate(a[:,1]))
        addr=word(p);cur=word(r[9]);end=word(r[20]);c=m['counters']
        # The memset helper alone does not establish a calling context.
        # Only interpret it when the live return address belongs to the
        # observed page descriptor initialization call site.
        direct=RANGES[0][0]<=addr<RANGES[0][1]
        helper=RANGES[1][0]<=addr<RANGES[1][1] and word(r[1])==0xc027570c
        known=(direct or helper) and 0x80400<=cur<=end==0x84000
        out=dict(retired=c['steps'],pc=hex(addr),traps=c['traps'],phase='page-descriptor-init' if known else 'unclassified',
            pfn_current=cur if known else None,pfn_limit=end if known else None,pfn_remaining=end-cur if known else None,
            image_sha256=m['image_hash'])
    out['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    return out
def report(paths):
    rows=[read(p) for p in paths]
    if any(b['retired']<=a['retired'] for a,b in zip(rows,rows[1:])):raise ValueError('checkpoint ordering')
    good=[r for r in rows if r['phase']=='page-descriptor-init']
    # Only compare endpoints when all samples are in the known phase.
    progress=None
    if len(good)==len(rows) and len(rows)>1 and all(b['pfn_current']>=a['pfn_current'] for a,b in zip(rows,rows[1:])):
        progress=rows[-1]['pfn_current']-rows[0]['pfn_current']
    return dict(status='PASS',profile='specific audited RV32 Linux image, PC and return-address guarded',
        rows=rows,progress_pages=progress)
def test():
    import tempfile
    from neural_register_r03 import encode
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'state.npz';reg=np.stack([encode(0,32)]*32);reg[9]=encode(0x82000,32);reg[20]=encode(0x84000,32)
        m=dict(schema=2,image_hash=IMAGE,counters=dict(steps=123,traps=1))
        def put(pc,ra=0,image=IMAGE):
            reg[1]=encode(ra,32);m['image_hash']=image
            np.savez(p,meta=np.frombuffer(json.dumps(m).encode(),dtype=np.uint8),regs=reg,pc=encode(pc,32))
        put(0xc0275738);assert read(p)['pfn_remaining']==8192
        put(0xc0253108,0xc027570c);assert read(p)['phase']=='page-descriptor-init'
        put(0xc0253108);assert read(p)['phase']=='unclassified'
        put(0x80000000);assert read(p)['pfn_remaining'] is None
        put(0xc0275738,image='invalid')
        try:read(p)
        except ValueError:pass
        else:raise AssertionError('wrong image accepted')
    return dict(status='PASS',phase_checks=5)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoints',type=Path,nargs='*');p.add_argument('--output',type=Path);a=p.parse_args()
    r=report(a.checkpoints) if a.checkpoints else test()
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

