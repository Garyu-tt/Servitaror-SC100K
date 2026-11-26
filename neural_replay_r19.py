"""R19 corrected Linux replay, new checkpoints only; audit ALU operations."""
import argparse,hashlib,json,time
from pathlib import Path
from neural_linux_r12 import load_image
from neural_resume_r13 import restore,save,advance
from neural_tlb_r17 import PageCache
from neural_recurrent_r15 import setup
from neural_alu_r01 import oracle
from neural_math_r05 import oracle as moracle
def main():
 p=argparse.ArgumentParser();p.add_argument('--image',type=Path,required=True);p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--input',type=Path);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--budget',type=int,default=20000);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if not 1<=a.budget<=100000:p.error('bounded budget required')
 data=a.image.read_bytes();h=hashlib.sha256(data).hexdigest();b,_=load_image(data,64);e=setup(b,a)
 if a.input:
  # Reject pre-fix state; companion report binds the corrected lineage.
  parent=a.input.with_suffix('.json')
  report=json.loads(parent.read_text())
  if report.get('revision')!='R19-corrected' or report.get('checkpoint_sha256')!=hashlib.sha256(a.input.read_bytes()).hexdigest():raise ValueError('unverified corrected checkpoint')
  restore(b,a.input,h)
 e.attach(b);c=PageCache(b);old=b.c.run;oldm=b.math.run;od=b.decoder.decode
 counts={};first=[]
 def ar(op,x,y,ablate=False):
  out=old(op,x,y,ablate)
  if not ablate and out[0]!=oracle(op,x,y):raise AssertionError((b.steps,op,x,y,out[0],oracle(op,x,y)))
  counts[op]=counts.get(op,0)+1;return out
 def mr(op,x,y,ablate=False):
  out=oldm(op,x,y,ablate)
  if not ablate and out!=moracle(op,x,y):raise AssertionError((b.steps,op,x,y,out,moracle(op,x,y)))
  counts[op]=counts.get(op,0)+1;return out
 def dec(w,ablate=False):
  n=od(w,ablate)
  if n=='SLTIU' and len(first)<16:
   x=b.reg[(w>>15)&31];imm=(w>>20);imm=(imm-4096 if imm&2048 else imm)&0xffffffff
   first.append(dict(step=b.steps,pc=hex(b.pc),operand=hex(x),imm=hex(imm),expected=int(x<imm)))
   # This read is diagnostic, not part of guest state accounting.
   b.reg.reads-=1
  return n
 b.c.run=ar;b.math.run=mr;b.decoder.decode=dec
 r=advance(b,a.budget);save(b,a.checkpoint,h)
 r.update(revision='R19-corrected',image_sha256=h,checkpoint_sha256=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),oracle_checks=sum(counts.values()),oracle_counts=counts,sltiu_examples=first,cache=c.report())
 a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()

