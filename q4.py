"""Standalone GENESIS R02 Q4 packing primitives."""
import hashlib
import numpy as np
OUTDEGREE=24
def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()
def pack_q4(w):
    u=(np.asarray(w,dtype=np.int16)&15).astype(np.uint8)
    assert len(u)%2==0
    return u[::2] | (u[1::2]<<4)
def unpack_q4(p):
    p=np.asarray(p,dtype=np.uint8)
    a=np.empty(len(p)*2,dtype=np.uint8)
    a[::2]=p&15;a[1::2]=p>>4
    x=a.astype(np.int16); x[x>=8]-=16
    return x.astype(np.int8)
