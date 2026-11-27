"""Compile on developer machines; frozen builds only load verified bundled binaries."""
import hashlib,json,os,sys,subprocess
from pathlib import Path

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def source_digest(p):return hashlib.sha256(p.read_text(encoding='utf-8').encode()).hexdigest()

def native_library(c,so,cc,ld,opt=2):
    c=Path(c);so=Path(so)
    if os.name=='nt':so=so.with_suffix('.dll')
    if getattr(sys,'frozen',False):
        root=Path(sys._MEIPASS)/'native'
        m=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        entry=m.get(so.name);lib=root/so.name;src=root/c.name
        if not entry or source_digest(c)!=entry['source_sha256'] or source_digest(src)!=entry['source_sha256'] or digest(lib)!=entry['binary_sha256']:
            raise RuntimeError('Bundled native graph mismatch: '+so.name)
        return lib
    # Windows cannot overwrite a DLL already loaded into the process.
    # A build directory is isolated per developer session.
    meta=so.with_suffix(so.suffix+'.json')
    key=dict(source_sha256=source_digest(c),compiler=str(cc),linker=str(ld),opt=opt)
    if so.exists() and meta.exists():
        try:
            m=json.loads(meta.read_text())
            if all(m.get(k)==v for k,v in key.items()) and m.get('binary_sha256')==digest(so):
                return so
        except (OSError,ValueError):pass
    if os.name=='nt':
        subprocess.run([str(cc),f'-O{opt}','-shared','-Wl,--export-all-symbols','-static-libgcc','-o',str(so),str(c)],check=True)
    elif ld:
        obj=so.with_suffix('.o')
        subprocess.run([str(cc),'-c','-o',str(obj),str(c)],check=True)
        subprocess.run([str(ld),'-shared','-z','noexecstack','-o',str(so),str(obj)],check=True)
    else:
        subprocess.run([str(cc),f'-O{opt}','-fPIC','-shared','-Wl,-z,noexecstack','-o',str(so),str(c)],check=True)
    key['binary_sha256']=digest(so);meta.write_text(json.dumps(key))
    return so

