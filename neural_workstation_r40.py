"""SC500K local workstation. No network service beyond loopback; no GitHub runtime.
The worker owns guest state and runs independently of the browser. Visualization
samples actual neural evaluations; structural layout is not anatomical XYZ.
"""
import argparse,base64,hashlib,json,queue,threading,time,webbrowser
from pathlib import Path
from http.server import ThreadingHTTPServer,BaseHTTPRequestHandler
import numpy as np
from neural_uart_r36 import fresh,restore,save,PageCache,oracle,moracle
from neural_trace_r34 import Recorder
from neural_demo_r35 import Spatial,AXES
from neural_linux_r12 import fixture
from region_diagnostics import capture

class Workstation:
    def __init__(self,b,a,h):
        self.b=b;self.a=a;self.hash=h;self.cache=PageCache(b)
        self.rec=Recorder(b,a);self.latest={};self.sample_boundary=None
        self.spatial=Spatial(self.rec,a.coordinates) if a.coordinates else None
        self.graphs=self.make_graphs()
        self.running=False;self.error=None;self.trap=None;self.checks=0
        self.start=time.monotonic();self.last_save=self.start;self.saved=None
        self.commands=queue.Queue(64);self.shutdown=threading.Event()
        self.snapshot={};self.last_steps=b.steps;self.last_refresh=self.start;self.ips=0
        self.dirty=False
        self.refresh_interval=1.0
        old=b.c.run;oldm=b.math.run
        def ar(op,x,y,ablate=False):
            v=old(op,x,y,ablate)
            if not ablate:
                assert v[0]==oracle(op,x,y);self.checks+=1
            return v
        def mr(op,x,y,ablate=False):
            v=oldm(op,x,y,ablate)
            if not ablate:
                assert v==moracle(op,x,y);self.checks+=1
            return v
        b.c.run=ar;b.math.run=mr
        self.publish()
        self.thread=threading.Thread(target=self.loop,name='neural-runtime')
    def make_graphs(self):
        result={}
        for n,g in zip(self.rec.names,self.rec.graphs):
            # Deterministically sample actual weighted edges, preserving their endpoints.
            stride=max(1,(len(g.edges)+1599)//1600)
            edges=g.edges[::stride];ids=np.unique(edges[:,:2]);idx={int(x):i for i,x in enumerate(ids)}
            levels=g.level[ids];pts=np.zeros((len(ids),2),dtype=float)
            pts[:,0]=levels
            for level in np.unique(levels):
                k=np.flatnonzero(levels==level)
                pts[k,1]=np.linspace(0,1,len(k)) if len(k)>1 else .5
            views={'circuit':pts.tolist()}
            if self.spatial:
                views.update({v:self.spatial.project(n,v)[ids].tolist() for v in AXES})
            result[n]=dict(ids=ids.tolist(),edges=[[idx[int(x)],idx[int(y)],int(w)] for x,y,w in edges],
                views=views,neurons=g.n,synapses=len(g.edges))
        return result
    def publish(self):
        b=self.b;now=time.monotonic();dt=now-self.last_refresh
        if dt>=.2:
            self.ips=round((b.steps-self.last_steps)/dt);self.last_steps=b.steps;self.last_refresh=now
        self.snapshot=dict(running=self.running,retired=b.steps,boundary=b.boundaries,
            pc=hex(b.pc),mode=b.mode,ips=self.ips,elapsed=round(now-self.start,1),
            uart=bytes(b.serial[-65536:]).decode(errors='replace'),
            rgb=base64.b64encode(b.rgb()).decode(),latest=dict(self.latest),
            sample_boundary=self.sample_boundary,checks=self.checks,alu=b.alu_calls,
            irq=b.interrupts,returns=b.returns,traps=b.traps,walks=b.walks,
            rx_free=b.rx.size-b.rx.count,error=self.error,trap=self.trap,
            poweroff=b.poweroff,checkpoint=self.saved,coordinates=bool(self.spatial),
            scope='Sampled ALU / decoder / control activity; RAM and recurrent states inspected separately',
            regions=capture(b,self.cache),
            fixture=self.a.fixture)
    def persist(self):
        if self.error:raise RuntimeError('unsafe state after host error')
        save(self.b,self.a.checkpoint,self.hash)
        self.saved=dict(path=str(self.a.checkpoint),retired=self.b.steps,
            sha256=hashlib.sha256(self.a.checkpoint.read_bytes()).hexdigest())
        self.last_save=time.monotonic();self.dirty=False
    def step(self,sample=False):
        b=self.b;before=b.traps
        if sample:
            self.rec.events.clear();self.rec.step()
            self.latest={e['region']:e for e in self.rec.events}
            self.sample_boundary=b.boundaries;self.rec.events.clear()
        else:b.step()
        self.dirty=True
        if b.traps!=before:
            self.running=False
            self.trap=dict(cause=hex(b.csrs.raw(0x342 if b.mode==3 else 0x142)),
                epc=hex(b.csrs.raw(0x341 if b.mode==3 else 0x141)),
                tval=hex(b.csrs.raw(0x343 if b.mode==3 else 0x143)))
        if b.poweroff:self.running=False
    def execute(self,op,data):
        if op=='run':
            if self.error or self.b.poweroff:raise ValueError('runtime stopped')
            self.trap=None;self.running=True
        elif op=='pause':self.running=False
        elif op=='step':
            self.running=False
            if self.error or self.b.poweroff:raise ValueError('runtime stopped')
            self.trap=None;self.step(True)
        elif op=='save':self.persist()
        elif op=='refresh_rate':
            v=data.get('seconds')
            if v not in (.5,1.0,2.0,5.0):raise ValueError('refresh interval')
            self.refresh_interval=v
        elif op=='serial':
            s=data.get('text')
            if not isinstance(s,str) or not 1<=len(s.encode())<=32:raise ValueError('UART limit: 1..32 bytes')
            if not self.b.uart_input(s.encode()):raise ValueError('UART RX full')
            self.dirty=True
        elif op=='key':
            k=data.get('code');p=data.get('pressed')
            if type(k)!=int or not 1<=k<=255 or type(p)!=bool:raise ValueError('key')
            if not self.b.keyboard.push(k,p):raise ValueError('keyboard FIFO full')
            self.dirty=True
        else:raise ValueError('unknown action')
    def submit(self,op,data=None):
        q=queue.Queue(1)
        try:self.commands.put_nowait((op,data or {},q))
        except queue.Full:raise ValueError('command queue full')
        try:ok,msg=q.get(timeout=3)
        except queue.Empty:return dict(accepted=True,pending=True)
        if not ok:raise ValueError(msg)
        return dict(accepted=True,pending=False)
    def loop(self):
        try:
            while not self.shutdown.is_set():
                try:
                    op,data,q=self.commands.get(timeout=0 if self.running else .1)
                    try:self.execute(op,data);self.publish();q.put((True,None))
                    except ValueError as e:q.put((False,str(e)))
                    except Exception as e:
                        self.error=repr(e);self.running=False;q.put((False,self.error))
                except queue.Empty:pass
                if self.running:
                    try:
                        for _ in range(128):
                            self.step()
                            if not self.running or self.shutdown.is_set() or not self.commands.empty():break
                        if time.monotonic()-self.last_refresh>=self.refresh_interval or not self.running:
                            if self.running and self.commands.empty() and not self.shutdown.is_set():self.step(True)
                            self.publish()
                        if not self.running or time.monotonic()-self.last_save>=self.a.interval:
                            if self.dirty:self.persist();self.publish()
                    except Exception as e:
                        self.error=repr(e);self.running=False;self.publish()
        finally:
            self.running=False
            if self.dirty and not self.error:
                try:self.persist()
                except Exception as e:self.error=repr(e)
            self.publish()
    def close(self):
        self.shutdown.set();self.thread.join()

HTML=Path(__file__).with_name('workstation.html')
def server(w,port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args):pass
        def reply(self,status,data,mime='application/json'):
            raw=data if isinstance(data,bytes) else json.dumps(data).encode()
            self.send_response(status);self.send_header('Content-Type',mime)
            self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff');self.end_headers();self.wfile.write(raw)
        def allowed(self):
            host=f'127.0.0.1:{self.server.server_port}'
            return self.headers.get('Host')==host and self.headers.get('Origin') in (None,f'http://{host}')
        def do_GET(self):
            if not self.allowed():return self.reply(403,{'error':'origin'})
            if self.path=='/':return self.reply(200,HTML.read_bytes(),'text/html; charset=utf-8')
            if self.path=='/state':return self.reply(200,w.snapshot)
            if self.path=='/graph':return self.reply(200,w.graphs)
            return self.reply(404,{'error':'not found'})
        def do_POST(self):
            if not self.allowed():return self.reply(403,{'error':'origin'})
            if self.path!='/action':return self.reply(404,{'error':'not found'})
            try:
                if self.headers.get('Content-Type')!='application/json':raise ValueError('JSON required')
                size=int(self.headers.get('Content-Length','0'))
                if not 0<size<=1024:raise ValueError('body limit')
                data=json.loads(self.rfile.read(size))
                self.reply(200,w.submit(data['op'],data))
            except (ValueError,TypeError,KeyError) as e:self.reply(400,{'error':str(e)})
    return ThreadingHTTPServer(('127.0.0.1',port),Handler)

def main():
    p=argparse.ArgumentParser()
    for k in ('image','input','checkpoint','build-dir','coordinates'):p.add_argument('--'+k,type=Path)
    p.add_argument('--fixture',action='store_true');p.add_argument('--self-test',action='store_true')
    p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--port',type=int,default=8765)
    p.add_argument('--interval',type=int,default=300);p.add_argument('--run',action='store_true');p.add_argument('--no-browser',action='store_true')
    a=p.parse_args();a.fixture=a.fixture or a.self_test
    if bool(a.image)==a.fixture:p.error('specify --image or --fixture/--self-test')
    if not a.checkpoint or not a.build_dir:p.error('checkpoint and build-dir required')
    if a.checkpoint.exists() or a.input and a.input.resolve()==a.checkpoint.resolve():p.error('new output checkpoint required')
    if not 1<=a.interval<=3600:p.error('interval out of range')
    a.checkpoint.parent.mkdir(parents=True,exist_ok=True)
    data=fixture() if a.fixture else a.image.read_bytes();h=hashlib.sha256(data).hexdigest()
    b=restore(data,a,a.input) if a.input else fresh(data,a)
    w=Workstation(b,a,h);s=server(w,0 if a.self_test else a.port)
    w.thread.start()
    try:
        if a.self_test:
            import urllib.request
            t=threading.Thread(target=s.serve_forever,daemon=True);t.start()
            url=f'http://127.0.0.1:{s.server_port}'
            def get(path):return json.load(urllib.request.urlopen(url+path,timeout=10))
            def act(op,**kw):
                return json.load(urllib.request.urlopen(urllib.request.Request(url+'/action',data=json.dumps(dict(op=op,**kw)).encode(),headers={'Content-Type':'application/json'}),timeout=10))
            assert get('/state')['retired']==0
            assert len(get('/graph'))==3
            act('step')
            assert get('/state')['retired']==1
            act('run')
            deadline=time.monotonic()+15
            while get('/state')['retired']<20 and time.monotonic()<deadline:time.sleep(.1)
            act('pause');time.sleep(.2);n=get('/state')['retired'];time.sleep(.3)
            assert n>=20 and get('/state')['retired']==n and get('/state')['uart']=='X'
            assert get('/state')['latest'] and len(base64.b64decode(get('/state')['rgb']))==192000
            act('serial',text='test\n');assert get('/state')['rx_free']==27
            s.shutdown();t.join()
            print(json.dumps(dict(status='PASS',retired=n,autonomous=True,pause=True,serial=True,real_frame=True,graph=True)),flush=True)
        else:
            url=f'http://127.0.0.1:{s.server_port}/';print(url,flush=True)
            if not a.no_browser:webbrowser.open(url)
            if a.run:w.submit('run')
            s.serve_forever()
    except KeyboardInterrupt:pass
    finally:s.server_close();w.close()
if __name__=='__main__':main()

