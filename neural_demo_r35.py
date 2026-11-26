"""R09: bounded live demo session, real trace and strict spatial provenance.

The GUI runs only on localhost and advances the guest in explicit batches.
Missing GENESIS XYZ is a visible blocker, not a reason to invent geometry.
"""
import argparse,base64,hashlib,json,random,time
from pathlib import Path
import numpy as np
from neural_trace_r34 import Recorder
from neural_uart_r36 import restore,save,fresh
from neural_paged_r29 import board,SynapticBoard,PagedCrossbar
from neural_devices_r30 import attach
from neural_framebuffer_r32 import upgrade as visual_upgrade
from neural_linux_r12 import fixture
from neural_tlb_r17 import PageCache

AXES={'top':(0,1),'side':(1,2),'front':(0,2)}
class Spatial:
    """Coordinates must be supplied by a graph-bound GENESIS export."""
    def __init__(self,rec,path):
        self.rec=rec;self.xyz={}
        with np.load(path,allow_pickle=False) as z:
            for name,g in zip(rec.names,rec.graphs):
                xyz=z[name+'_xyz'];ids=z[name+'_ids'];edges=z[name+'_edges']
                if xyz.shape!=(g.n,3) or not np.isfinite(xyz).all() or xyz.dtype.kind!='f':raise ValueError('coordinate shape')
                if not np.array_equal(ids,np.arange(g.n)) or not np.array_equal(edges,g.edges):raise ValueError('graph identity')
                self.xyz[name]=xyz.copy()
    def project(self,name,view):
        return self.xyz[name][:,AXES[view]]

class Session:
    def __init__(self,b,a):
        self.b=b;PageCache(b);self.r=Recorder(b,a);self.running=False
        self.seq=0;self.total=0;self.latest={};self.last_counts={}
        self.started=time.monotonic();self.error=None;self.spatial=Spatial(self.r,a.coordinates) if a.coordinates else None
        self.initial_retired=b.steps;self.trace_path=a.trace
        self.trace_every=getattr(a,'trace_every',1);self.trace_limit=getattr(a,'trace_limit_mib',256)<<20
        self.log=a.trace.open('x',encoding='utf-8')
        graph=self.r.graph(a.graph)
        self.emit(dict(kind='initial',boundary=b.boundaries,retired=b.steps,pc=hex(b.pc),
            image_sha256=a.image_hash,mode=a.mode,graph=graph,spatial_loaded=bool(self.spatial),trace_every=self.trace_every,
            uart_hex=bytes(b.serial).hex(),frame_sha256=hashlib.sha256(b.scanout()).hexdigest()))
    def emit(self,r):
        self.log.write(json.dumps(r,separators=(',',':'))+'\n');self.log.flush()
    def batch(self,n):
        if not 1<=n<=256:raise ValueError('batch size')
        b=self.b;start=b.steps;uart=len(b.serial);count=0;traps=b.traps
        self.latest={};self.last_counts={}
        for _ in range(n):
            if self.log.tell()>=self.trace_limit:
                self.error='trace size limit reached';self.running=False;break
            self.r.events.clear()
            try:self.r.step()
            except RuntimeError as e:self.error=str(e);self.running=False
            events=self.r.events
            for event in events:
                name=event['region'];self.latest[name]=event
                self.last_counts[name]=self.last_counts.get(name,0)+event['spikes']
            self.total+=sum(e['spikes'] for e in events);count+=len(events)
            self.seq+=1
            self.emit(dict(kind='boundary',sequence=self.seq,boundary=b.boundaries,retired=b.steps,
                events=events if self.seq%self.trace_every==0 or self.seq==1 or self.error else [],
                evaluations=len(events),spikes=sum(e['spikes'] for e in events),sampled=self.seq%self.trace_every==0 or self.seq==1 or bool(self.error),
                pc=hex(b.pc),uart_hex=bytes(b.serial[uart:]).hex(),error=self.error))
            uart=len(b.serial)
            if self.error or b.poweroff or b.traps!=traps:self.running=False;break
        self.r.events.clear()
        self.emit(dict(kind='sample',sequence=self.seq,boundary=b.boundaries,retired=b.steps,
            frame_sha256=hashlib.sha256(b.scanout()).hexdigest(),batch_evaluations=count))
        return b.steps-start
    def snapshot(self):
        b=self.b
        return dict(sequence=self.seq,boundary=b.boundaries,retired=b.steps,pc=hex(b.pc),running=self.running,
            uart=bytes(b.serial[-65536:]).decode(errors='replace'),rx_free=b.rx.size-b.rx.count,rgb=base64.b64encode(b.rgb()).decode(),
            total_spikes=self.total,counts=self.last_counts,latest=self.latest,error=self.error,
            elapsed=round(time.monotonic()-self.started,3),coordinates=bool(self.spatial),trace_every=self.trace_every,
            regions=[dict(name=n,neurons=g.n,edges=len(g.edges)) for n,g in zip(self.r.names,self.r.graphs)])
    def close(self):
        self.log.close()
    def serial_input(self,data):
        accepted=self.b.uart_input(data)
        self.emit(dict(kind='serial_input',boundary=self.b.boundaries,bytes_hex=data.hex(),accepted=accepted))
        return accepted
    def key(self,code,pressed):
        accepted=self.b.keyboard.push(code,pressed)
        self.emit(dict(kind='input',boundary=self.b.boundaries,code=code,pressed=pressed,accepted=accepted))
        return accepted

HTML=r"""<!doctype html><meta charset=utf-8><title>SC500K live</title>
<style>body{background:#121715;color:#d4e5da;font:15px monospace;margin:20px}button,select{font:inherit;margin:4px}main{display:grid;grid-template-columns:1fr 1fr;gap:14px}section{border:1px solid #48574c;padding:12px;min-width:0}canvas{width:100%;image-rendering:pixelated;background:#000}pre{height:300px;overflow:auto;white-space:pre-wrap}#info{white-space:pre-wrap}small{color:#a7b3ac}</style>
<h2>SC500K / live evidence</h2><button onclick="act('run')">Run</button><button onclick="act('pause')">Pause</button><button onclick="act('step')">Step</button><select id=view><option>top</option><option>side</option><option>front</option></select><p id=info></p>
<main><section><h3>Guest serial</h3><pre id=serial></pre><form id=serialform><input id=serialtext placeholder="Guest serial input" maxlength=31 autocomplete=off><button type=submit>Send</button></form><small id=rxinfo></small><h3>Guest framebuffer</h3><canvas id=fb width=320 height=200></canvas><small>Keyboard events use the guest MMIO FIFO, not a host shell. Linux driver pending.</small></section>
<section><h3>Measured neural activity</h3><p id=geometry></p><canvas id=brain width=640 height=400></canvas><pre id=counts></pre></section></main>
<script>
let running=false,busy=false,last=null;
const el=id=>document.getElementById(id);
async function request(path,data){let r=await fetch(path,data===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(!r.ok)throw Error(await r.text());return r.json()}
function draw(s){last=s;running=s.running;el('info').textContent='boundary '+s.boundary+' | retired '+s.retired+' | PC '+s.pc+' | measured activations '+s.total_spikes+' | '+s.elapsed+' s | trace every '+s.trace_every+' boundary'+(s.error?' | '+s.error:'');el('serial').textContent=s.uart;el('serial').scrollTop=el('serial').scrollHeight;el('rxinfo').textContent='UART RX free: '+s.rx_free+' bytes';
let rgb=atob(s.rgb),c=el('fb').getContext('2d'),im=c.createImageData(320,200);for(let i=0;i<64000;i++){im.data[4*i]=rgb.charCodeAt(3*i);im.data[4*i+1]=rgb.charCodeAt(3*i+1);im.data[4*i+2]=rgb.charCodeAt(3*i+2);im.data[4*i+3]=255}c.putImageData(im,0,0);
el('counts').textContent=JSON.stringify(s.counts,null,2);el('geometry').textContent=s.coordinates?'Graph-bound XYZ available; choose projection.':'GENESIS XYZ missing. Activity raster by stable rail ID, NOT brain anatomy.';el('view').disabled=!s.coordinates;if(!s.coordinates){let x=el('brain').getContext('2d');x.clearRect(0,0,640,400);let y=0;for(let g of s.regions){let e=s.latest[g.name];x.fillStyle='#b8c5bd';x.fillText(g.name,4,y+12);y+=18;if(e){for(let i=0;i<g.neurons/5;i++){let b=parseInt(e.rail_bits.slice(2*(i>>3),2*(i>>3)+2),16);if(b&(1<<(i%8))){x.fillStyle='#67df9a';x.fillRect((i%160)*4,y+Math.floor(i/160)*3,3,2)}}}y+=Math.ceil(g.neurons/5/160)*3+8}}}
async function projection(){if(!last||!last.coordinates)return;let v=el('view').value,p=await request('/projection?view='+v);let c=el('brain').getContext('2d');c.clearRect(0,0,640,400);let pts=p.points;if(!pts.length)return;let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];for(let q of pts){for(let j=0;j<2;j++){lo[j]=Math.min(lo[j],q[j]);hi[j]=Math.max(hi[j],q[j])}}let scale=Math.min(620/Math.max(hi[0]-lo[0],1e-9),380/Math.max(hi[1]-lo[1],1e-9)),xy=q=>[10+(q[0]-lo[0])*scale,10+(q[1]-lo[1])*scale];c.strokeStyle='#36453c';c.beginPath();for(let [i,j] of p.edges){let a=xy(pts[i]),b=xy(pts[j]);c.moveTo(...a);c.lineTo(...b)}c.stroke();for(let i=0;i<pts.length;i++){c.fillStyle=p.active[i]?'#67df9a':'#708078';let a=xy(pts[i]);c.fillRect(...a,2,2)}el('geometry').textContent='Sampled structural edges; '+p.points.length+' neurons, '+p.edges.length+' edges. Activity = latest recorded evaluation per region.'}
async function act(op){if(busy)return;busy=true;try{draw(await request('/action',{op}));await projection()}finally{busy=false}}
el('view').onchange=projection;setInterval(()=>{if(running)act('tick')},100);request('/state').then(s=>{draw(s);projection()});
el('serialform').onsubmit=async e=>{e.preventDefault();let data=el('serialtext').value+'\n';let r=await request('/serial',{text:data});if(r.accepted)el('serialtext').value='';else el('rxinfo').textContent='UART RX full; retry';};
addEventListener('keydown',e=>{if(e.repeat||['SELECT','BUTTON','INPUT'].includes(e.target.tagName))return;let code=e.keyCode;if(code>0&&code<256)request('/key',{code,pressed:true})});addEventListener('keyup',e=>{if(['SELECT','BUTTON','INPUT'].includes(e.target.tagName))return;let code=e.keyCode;if(code>0&&code<256)request('/key',{code,pressed:false})});
</script>"""
def projection(s,view,limit=4000):
    if view not in AXES or s.spatial is None:raise ValueError('spatial coordinates unavailable')
    points=[];edges=[];active=[];offset=0
    for name,g in zip(s.r.names,s.r.graphs):
        # Stable deterministic sampling, same identities in all views.
        ids=np.arange(0,g.n,max(1,(g.n+limit-1)//limit))
        lookup={int(x):offset+i for i,x in enumerate(ids)}
        points.extend(s.spatial.project(name,view)[ids].tolist())
        event=s.latest.get(name);bits=np.unpackbits(np.frombuffer(bytes.fromhex(event['rail_bits']),dtype=np.uint8),bitorder='little') if event else np.zeros(g.n//5,dtype=np.uint8)
        active.extend(bits[ids//5].tolist())
        edges.extend([lookup[int(x)],lookup[int(y)]] for x,y,_ in g.edges if int(x) in lookup and int(y) in lookup)
        offset+=len(ids)
    return dict(points=points,edges=edges,active=active,sampled=True,view=view)
def serve(s,port):
    from http.server import HTTPServer,BaseHTTPRequestHandler
    from urllib.parse import urlparse,parse_qs
    class Handler(BaseHTTPRequestHandler):
        def reply(self,status,data,mime='application/json'):
            raw=data.encode() if isinstance(data,str) else json.dumps(data).encode()
            self.send_response(status);self.send_header('Content-Type',mime);self.send_header('Content-Length',str(len(raw)));self.send_header('Cache-Control','no-store');self.end_headers();self.wfile.write(raw)
        def do_GET(self):
            if self.headers.get('Host')!=f'127.0.0.1:{port}':self.reply(403,{'error':'host'});return
            p=urlparse(self.path)
            try:
                if p.path=='/':self.reply(200,HTML,'text/html; charset=utf-8')
                elif p.path=='/state':self.reply(200,s.snapshot())
                elif p.path=='/projection':self.reply(200,projection(s,parse_qs(p.query).get('view',[''])[0]))
                else:self.reply(404,{'error':'not found'})
            except ValueError as e:self.reply(400,{'error':str(e)})
        def do_POST(self):
            # Reject cross-origin browser requests, including DNS rebinding.
            origin=self.headers.get('Origin');host=self.headers.get('Host')
            if host!=f'127.0.0.1:{port}' or (origin and origin!=f'http://{host}'):
                self.reply(403,{'error':'origin'});return
            try:
                n=int(self.headers.get('Content-Length','0'))
                if not 0<n<=1024:raise ValueError('body size')
                x=json.loads(self.rfile.read(n))
                if self.path=='/serial':
                    if not isinstance(x['text'],str):raise ValueError('text')
                    data=x['text'].encode('utf-8')
                    if not 1<=len(data)<=32:raise ValueError('serial byte limit')
                    self.reply(200,{'accepted':s.serial_input(data)});return
                if self.path=='/key':self.reply(200,{'accepted':s.key(x['code'],x['pressed'])});return
                if self.path!='/action':self.reply(404,{'error':'not found'});return
                op=x['op']
                if op=='run':s.running=not bool(s.error or s.b.poweroff)
                elif op=='pause':s.running=False
                elif op=='step':
                    s.running=False
                    if not s.error and not s.b.poweroff:s.batch(1)
                elif op=='tick':
                    if s.running:s.batch(64)
                else:raise ValueError('action')
                self.reply(200,s.snapshot())
            except (ValueError,TypeError,KeyError) as e:self.reply(400,{'error':str(e)})
    with HTTPServer(('127.0.0.1',port),Handler) as server:
        print(f'http://127.0.0.1:{port}',flush=True);server.serve_forever()
def test(a):
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        root=Path(td);a.trace=root/'trace.jsonl';a.graph=root/'graph.npz';a.image_hash=hashlib.sha256(fixture()).hexdigest();a.mode='fixture';a.coordinates=None
        b=fresh(fixture(),a);s=Session(b,a)
        try:
            assert s.snapshot()['retired']==0 and not s.latest
            s.batch(5);assert b.steps==5 and b.serial==b'X'
            assert s.key(30,True);assert b.keyboard.pop()==0x11e
            assert s.serial_input(b'hello\n');assert bytes(b.rx.pop() for _ in range(6))==b'hello\n'
            seq=s.seq;size=a.trace.stat().st_size;time.sleep(.02);assert s.seq==seq and a.trace.stat().st_size==size
            try:projection(s,'top')
            except ValueError:pass
            else:raise AssertionError('invented geometry')
            rows=[json.loads(x) for x in a.trace.read_text().splitlines()]
            assert rows[0]['retired']==0 and len([r for r in rows if r['kind']=='boundary'])==5
            assert sum(e['spikes'] for r in rows if r['kind']=='boundary' for e in r['events'])==s.total
            return dict(status='PASS',fixture_instructions=5,uart='X',pause_no_events=True,input_fifo=True,
                initial_before_first_instruction=True,trace_count_equal=True,coordinates_missing_reported=True)
        finally:s.close()
def main(a):
    if a.test:return test(a)
    data=a.image.read_bytes();a.image_hash=hashlib.sha256(data).hexdigest();a.mode='cold' if a.cold else 'resume'
    b=fresh(data,a) if a.cold else restore(data,a,a.input)
    s=Session(b,a)
    try:
        if a.serve:
            try:serve(s,a.port)
            except KeyboardInterrupt:pass
        else:
            for _ in range(a.batches):
                s.batch(a.batch)
                if s.error or b.poweroff:break
        if a.checkpoint:save(b,a.checkpoint,a.image_hash)
        return dict(retired=b.steps,delta=b.steps-s.initial_retired,measured_spikes=s.total,boundaries=s.seq,
            coordinates_available=bool(s.spatial),error=s.error,trace_sha256=hashlib.sha256(a.trace.read_bytes()).hexdigest())
    finally:s.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--build-dir',type=Path,required=True);p.add_argument('--cc',default='cc');p.add_argument('--ld');p.add_argument('--image',type=Path);p.add_argument('--input',type=Path);p.add_argument('--cold',action='store_true');p.add_argument('--test',action='store_true');p.add_argument('--coordinates',type=Path);p.add_argument('--graph',type=Path);p.add_argument('--trace',type=Path);p.add_argument('--checkpoint',type=Path);p.add_argument('--serve',action='store_true');p.add_argument('--port',type=int,default=8765);p.add_argument('--batch',type=int,default=32);p.add_argument('--batches',type=int,default=1);p.add_argument('--output',type=Path);p.add_argument('--trace-every',type=int,default=1);p.add_argument('--trace-limit-mib',type=int,default=256);a=p.parse_args()
    if not a.test and not(a.image and a.graph and a.trace and bool(a.input)!=bool(a.cold)):p.error('image, graph, trace and exactly one of cold/input required')
    if not 1<=a.batch<=256 or not 1<=a.batches<=1024 or not 1<=a.trace_every<=1024 or not 1<=a.trace_limit_mib<=4096:p.error('bounded batches/trace')
    r=main(a)
    if a.output:a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps(r,indent=2))

