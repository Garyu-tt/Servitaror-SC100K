"""Native desktop front end. The simulation never runs in the UI thread."""
import argparse,base64,hashlib,json,os,sys,threading,time,urllib.request
from pathlib import Path
from types import SimpleNamespace
import tkinter as tk
from tkinter import ttk,filedialog,messagebox
from neural_workstation_r40 import Workstation
from neural_uart_r36 import fresh,restore
from neural_linux_r12 import fixture
from native_bundle import source_digest,digest
from region_diagnostics import REGIONS,EXPLICIT
LABELS={v[0]:k for k,v in REGIONS.items()}
VIEWS={"Топология связей":"circuit","Карта активности / состояния":"raster","XY (GENESIS)":"top","YZ (GENESIS)":"side","XZ (GENESIS)":"front"}

ROOT=Path(os.environ.get('LOCALAPPDATA',Path.home()/'.local/share'))/'Servitaror'
IMAGE_HASH='9feee94f9fbb3f9a761485fb982c78977594d9c0db8b0b411b5677dfb5efc325'
IMAGE_URL='https://raw.githubusercontent.com/bane9/RISCVBox/main/linux/prebuilt/Image'
BG='#101714';PANEL='#18221c';FG='#dcebe0';GREEN='#9de6aa';MUTED='#9baea1'

def image():
    ROOT.mkdir(parents=True,exist_ok=True)
    p=ROOT/'Image'
    if p.exists() and digest(p)==IMAGE_HASH:return p.read_bytes()
    with urllib.request.urlopen(IMAGE_URL,timeout=60) as r:data=r.read(16<<20)
    if hashlib.sha256(data).hexdigest()!=IMAGE_HASH:raise ValueError('Linux image SHA-256 mismatch')
    tmp=p.with_suffix('.tmp');tmp.write_bytes(data);os.replace(tmp,p);return data

def create(data,a):
    h=hashlib.sha256(data).hexdigest()
    b=restore(data,a,a.input) if a.input else fresh(data,a)
    return Workstation(b,a,h)

class Desktop:
    def __init__(self,root):
        self.root=root;self.w=None;self.loading=False;self.closing=False;self.photo=None;self.lastframe=None
        self.graphkey=None;self.serialtext='';self.result=None;self.error=None
        root.title('SERVITAROR | Cogitator Workstation');root.geometry('1320x850');root.minsize(900,650);root.configure(bg=BG)
        style=ttk.Style(root);style.theme_use('clam')
        style.configure('.',background=BG,foreground=FG,font=('Segoe UI',10))
        style.configure('TButton',background='#293b2e',foreground=FG,padding=8)
        style.map('TButton',background=[('active','#3e5845'),('disabled',PANEL)],foreground=[('disabled','#718478')])
        style.configure('TCombobox',fieldbackground=PANEL,background=PANEL,foreground=FG)
        style.map('TCombobox',fieldbackground=[('readonly',PANEL)],foreground=[('readonly',FG)],selectbackground=[('readonly','#293b2e')],selectforeground=[('readonly',FG)])
        root.option_add('*TCombobox*Listbox.background',PANEL)
        root.option_add('*TCombobox*Listbox.foreground',FG)
        root.option_add('*TCombobox*Listbox.selectBackground','#365641')
        root.option_add('*TCombobox*Listbox.selectForeground',FG)
        style.configure('TLabel',background=BG,foreground=FG)
        header=ttk.Frame(root,padding=14);header.pack(fill='x')
        ttk.Label(header,text='SERVITAROR  /  SC500K',font=('Segoe UI',18,'bold'),foreground=GREEN).pack(side='left')
        self.status=tk.StringVar(value='ГОТОВ К ЗАПУСКУ');ttk.Label(header,textvariable=self.status,foreground=GREEN).pack(side='right')
        bar=ttk.Frame(root,padding=(14,0,14,12));bar.pack(fill='x')
        self.buttons={}
        for op,label,fn in [('cold','Новая загрузка',lambda:self.load(None)),('resume','Открыть сохранение',self.open_checkpoint),('run','▶ Пуск',lambda:self.action('run')),('pause','Ⅱ Пауза',lambda:self.action('pause')),('step','Шаг',lambda:self.action('step')),('save','Сохранить',lambda:self.action('save'))]:
            b=ttk.Button(bar,text=label,command=fn);b.pack(side='left',padx=(0,7));self.buttons[op]=b
        panes=ttk.Panedwindow(root,orient='horizontal');panes.pack(fill='both',expand=True,padx=14,pady=(0,14))
        left=ttk.Frame(panes,padding=9);right=ttk.Frame(panes,padding=9);panes.add(left,weight=1);panes.add(right,weight=1)
        ttk.Label(left,text='НЕЙРОННЫЕ КОНТУРЫ',foreground=GREEN,font=('Segoe UI',11,'bold')).pack(anchor='w')
        row=ttk.Frame(left);row.pack(fill='x',pady=7)
        self.region=tk.StringVar(value=REGIONS['alu'][0]);self.view=tk.StringVar(value='Топология связей')
        self.regionbox=ttk.Combobox(row,textvariable=self.region,values=list(LABELS),state='readonly',width=24)
        self.regionbox.pack(side='left',fill='x',expand=True)
        self.viewbox=ttk.Combobox(row,textvariable=self.view,values=list(VIEWS)[:2],state='readonly',width=25);self.viewbox.pack(side='left',padx=8)
        self.regionbox.bind('<<ComboboxSelected>>',self.change_region)
        self.rate=tk.StringVar(value='1.0')
        ratebar=ttk.Frame(left);ratebar.pack(fill='x')
        ttk.Label(ratebar,text='Обновление диагностики, с:').pack(side='left')
        ratebox=ttk.Combobox(ratebar,textvariable=self.rate,values=['0.5','1.0','2.0','5.0'],state='readonly',width=5)
        ratebox.pack(side='left',padx=6);ratebox.bind('<<ComboboxSelected>>',lambda e:self.action('refresh_rate',seconds=float(self.rate.get())))
        self.brain=tk.Canvas(left,bg='#090e0b',highlightbackground='#34463a',highlightthickness=1);self.brain.pack(fill='both',expand=True)
        self.graphinfo=tk.StringVar(value='Реальные связи графа появятся после загрузки');ttk.Label(left,textvariable=self.graphinfo,wraplength=600,foreground=MUTED).pack(anchor='w',pady=6)
        self.metrics=tk.StringVar(value='Нет активной системы')
        ttk.Label(left,text='ДИАГНОСТИКА',foreground=GREEN,font=('Segoe UI',11,'bold')).pack(anchor='w',pady=(6,0))
        ttk.Label(left,textvariable=self.metrics,font=('Consolas',11),justify='left').pack(anchor='w',pady=6)
        self.checkpoint=tk.StringVar(value='');ttk.Label(left,textvariable=self.checkpoint,wraplength=600,foreground=MUTED).pack(anchor='w')
        ttk.Label(right,text='ЭКРАН ГОСТЕВОЙ СИСТЕМЫ',foreground=GREEN,font=('Segoe UI',11,'bold')).pack(anchor='w')
        self.fb=tk.Canvas(right,bg='black',highlightbackground='#34463a',highlightthickness=1,height=280,takefocus=True);self.fb.pack(fill='both',expand=True,pady=7)
        self.fb.bind('<Button-1>',lambda e:self.fb.focus_set())
        for event,pressed in [('<KeyPress>',True),('<KeyRelease>',False)]:
            self.fb.bind(event,lambda e,p=pressed:self.key(e,p))
        ttk.Label(right,text='UART  /  TERMINAL',foreground=GREEN,font=('Segoe UI',11,'bold')).pack(anchor='w')
        log=ttk.Frame(right);log.pack(fill='both',expand=True,pady=7)
        self.terminal=tk.Text(log,height=14,bg='#090e0b',fg='#b9e7bf',insertbackground=FG,font=('Consolas',10),wrap='word',state='disabled',borderwidth=0)
        scroll=ttk.Scrollbar(log,command=self.terminal.yview);self.terminal.configure(yscrollcommand=scroll.set);scroll.pack(side='right',fill='y');self.terminal.pack(side='left',fill='both',expand=True)
        row=ttk.Frame(right);row.pack(fill='x')
        self.entry=ttk.Entry(row);self.entry.pack(side='left',fill='x',expand=True);self.entry.bind('<Return>',lambda e:self.send())
        ttk.Button(row,text='Отправить',command=self.send).pack(side='left',padx=(8,0))
        self.rx=tk.StringVar(value='Гостевой ввод');ttk.Label(right,textvariable=self.rx,foreground=MUTED).pack(anchor='w',pady=5)
        self.msg=tk.StringVar(value='');ttk.Label(root,textvariable=self.msg,foreground='#e9c38b',wraplength=1100,padding=(14,0,14,10)).pack(fill='x')
        root.protocol('WM_DELETE_WINDOW',self.close)
        root.after(150,self.refresh)
    def open_checkpoint(self):
        p=filedialog.askopenfilename(title='Сохранение мозга',filetypes=[('Checkpoint','*.npz')],initialdir=str(ROOT))
        if p:self.load(Path(p))
    def load(self,p):
        if self.loading or self.w:return
        self.loading=True;self.status.set('ЗАГРУЗКА НЕЙРОННОГО ЯДРА')
        self.msg.set('Подготовка образа Linux и проверка встроенных C-библиотек…')
        def work():
            try:
                session=ROOT/time.strftime('session-%Y%m%d-%H%M%S')
                session.mkdir(parents=True,exist_ok=False)
                a=SimpleNamespace(build_dir=session/'native',checkpoint=session/'checkpoint.npz',input=p,coordinates=None,cc='cc',ld=None,interval=300,fixture=False)
                w=create(image(),a);w.thread.start();self.result=w
            except Exception as e:self.error=repr(e)
        threading.Thread(target=work,daemon=True).start()
    def action(self,op,**kw):
        if not self.w:return
        # Worker queue is bounded; UI does not block on saves.
        try:
            import queue
            q=queue.Queue(1);self.w.commands.put_nowait((op,kw,q))
            def check():
                if self.closing:return
                try:ok,m=q.get_nowait()
                except queue.Empty:self.root.after(100,check);return
                if not ok:self.msg.set(m)
            self.root.after(100,check)
        except Exception as e:self.msg.set(str(e))
    def send(self):
        v=self.entry.get()+'\n'
        if not 1<=len(v.encode())<=32:self.msg.set('UART: не более 32 байт за отправку');return
        self.action('serial',text=v);self.entry.delete(0,'end')
    def key(self,e,p):
        if 1<=e.keycode<=255:self.action('key',code=e.keycode,pressed=p)
        return 'break'
    def change_region(self,event=None):
        name=LABELS[self.region.get()]
        views=list(VIEWS)[:2] if name in EXPLICIT else [list(VIEWS)[1]]
        if name in EXPLICIT and self.w and self.w.spatial:views=list(VIEWS)
        self.viewbox.configure(values=views)
        if self.view.get() not in views:self.view.set(views[0])
        self.graphkey=None
    def draw_graph(self,s):
        name=LABELS[self.region.get()];view=VIEWS[self.view.get()]
        if name not in EXPLICIT:
            self.draw_region(s,name);return
        g=self.w.graphs[name]
        c=self.brain;w,h=max(c.winfo_width(),20),max(c.winfo_height(),20)
        key=(name,view,w,h,s['sample_boundary'])
        if key==self.graphkey:return
        self.graphkey=key;c.delete('all')
        ev=s['latest'].get(name)
        if view=='raster':
            bits=bytes.fromhex(ev['rail_bits']) if ev else b''
            values=[(bits[i//8]>>(i%8))&1 for i in range(g['neurons']//5)] if bits else []
            self.raster(values,128,0,lambda x:GREEN if x else '#24342a')
            self.graphinfo.set(REGIONS[name][1]+f" · {g['neurons']:,} нейронов. Реальные спайки последнего измеренного вычисления; 1 ячейка = 5 синхронных нейронов.")
            return
        pts=g['views'].get(view,g['views']['circuit'])
        lo=[min(p[i] for p in pts) for i in (0,1)];hi=[max(p[i] for p in pts) for i in (0,1)]
        sx=(w-24)/max(hi[0]-lo[0],1e-9);sy=(h-24)/max(hi[1]-lo[1],1e-9)
        if view!='circuit':sx=sy=min(sx,sy)
        xy=[(12+(p[0]-lo[0])*sx,12+(p[1]-lo[1])*sy) for p in pts]
        bits=bytes.fromhex(ev['rail_bits']) if ev else b''
        active=[bool(bits[i//40]&(1<<((i//5)%8))) if i//40<len(bits) else False for i in g['ids']]
        for i,j,_ in g['edges']:c.create_line(*xy[i],*xy[j],fill='#6ca779' if active[i] and active[j] else '#2b4031')
        for (x,y),on in zip(xy,active):
            r=2 if on else 1;c.create_oval(x-r,y-r,x+r,y+r,fill=GREEN if on else '#526d59',outline='')
        self.graphinfo.set(REGIONS[name][1]+f" · {g['neurons']:,} нейронов · {g['synapses']:,} синапсов · выборка {len(g['ids'])} узлов. Схема топологии, не анатомия. Спайки измерены; связи подсвечены по активным концам.")
    def raster(self,values,cols,offset,color):
        c=self.brain;w=max(c.winfo_width(),20);h=max(c.winfo_height(),20)
        if not values:
            c.create_text(w/2,h/2,text='Нет измеренных данных',fill=MUTED);return
        rows=(len(values)+cols-1)//cols
        sx=(w-16)/cols;sy=max(1,h-16-offset)/max(rows,1)
        for i,x in enumerate(values):
            a=8+(i%cols)*sx;b=8+offset+(i//cols)*sy
            c.create_rectangle(a,b,a+max(.5,sx-.6),b+max(.5,sy-.6),fill=color(x),outline='')
    def draw_region(self,s,name):
        d=s['regions'][name];c=self.brain
        w,h=max(c.winfo_width(),20),max(c.winfo_height(),20)
        key=(name,w,h,s['boundary'],str(d['stats']))
        if key==self.graphkey:return
        self.graphkey=key;c.delete('all')
        stat=d['stats'];kind=d['kind']
        self.graphinfo.set(REGIONS[name][1]+(' · '+f"{d['cells']:,} логических нейронов (оценка)" if d['cells'] is not None else ' · Общие контуры / хост; отдельная нейросеть отсутствует'))
        if kind=='weights':
            raw=list(bytes.fromhex(d['bits']));cols=d['shape'][1]
            c.create_text(8,8,anchor='nw',text=f"Банк {stat['bank']} / строка {stat['row']} / колонка {stat['col']}\nЧтений {stat['reads']:,} | Записей {stat['writes']:,}\nПоследняя операция: {stat['last']}\nРеальные пары Q4-весов (не спайки)",fill=FG,font=('Consolas',10))
            self.raster(raw,cols,85,lambda x:'#93dca0' if x==0x71 else '#344b3b' if x==0x17 else '#dc9363')
        elif kind=='rails':
            raw=bytes.fromhex(d['bits']);n=d['shape'][0]*10
            values=[sum((raw[j//8]>>(j%8))&1 for j in range(i,i+5)) for i in range(0,n,5)]
            stats=' | '.join(f'{k}: {v}' for k,v in stat.items() if k!='addresses')
            c.create_text(8,8,anchor='nw',text=stats+'\nРеальные состояния групп из 5 клеток, не события спайков.',width=w-16,fill=FG)
            self.raster(values,80,60,lambda x:GREEN if x==5 else '#293c30' if x==0 else '#dc9363')
        else:
            c.create_text(16,16,anchor='nw',text='\n'.join(f'{k}: {v}' for k,v in stat.items())+'\n\nДоступны реальные счётчики; отдельной трассировки\nэтой области пока нет.',fill=FG,font=('Consolas',12),width=w-32)
    def refresh(self):
        if self.closing:return
        if self.result:
            self.w=self.result;self.result=None;self.loading=False;self.msg.set('')
            self.change_region()
        if self.error:
            self.loading=False;self.status.set('ОШИБКА ЗАГРУЗКИ');self.msg.set(self.error);self.error=None
        s=self.w.snapshot if self.w else None
        for op,b in self.buttons.items():
            enabled=(not self.loading and not self.w) if op in ('cold','resume') else bool(s and not s['error'] and not s['poweroff'] and (op=='save' or op=='pause' and s['running'] or op in ('run','step') and not s['running']))
            b.configure(state='normal' if enabled else 'disabled')
        if s:
            self.status.set('ОШИБКА' if s['error'] else 'РАБОТАЕТ' if s['running'] else 'ПАУЗА')
            self.metrics.set(f"Инструкции {s['retired']:>12,}   Скорость {s['ips']:>8,}/с\nALU        {s['alu']:>12,}   PC       {s['pc']:>10}\nПроверки   {s['checks']:>12,}   IRQ      {s['irq']:>10,}\nMMU        {s['walks']:>12,}   Trap     {s['traps']:>10,}")
            if s['checkpoint']:self.checkpoint.set('Сохранено: '+s['checkpoint']['path'])
            if s['error'] or s['trap']:self.msg.set(str(s['error'] or s['trap']))
            self.rx.set(f"UART RX: {s['rx_free']} байт свободно · экран 320×200")
            if s['uart']!=self.serialtext:
                self.serialtext=s['uart'];self.terminal.configure(state='normal');self.terminal.delete('1.0','end');self.terminal.insert('end',s['uart']);self.terminal.see('end');self.terminal.configure(state='disabled')
            scale=max(1,min(self.fb.winfo_width()//320,self.fb.winfo_height()//200))
            fkey=(s['rgb'],scale)
            if fkey!=self.lastframe:
                self.lastframe=fkey;raw=b'P6\n320 200\n255\n'+base64.b64decode(s['rgb'])
                self.photo=tk.PhotoImage(data=raw,format='PPM').zoom(scale,scale);self.fb.delete('all');self.fb.create_image(self.fb.winfo_width()//2,self.fb.winfo_height()//2,image=self.photo)
            self.draw_graph(s)
        self.root.after(400,self.refresh)
    def close(self):
        if self.loading:
            self.msg.set('Дождитесь окончания инициализации перед закрытием');return
        self.closing=True
        if not self.w:self.root.destroy();return
        self.status.set('СОХРАНЕНИЕ СОСТОЯНИЯ')
        t=threading.Thread(target=self.w.close);t.start()
        def finish():
            if t.is_alive():self.root.after(100,finish)
            else:self.root.destroy()
        self.root.after(100,finish)

def main():
    p=argparse.ArgumentParser();p.add_argument('--prepare-native',type=Path);p.add_argument('--self-test',type=Path);p.add_argument('--report',type=Path);p.add_argument('--cc',default='gcc')
    a=p.parse_args()
    if a.prepare_native or a.self_test:
        out=(a.prepare_native or a.self_test).resolve();out.mkdir(parents=True,exist_ok=True)
        opts=SimpleNamespace(build_dir=out/'native',checkpoint=out/'checkpoint.npz',input=None,coordinates=None,cc=a.cc,ld=None,interval=300,fixture=True)
        w=create(fixture(),opts)
        if a.prepare_native:
            m={}
            for lib in opts.build_dir.rglob('*.dll' if os.name=='nt' else '*.so'):
                s=lib.with_suffix('.c');m[lib.name]=dict(source_sha256=source_digest(s),binary_sha256=digest(lib))
            assert len(m)>=8,m
            (opts.build_dir/'manifest.json').write_text(json.dumps(m,indent=2))
        w.thread.start()
        try:
            w.submit('step');w.submit('run');deadline=time.monotonic()+15
            while w.snapshot['retired']<20 and time.monotonic()<deadline:time.sleep(.1)
            w.submit('pause');assert w.snapshot['retired']>=20 and w.snapshot['uart']=='X'
            assert w.snapshot['latest'] and len(base64.b64decode(w.snapshot['rgb']))==192000
            w.submit('serial',{'text':'test\n'});assert w.snapshot['rx_free']==27
            if a.self_test:
                root=tk.Tk()
                def callback_error(kind,value,tb):
                    raise value.with_traceback(tb)
                root.report_callback_exception=callback_error
                ui=Desktop(root);ui.w=w;root.update();ui.refresh();root.update();assert ui.photo is not None
                for name in REGIONS:
                    ui.region.set(REGIONS[name][0]);ui.change_region();ui.draw_graph(w.snapshot)
                assert set(w.snapshot['regions'])==set(REGIONS)-set(EXPLICIT)
                assert set(w.graphs)==set(EXPLICIT)
                assert len(REGIONS)==18
                for name in EXPLICIT:
                    ui.region.set(REGIONS[name][0]);ui.change_region();ui.view.set('Карта активности / состояния');ui.draw_graph(w.snapshot)
                style=ttk.Style(root)
                assert style.lookup('TCombobox','foreground',('readonly',))==FG
                assert style.lookup('TCombobox','fieldbackground',('readonly',))==PANEL
                root.destroy()
        finally:w.close()
        assert opts.checkpoint.is_file()
        report=dict(status='PASS',retired=w.snapshot['retired'],frozen=bool(getattr(sys,'frozen',False)),checkpoint=digest(opts.checkpoint),native_files=len(m) if a.prepare_native else None,regions=len(REGIONS),state_regions=len(w.snapshot['regions']))
        if a.report:a.report.write_text(json.dumps(report,indent=2))
        elif sys.stdout:print(json.dumps(report))
        return
    root=tk.Tk();Desktop(root);root.mainloop()
if __name__=='__main__':
    if '--self-test' in sys.argv or '--prepare-native' in sys.argv:
        try:main()
        except BaseException:
            import traceback
            error=traceback.format_exc()
            if '--report' in sys.argv:
                Path(sys.argv[sys.argv.index('--report')+1]).write_text(json.dumps(dict(status='FAIL',error=error),indent=2),encoding='utf-8')
            if sys.stderr:sys.stderr.write(error)
            sys.exit(1)
    else:main()

