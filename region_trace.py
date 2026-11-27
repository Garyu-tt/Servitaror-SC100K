"""Bounded, read-only execution trace. No guest instruction substitution.
Explicit graphs: measured representative spikes. Other regions: observed
operation boundaries, actual rail/weight state and host operations labeled.
"""
from collections import deque
import numpy as np
from region_diagnostics import REGIONS,EXPLICIT,MEMORY,railmap
from neural_atomic_r06 import A_OPS

# Functional stage descriptions, not invented neuron-to-neuron edges.
STAGES={
 'registers':('Адрес','Селектор','Рельсы регистра','Read / write'),
 'pc':('Вход PC','Feedback','Рельсы PC','Readout'),
 'csr':('Адрес CSR','Маски / feedback','Рельсы CSR','Readout'),
 'priv':('Режим','Feedback','Рельсы режима','Readout'),
 'math':('Операнды','AND / OR','Перенос / mux','Результат'),
 'ram':('Банк','Строка','Колонка','Q4 веса','Read / write'),
 'disk':('Банк','Строка','Колонка','Q4 веса','Read / write'),
 'framebuffer':('Банк','Строка','Колонка','Q4 веса','Пиксели'),
 'palette':('Банк','Строка','Колонка','Q4 веса','RGB'),
 'gpu':('Маски','Селектор','Hold / put','Пиксели'),
 'keyboard':('Событие','Селектор FIFO','Рельсы','Readout'),
 'uart':('Байт / MMIO','Селектор FIFO','Рельсы RX','Readout'),
 'mmu':('VA','TLB / walker','Права доступа','PA'),
 'timer':('Такт / MMIO','Pending','Выбор IRQ','Доставка'),
 'atomic':('LR / SC / AMO','Резервирование','ALU / mux','RAM')}
def scalar(x):
    if x is None or isinstance(x,(str,bool,int,float)):return x
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(tuple,list)):return [scalar(v) for v in x[:12]]
    return type(x).__name__

class RegionTrace:
    def __init__(self,b):
        self.b=b;self.active=False;self.sampled=False;self.seq=0
        self.history={k:deque(maxlen=48) for k in REGIONS}
        self.latest={};self.count={k:0 for k in REGIONS}
        self.install()
    def emit(self,region,op,args=(),result=None,rails=None,route=None,kind='operation',error=None,boundary=None,**extra):
        self.seq+=1;self.count[region]+=1
        e=dict(seq=self.seq,boundary=self.b.boundaries if boundary is None else boundary,
            op=op,args=scalar(args),result=scalar(result),kind=kind,error=error,route=scalar(route),**extra)
        if rails is not None:e['bits'],e['shape']=railmap(rails)
        self.history[region].append(e);self.latest[region]=e
    def hook(self,obj,method,region,rare=False,filter=None,payload=None,kind='operation'):
        old=getattr(obj,method)
        def wrapped(*args,**kw):
            observe=self.active and (self.sampled or rare) and (filter is None or filter(args))
            if not observe:return old(*args,**kw)
            try:r=old(*args,**kw)
            except Exception as e:
                self.emit(region,method,args,error=type(e).__name__,kind=kind);raise
            extra=payload(args,r) if payload else {}
            self.emit(region,method,args,r,kind=kind,**extra)
            return r
        setattr(obj,method,wrapped)
    def install(self):
        b=self.b
        for name,attr in MEMORY.items():
            m=getattr(b,attr)
            def mem_payload(args,result,m=m):
                route=getattr(m,'last_route',None)
                if route is None:return {}
                bank,row,col=route
                return dict(route=route,weights=m.weights[bank,row,col].tobytes().hex())
            for op in ('read','write'):
                self.hook(m,op,name,rare=name!='ram',payload=mem_payload,kind='synaptic-access')
            m.program=m.write
        for op in ('read','write'):
            self.hook(b.reg,op,'registers',payload=lambda a,r:dict(rails=b.reg._state[a[0]]),kind='rail-state')
        for op in ('read','commit'):
            self.hook(b.counter,op,'pc',payload=lambda a,r:dict(rails=b.counter.state),kind='rail-state')
        for op in ('raw','read','commit'):
            def csr(a,r,op=op):
                addr=a[0][0] if op=='commit' and a and a[0] else a[0] if a and op!='commit' else None
                rail=b.csrs.state.get(addr,b.csrs.fixed.get(addr))
                return dict(rails=rail) if rail is not None else {}
            self.hook(b.csrs,op,'csr',payload=csr,kind='rail-state')
        self.hook(b.priv,'get','priv',payload=lambda a,r:dict(rails=b.priv.state),kind='rail-state')
        self.hook(b.priv,'set','priv',rare=True,payload=lambda a,r:dict(rails=b.priv.state),kind='rail-state')
        # Gate results are observed, not recomputed. Keep last 48 gate events.
        self.hook(b.math,'gate','math',rare=True,payload=lambda a,r:dict(rails=r,gate=a[2]),kind='measured-gate')
        self.hook(b.math,'run','math',rare=True,kind='operation')
        self.hook(b.gpu,'rectangle','gpu',rare=True,payload=lambda a,r:dict(rails=b.gpu.state),kind='rail-state')
        for name,obj in [('keyboard',b.keyboard),('uart',b.rx)]:
            for op in ('push','pop'):
                self.hook(obj,op,name,rare=True,payload=lambda a,r,obj=obj:dict(rails=obj.state,queued=obj.count),kind='rail-state')
        for op in ('uart_read','uart_write'):
            self.hook(b,op,'uart',rare=True,kind='host-mmio')
        self.hook(b,'translate','mmu',kind='host-dispatch')
        self.hook(b,'take_trap','mmu',rare=True,kind='host-dispatch')
        for op in ('advance','read','write'):
            self.hook(b.timer,op,'timer',rare=op!='advance',kind='host-device')
        self.hook(b,'take_irq','timer',rare=True,kind='host-dispatch')
        self.hook(b,'finish_instruction_extension','atomic',rare=True,filter=lambda a:a[0] in A_OPS,
            payload=lambda a,r:dict(reservation=b.reservation),kind='host-dispatch')
    def explicit(self,events):
        for e in events:
            self.emit(e['region'],'evaluate',kind='measured-spikes',boundary=e['boundary'],
                rail_bits=e['rail_bits'],spikes=e['spikes'],valid=e['valid'],ablated=e['ablated'])
    def snapshot(self):
        return {k:dict(count=self.count[k],latest=self.latest.get(k),history=list(self.history[k]),stages=STAGES.get(k,())) for k in REGIONS}

