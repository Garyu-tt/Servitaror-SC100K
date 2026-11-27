"""Read-only region diagnostics. No invented synapses or spatial coordinates."""
import numpy as np
from neural_register_r03 import validate

EXPLICIT={'alu':('АЛУ','Сложение, вычитание, логика, сдвиги, сравнения.'),
'decoder':('Декодер инструкций','Преобразует машинную инструкцию RISC-V в сигнал выбора операции.'),
'control':('Управление','Выбор следующего PC, ветвление и переходы.')}
REGIONS={**EXPLICIT,
 'registers':('Регистры CPU','32 × 32 бита, рекуррентные двухрельсовые ячейки.'),
 'pc':('Счётчик команд PC','Адрес текущей инструкции, рекуррентное состояние.'),
 'csr':('Системные регистры CSR','Привилегии, обработчики исключений и управление MMU.'),
 'priv':('Режим привилегий','Двухбитное рекуррентное состояние M/S/U.'),
 'math':('Умножение и деление','Нейронные контуры RV32M, многотактная арифметика.'),
 'ram':('Гиппокамп / RAM','64 МиБ адресуемой синаптической памяти, банки, строки и колонки.'),
 'disk':('Долговременная память','16 МиБ синаптического хранилища, программируемые веса.'),
 'framebuffer':('Видеопамять','Индексированные пиксели, 320 × 200.'),
 'palette':('Палитра','256 RGB-цветов в синаптической памяти.'),
 'gpu':('Растеризатор','8 × 8 рекуррентных пикселей, прямоугольные маски.'),
 'keyboard':('Клавиатура','Рекуррентный FIFO событий клавиш.'),
 'uart':('UART RX','Рекуррентный FIFO входных байтов терминала.'),
 'mmu':('MMU / адресация','Нейронная ALU выполняет адресные операции; TLB и часть контроля на хосте.'),
 'timer':('Таймер / IRQ','Таймер и доставка прерываний: часть устройства моделируется на хосте.'),
 'atomic':('Атомарные операции','RV32A использует ALU и память; резервирование на хосте.')}
MEMORY={'ram':'memory','disk':'disk','framebuffer':'framebuffer','palette':'palette'}
def railmap(a):
    a=np.asarray(a).reshape(-1,2,5)
    # Actual state cells, not a visualization of firing events.
    return np.packbits(a.reshape(-1),bitorder='little').tobytes().hex(),list(a.shape)
def capture(b,cache):
    d={}
    for name,arr,cells,stats in [
        ('registers',b.reg._state,b.reg.neurons,dict(reads=b.reg.reads,writes=b.reg.writes)),
        ('pc',b.counter.state,b.counter.neurons,dict(commits=b.counter.commits)),
        ('csr',np.stack([*b.csrs.state.values(),*b.csrs.fixed.values()]),b.csrs.neurons,dict(reads=b.csrs.reads,writes=b.csrs.writes,addresses=[hex(a) for a in [*b.csrs.state,*b.csrs.fixed]])),
        ('priv',b.priv.state,b.priv.neurons,dict(mode=b.mode)),
        ('gpu',b.gpu.state,b.gpu.cells,dict(writes=b.gpu_writes)),
        ('keyboard',b.keyboard.state,b.keyboard.cells,dict(queued=b.keyboard.count,reads=b.keyboard.reads,events=b.keyboard.events)),
        ('uart',b.rx.state,b.rx.cells,dict(queued=b.rx.count,received=b.rx.received,reads=b.rx.reads))]:
        bits,shape=railmap(arr);d[name]=dict(kind='rails',cells=cells,shape=shape,bits=bits,stats=stats)
    for name,attr in MEMORY.items():
        m=getattr(b,attr);last=getattr(m,'last_access',None)
        a=last[0] if last else 0
        bank,o=divmod(a,m.rows*m.cols);row,col=divmod(o,m.cols)
        v=m.weights[bank,row].copy()
        d[name]=dict(kind='weights',cells=m.cells,shape=list(v.shape),bits=v.tobytes().hex(),
            stats=dict(bytes=m.n*4,banks=m.banks,rows=m.rows,cols=m.cols,bank=bank,row=row,col=col,
                last=last,reads=m.reads,writes=m.writes,coefficients=int(m.weights.size)))
    d['math']=dict(kind='counters',cells=b.math.neurons,stats=dict(calls=b.math.calls,rounds=b.math.rounds,gates=b.math.gates))
    d['mmu']=dict(kind='counters',cells=None,stats=dict(walks=b.walks,pte_reads=b.pte_reads,ad_updates=b.ad_updates,**cache.report()))
    d['timer']=dict(kind='counters',cells=None,stats=dict(ticks=b.timer.now,interrupts=b.interrupts,returns=b.returns,deadline=b.deadline))
    d['atomic']=dict(kind='counters',cells=None,stats=dict(calls=b.atomic_calls,selects=b.atomic_selects,reservation=b.reservation))
    return d

