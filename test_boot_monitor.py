from types import SimpleNamespace
from boot_monitor import BootMonitor
b=SimpleNamespace(steps=0,boundaries=0,serial=bytearray(),fetch=lambda a:0x6f)
m=BootMonitor(b)
for i in range(18000):
    b.steps+=1;b.boundaries+=1;b.fetch(0xc0001000)
assert len(m.samples)==512
assert not m.report()['possible_loop']
m.active=31;m.uart_time-=31
assert m.report()['possible_loop']
b.serial.extend(b'progress')
assert not m.report()['possible_loop']
for i in range(18000):
    b.steps+=1;b.boundaries+=1;b.fetch(0xc0000000+(i%997)*4)
r=m.report()
assert r['unique_pc']>16 and not r['possible_loop']
print('PASS: bounded samples, sustained loop, UART resets alert, varied PC clears alert')

