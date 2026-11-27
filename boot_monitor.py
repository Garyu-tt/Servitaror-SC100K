"""Non-invasive boot progress monitor; sampled executed fetches, no extra guest reads."""
from collections import Counter,deque
import time

class BootMonitor:
    def __init__(self,b):
        self.b=b;self.started=time.monotonic();self.initial=b.steps
        self.active=0.;self.initial_uart=len(b.serial);self.uart_len=self.initial_uart
        self.uart_time=self.started;self.uart_step=b.steps;self.uart_observed=False
        self.samples=deque(maxlen=512);self.next_sample=b.boundaries
        self.stable_since=None;self.signature=None;self.sample_phase=False
        old=b.fetch
        def fetch(address):
            ins=old(address)
            if b.boundaries>=self.next_sample:
                self.record(address,ins,b.boundaries)
                # Alternating coprime gaps reduce periodic aliasing.
                self.sample_phase=not self.sample_phase
                self.next_sample=b.boundaries+(31 if self.sample_phase else 37)
            return ins
        b.fetch=fetch
    def record(self,pc,word,boundary):
        self.samples.append((boundary,pc,word))
    def report(self):
        b=self.b;now=time.monotonic()
        if len(b.serial)!=self.uart_len:
            self.uart_len=len(b.serial);self.uart_time=now;self.uart_step=b.steps;self.uart_observed=True
        counts=Counter(pc for _,pc,_ in self.samples)
        top=counts.most_common(8);n=len(self.samples)
        coverage=sum(c for _,c in top)/max(1,n)
        narrow=n>=128 and coverage>=.95 and len(counts)<=16
        sig=tuple(sorted(pc for pc,_ in top)) if narrow else None
        if sig!=self.signature:
            self.signature=sig;self.stable_since=self.active if sig else None
        stable=self.active-self.stable_since if self.stable_since is not None else 0.
        # This is a heuristic, not proof of a deadlock. Idle and productive loops may look alike.
        suspect=bool(narrow and stable>=30 and now-self.uart_time>=30)
        last=self.samples[-1] if self.samples else None
        return dict(session_seconds=round(now-self.started,1),active_seconds=round(self.active,3),
            session_retired=b.steps-self.initial,average_ips=round((b.steps-self.initial)/max(.001,self.active)) if self.active else 0,
            uart_silent_seconds=round(now-self.uart_time,1),uart_silent_instructions=b.steps-self.uart_step,
            uart_observed_since_resume=self.uart_observed,uart_bytes=self.uart_len,
            samples=n,unique_pc=len(counts),top_pc=[dict(pc=hex(pc),samples=c) for pc,c in top],
            top8_fraction=round(coverage,3),narrow_window_seconds=round(stable,1),
            possible_loop=suspect,last_fetch=dict(boundary=last[0],pc=hex(last[1]),word=f'0x{last[2]:08x}') if last else None,
            recent_fetches=[dict(boundary=t,pc=hex(pc),word=f'0x{word:08x}') for t,pc,word in list(self.samples)[-32:]])

