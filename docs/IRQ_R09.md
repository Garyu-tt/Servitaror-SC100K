# R09: timer and interrupt delivery

A deterministic CLINT-like timer provides word MMIO for MSIP, MTIME, MTIMECMP and read-only TIME/TIMEH CSRs with counteren checks. Interrupt pending state is neural CSR state, while host peripheral and priority/delegation policy select the interrupt. Existing Q4 ALU computes masks, status updates and vectored addresses. Direct/vectored delivery and return to interrupted code work in the tested scenarios.

Run `python neural_irq_r09.py`. Host run: 15 scenarios including actual guest machine timer MMIO, MRET/SRET roundtrips, masking, priority, counter rollover and access policy. R08 regression passed. Estimated budget 122590 logical cells, not a measured continuous neural simulation. Timer increments per step boundary rather than wall time; supervisor timer/external lines are test injections. No PLIC, SBI, real WFI sleep, MMU, PMP, Sstc or full conformance. Linux and DOOM not yet booted.
