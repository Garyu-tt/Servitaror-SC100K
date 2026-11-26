# Linux boot acceptance

1. R01: deterministic dual-rail Q4 arithmetic, test vectors and bare-metal RV32I smoke (done).
2. Neural decoder, control and register interface; full RV32I conformance.
3. RV32M/A, privileged CSR/trap semantics and Sv32 MMU; UART, timer, interrupt controller, DTB and boot firmware.
4. Boot unmodified RV32 Linux kernel to initramfs and BusyBox shell. Record serial log and count neural vs conventional execution.
5. Optimize sparse spike scheduling, latency and resource usage; evaluate plasticity without corrupting ISA correctness.

Host RAM and I/O are allowed peripherals. CPU execution must not silently fall back to conventional arithmetic. R01 is an engineered feedforward overlay, not an autonomously learned CPU or full GENESIS simulation. No Linux boot claim yet.
