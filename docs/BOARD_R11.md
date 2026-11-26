# R11: boot platform

32 MiB RAM at 0x80000000, byte-oriented NS16550-style UART transmit, deterministic timer, generated FDT, S-mode entry with hart ID in a0 and FDT pointer in a1. Minimal host SBI interception supports BASE discovery, TIME, SRST shutdown and legacy console putchar. This is explicitly a host firmware boundary, not guest OpenSBI or a replacement for guest CPU instructions.

Run `python neural_board_r11.py`. Local board smoke: 21 guest instructions, three SBI calls, UART output `OK`, high-address LR/SC, Sv32 high physical mappings and shutdown. FDT header and tokens checked internally; no external dtc or Linux validation. No UART receive IRQ, PLIC, actual guest firmware or Linux image. Next obtain/build RV32 kernel and initramfs, validate DTB, improve platform conformance, and capture real boot progress. Linux and DOOM remain unbooted. See #9.
