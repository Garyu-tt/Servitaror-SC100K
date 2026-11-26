# R20: corrected Linux MMU replay and FDT diagnostic

Continued clean R19 replay through early RAM clear, expected Sv32 relocation fault at instruction 154401, and another 127000 instructions in virtual addressing. Final retired 281401, delta 169401, 961585 ALU calls, 700261 new arithmetic oracle comparisons, no mismatches. 27 logged SLTIU examples, including nonzero results that the old dispatcher incorrectly returned as zero. Oracle is verification only, never guest execution. R19 CI 2/2 PASS.

Read-only diagnostic at 229401 instructions shows guest RAM log line OF: fdt: Ignoring memory range 0x80000000 - 0x80400000 (the direct log buffer, not the nearby stale-suffix copy). Previous R18 output contained 0x20000000000. FDT bytes remain unchanged. The corrected range is consistent with Linux 5.15 early_init_dt_add_memory_arch trimming memory below __pa(PAGE_OFFSET), here kernel load address 0x80400000. This explains the remaining warning; full memory initialization not yet verified. No modifications to guest image or memory to bypass it.

Final checkpoint SHA-256 782bff6cec5ffedeb7dd3b6a540b05abf3d49226e24363b974d80773f69b3ca1, saved locally, no binaries committed. UART empty, Linux shell and DOOM not reached. Fixed separate neural circuits and host dispatch/peripherals remain. Supports #9.
