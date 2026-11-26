# R06: single-hart RV32A word operations

Eleven LR/SC/AMO word instructions enter through the neural classifier. RAM bus serializes access and holds reservation metadata. AMOADD/XOR/AND/OR use the existing neural ALU; MIN/MAX use neural comparison and Q4 multiplexer; swap is a direct bus transfer. Both aq/rl bits decode; the synchronous single-hart bus has no independent ordering effect. No multi-hart coherence or DMA.

Run `python neural_atomic_r06.py`. Host run passed 386 decode cases, 126 AMO vectors, 18-instruction guest program and reservation success/failure, overlap/disjoint invalidation and misalignment tests. Estimated 111790 logical cells, not a measured continuous GENESIS graph. Privileged traps, MMU and Linux boot remain open.
