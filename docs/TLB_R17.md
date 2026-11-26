# R17: conservative Sv32 page-walk cache

The host cache stores successful R10 neural page-walk mappings. Key includes satp, privilege, SUM/MXR, access type and virtual 4 KiB page. On hits the neural ALU still masks the offset and combines it with the physical base. All observed page-table pages are monitored; any write through the board physical bus to a monitored page flushes the cache, as does SFENCE.VMA. No negative caching; A/D updates remain in the original walker. Cache is derived state and starts empty on checkpoint restore. Single hart and no DMA/direct external RAM changes are required. No claim of a fully neural TLB or full RISC-V conformance.

Run `python neural_tlb_r17.py --build-dir /tmp/sc100k-native-r17`. Local PASS: 27 mapping checks, permissions, partial PTE writes, remapping, SFENCE, superpage, exact architectural state after 64 paged guest instructions versus R15 (excluding diagnostic counters).

Actual Linux resumed from R16: two segments of 16000 instructions, cumulative 189401, 564030 neural ALU calls, no new traps, no UART output. First segment 22.313 s (717 IPS), second 22.671 s (706 IPS). Combined 40928 cache hits and 72 misses (99.82% hit rate), workload specific. Final checkpoint SHA-256 f78f7d2bab77aa354aa27e4af9d91b5ba7abd64d85e00d2046d7001fafc41647. No checkpoint binary committed. Linux still booting; DOOM not executed. Supports #9.
