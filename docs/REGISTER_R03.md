# R03: clocked recurrent register file

32 x 32-bit registers stored exclusively in 10,240 persistent dual-rail spike cells, with fixed weighted address selection, read gates, write gates and recurrent feedback. No host integer shadow register file. Host performs phase scheduling, carries spike arrays, and encodes addresses and write data. x0 has a fixed write-inhibit. A failed dual-rail write is rejected before feedback. The single read port is invoked twice for two operands.

Capacity accounting: ALU 28,650 + decoder 4,870 + register bank 41,980 = 75,500 logical cells out of 131,072. These are separate fixed circuits, not one integrated running GENESIS network. The register circuit is vectorized using Q4 bundle equations, not a fully enumerated synapse graph.

Run `python neural_register_r03.py`. Test checks random writes/readback, x0, fault ablation, and the 64-instruction guest smoke program. Not a complete ISA conformance test or Linux boot.

Next: instruction-field extraction/control, PC/traps, RV32M/A and privileged architecture. RAM and external devices may remain conventional peripherals.
