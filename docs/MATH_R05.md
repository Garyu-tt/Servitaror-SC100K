# R05: neural RV32M microsequencer

All eight RV32M instructions are implemented with dual-rail Q4 threshold gates. The 64-bit multiplication uses 32 fixed shift/add rounds; unsigned division uses 32 restoring rounds with a 33-bit remainder. Carry uses a parallel-prefix network. Signed magnitude conversion, quotient correction, and zero-divisor behavior are neural gates. Host performs fixed wire permutation, phase scheduling, opcode-dependent routing and spike readout; Python integer arithmetic on guest operands is restricted to the test oracle. The decoder now recognizes 47 patterns.

Run `python neural_math_r05.py`. Recorded host run passed 160 arithmetic vectors, including division by zero and signed overflow, ablation rejection and an 11-instruction guest program executing all eight M operations. All five earlier smoke scripts passed again.

Capacity estimate: 110,200 logical cells including a conservative 30,000-cell reusable math lane budget; this is not a measured integrated neuron count. Fixed vectorized gates are not an autonomously learned or continuous GENESIS brain. RV32A, privileged ISA, MMU and Linux boot remain open.
