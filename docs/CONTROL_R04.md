# R04: neural PC and branch control

The PC is stored exclusively as a 32-bit dual-rail spike state. A fixed Q4 control circuit reduces XOR rails for equality, consumes neural ALU signed/unsigned comparison outputs, gates the branch/jump predicates, and multiplexes the sequential and target addresses. Every guest instruction commits the next PC through synaptic threshold feedback. Host checks alignment and stages memory/register updates after validating the next address. Tests exercise taken and untaken misaligned branches/jumps, fault ablation, 1,152 control vectors, 256 PC vectors and the original 64-instruction guest smoke program.

Capacity: 78,920 logical cells across independent circuits (ALU, decoder, register bank, PC and control), of 131,072 budget. These are vectorized engineered circuits, not one continuous GENESIS brain. Host still extracts fields, dispatches instruction classes, gates bus operations and handles traps. No Linux boot, full ISA conformance or privileged ISA yet.

Run `python neural_control_r04.py`.
