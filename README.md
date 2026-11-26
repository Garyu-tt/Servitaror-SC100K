# ServitoriumCogitatorium

CPU-first experimental spiking processor derived from GENESIS R02 Q4 dynamics. Goal: boot Linux without a conventional hidden ALU.

## Reproduce R01

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python neural_alu_r01.py
python neural_rv32i_r01.py
```

Standalone tests need no private GENESIS checkpoint. The host prototype had a 131,072-neuron capacity; the engineered overlay activates 28,650 neurons and 279,000 synapses. It passed 1,640 arithmetic vectors and a 64-instruction bare-metal RV32I program. See recorded JSON reports. Host still performs decode, registers, RAM and device control. No Linux boot yet. This is a fixed feedforward spiking circuit, not a learned CPU, and the original resident brain is not included.

See [roadmap](docs/ROADMAP.md).
