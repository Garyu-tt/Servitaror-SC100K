# Servitaror-SC100K

Formerly ServitoriumCogitatorium. CPU-first experimental spiking processor derived from GENESIS R02 Q4 dynamics. Goal: boot Linux without a conventional hidden ALU or instruction classifier.

## Reproduce R01/R02

```sh
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python neural_alu_r01.py
python neural_rv32i_r01.py
python neural_decoder_r02.py
```

Standalone tests need no private GENESIS checkpoint. The host prototype had a 131,072-neuron capacity. R01 ALU activates 28,650 neurons and passed 1,640 arithmetic vectors. R02 adds 4,870 decoder neurons and classifies 39 implemented instruction patterns. Both cores run a 64-instruction bare-metal RV32I program. Host still performs field extraction, dispatch, register storage, RAM, exception and device control. No Linux boot yet. Fixed engineered spiking circuits, not an autonomously learned CPU; the original resident brain is not included.

See [roadmap](docs/ROADMAP.md).
