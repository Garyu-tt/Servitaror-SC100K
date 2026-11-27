# SC500K Workstation (R40)

Local CPU-first runtime. The browser is only the display/control surface. Guest execution continues when the browser is closed. No GitHub API or external AI service is required during execution.

## Run on Linux or WSL2

Install Python 3.10+, python3-venv, GCC/binutils and curl (Debian/Ubuntu: `sudo apt install python3 python3-venv build-essential curl`).

```sh
bash launch-workstation.sh
```

First launch creates a private data directory, installs NumPy 2.x if necessary, downloads the pinned historical RV32 Linux image and verifies its SHA-256. Alternatively supply `--image /path/to/Image` as the first argument. All subsequent computation is local. Open http://127.0.0.1:8765/ if the browser does not open. In WSL2 open this URL in the Windows browser. Native Windows execution is not implemented (runtime compiles ELF shared objects).

The initial state is paused; press Run. Use `--run` to start automatically. Close the browser freely; use Ctrl+C in the terminal for a graceful checkpoint and shutdown. Pause/Step/Save and guest UART input are available in the interface. A detected guest synchronous trap pauses execution for diagnosis. Host failures stop execution and retain the last good checkpoint.

Resume from a saved checkpoint with the same kernel image:

```sh
bash launch-workstation.sh --input /path/to/checkpoint.npz
```

Each launch creates a new session directory and writes checkpoints atomically, at five-minute intervals during execution and on normal exit. Never overwrite the input checkpoint. Optional flags: `--port 8766`, `--no-browser`, `--interval 60`, `--coordinates /path/to/graph-bound-xyz.npz`. Keep the server loopback-only.

## What is visible

Left: actual sampled ALU, decoder and control graph edges and neural activation, diagnostic counters. Circuit layout uses graph levels and stable neuron IDs, not biological coordinates. XY/YZ/XZ are enabled only for an identity-validated coordinate export. Edge highlighting indicates active endpoints, not directly measured individual synaptic transmission. Sampling is intentionally sparse; RAM and other implicit circuits are not traced.

Right: actual 320x200 indexed guest framebuffer converted at scanout, serial output and input. Keyboard events are delivered to the modeled FIFO. The Linux framebuffer/keyboard drivers and full userspace boot are not yet demonstrated. UART is polled; interrupt-driven 8250 support remains pending. This application does not fake a guest desktop or shell.

## Test

```sh
python neural_workstation_r40.py --self-test --build-dir /tmp/sc500k-workstation-test-native --checkpoint /tmp/sc500k-workstation-fixture.npz
```

Use a fresh output path. Fixture is RV32 guest code, not Linux. The test exercises HTTP state/graph, autonomous execution, pause, step, UART input and framebuffer scanout.

