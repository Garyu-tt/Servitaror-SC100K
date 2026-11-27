# Servitaror Desktop R41

Portable Windows x64 application. Extract the entire archive and double-click Servitaror.exe. No Python, compiler, WSL, or browser installation required. Keep the _internal directory next to the executable. Python, NumPy, Tcl/Tk and verified compiled neural DLLs are bundled. The first launch of the guest downloads the pinned Linux image automatically (6.1 MB); subsequent launches use the verified local copy. No network is needed for simulation. Data and checkpoints: %LOCALAPPDATA%/Servitaror.

Native desktop controls: cold boot, resume NPZ checkpoint, Run, Pause, Step, Save. Closing the window checkpoints the guest and stops the worker. The left panel shows sampled measured activation and structural edges, not anatomical coordinates; the right shows actual guest framebuffer, UART and diagnostic counters. Linux userspace/DOOM not yet demonstrated. Graphics and keyboard guest drivers are pending.

The executable has no online AI dependency. The arithmetic oracle remains independent of the neural execution path. Native C code is compiled only during packaging; bundled graph source and DLL SHA-256 hashes are verified at runtime. First-launch guest image is fetched from the upstream RISCVBox project with SHA-256 9feee94f9fbb3f9a761485fb982c78977594d9c0db8b0b411b5677dfb5efc325.

Windows package is unsigned; Windows may present a SmartScreen warning. Do not disable security controls. Verify the ZIP hash and obtain packages only from the project's build.

