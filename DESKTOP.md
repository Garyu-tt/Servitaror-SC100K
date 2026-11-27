# Servitaror Desktop R41

Portable Windows x64 application. Extract the entire archive and double-click Servitaror.exe. No Python, compiler, WSL, or browser installation required. Keep the _internal directory next to the executable. Python, NumPy, Tcl/Tk and verified compiled neural DLLs are bundled. The first launch of the guest downloads the pinned Linux image automatically (6.1 MB); subsequent launches use the verified local copy. No network is needed for simulation. Data and checkpoints: %LOCALAPPDATA%/Servitaror.

Native desktop controls: cold boot, resume NPZ checkpoint, Run, Pause, Step, Save. Closing the window checkpoints the guest and stops the worker. The left panel shows sampled measured activation and structural edges, not anatomical coordinates; the right shows actual guest framebuffer, UART and diagnostic counters. Linux userspace/DOOM not yet demonstrated. Graphics and keyboard guest drivers are pending.

The executable has no online AI dependency. The arithmetic oracle remains independent of the neural execution path. Native C code is compiled only during packaging; bundled graph source and DLL SHA-256 hashes are verified at runtime. First-launch guest image is fetched from the upstream RISCVBox project with SHA-256 9feee94f9fbb3f9a761485fb982c78977594d9c0db8b0b411b5677dfb5efc325.

Windows package is unsigned; Windows may present a SmartScreen warning. Do not disable security controls. Verify the ZIP hash and obtain packages only from the project's build.


## R42 diagnostics and performance

17 named regions: three explicit graphs with topology and activation rasters, seven recurrent-state maps, four synaptic memory maps, and math/MMU/timer/atomic counters (some regions share the same hardware). Memory maps show actual Q4 coefficients of the last accessed row, not firing events. No synthetic anatomy. Display refresh selectable from 0.5 to 5 seconds. Native selector and row gates are factored by proven local truth tables; all coefficients remain validated.

## R43 all-region trace

Both topology and activity/state views are available for every region.
ALU, decoder and control display measured spikes of fixed graphs. Math
captures actual gate outputs; recurrent regions display sampled operation
boundaries and rail states; crossbars record the actual selected bank,
row and column and word coefficients. MMU, IRQ and atomic reservation
dispatch is explicitly labeled host operation, not spikes. Functional
diagrams are not individual neuron graphs or anatomical projections.
History retains 48 events per region, with monotonic sequence and guest
boundary IDs; counts include observed events, not every internal spike.
Default samples eight guest steps each refresh, with rare operations
recorded between samples. Optional full mode observes every instruction
and is slower. Paused diagnostics do not generate guest events.
Checkpoint format remains unchanged.

## R44 boot monitor

Session and active execution time, average retired instructions per active second, UART silence since observation, and a bounded 512-sample history of successful guest instruction fetches. Alternating 31/37-boundary sampling does not fetch extra instructions. A sustained narrow address set with no UART activity is an advisory heuristic, not proof of deadlock; idle and productive loops are possible. Counters restart on checkpoint resume. A separate diagnostics JSON is written alongside the checkpoint; no change to checkpoint format. Copy diagnostics includes the last 8192 UART characters.
