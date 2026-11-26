# R08: privilege and synchronous traps

M/S/U modes are backed by dual-rail state. ECALL, EBREAK, MRET, SRET and WFI (hint) are recognized, with synchronous exception dispatch, delegation, EPC/cause/tval storage and status bit updates through neural ALU. Host controls architectural policy and commit sequencing. Faulting instructions do not retire; guest faults are distinguished from host neural failures. Tested MRET to user, delegated user ECALL to supervisor, handler code and SRET to user, and 18 total scenarios. No interrupt delivery, MMU, PMP, counters, TVM/TW/TSR/MPRV or full privileged ISA conformance yet.

Run `python neural_priv_r08.py`. R04–R08 local regression passed. Estimated budget 121950 cells out of 131072, not a measured continuous neural simulation. Owner authorizes expansion up to 500000 cells if necessary; R08 does not grow the checkpoint. Prefer sharing existing lanes before expanding. Linux has not booted.
