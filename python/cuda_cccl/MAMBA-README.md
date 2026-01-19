# Mamba Selective Scan (Numba + cuda.coop)

This document will summarize the process of porting the Mamba v1 selective scan forward kernel to Numba using cuda.coop primitives and the kernel-traits pattern.

## Scope (initial)
- Start with a minimal forward path: kNRows=1, non-complex, fixed B/C, single chunk.
- Use `coop.block.load`, `coop.block.store`, and `coop.block.scan` with a custom scan op + prefix callback.

## Status
- Kernel-traits gpu_dataclass path stabilized (see SINGLE-PHASE-LOG.md).
- Float2 POD type, prefix callback op, simplified kernel, and test are in place.
- Kernel traits currently cover block load/store; block scan is still single-phase due to typing limitations for two-phase instances (see MAMBA-TODO.md).
