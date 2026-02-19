# Mamba Selective Scan TODO

- [x] Stabilize kernel-traits gpu_dataclass path (prepare_args flattening + attr handling).
- [x] Implement Float2 POD type + scan op + prefix callback op for selective scan.
- [x] Implement simplified selective scan forward kernel (kNRows=1, non-complex, fixed B/C, single chunk).
- [x] Add test coverage + CPU reference for the simplified kernel.
- [x] Enable two-phase block_scan in kernel traits (typing + instance defaults) and switch scan call to traits.
- [ ] Iterate toward parity: variable B/C, multiple chunks (running prefix), optional z, complex path, and vectorized load/store.
- [ ] Document kernel-traits pattern + coop primitives used (final MAMBA-README.md).
