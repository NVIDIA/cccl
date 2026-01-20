# Mamba Selective Scan Log

## 2026-01-19
- Started the mamba selective-scan port after fixing kernel-traits gpu_dataclass crashes.
- Plan: implement a simplified forward kernel first (kNRows=1, non-complex, fixed B/C, single chunk), then expand.
- Files to add: `tests/coop/mamba_selective_scan_fwd.py`, `tests/coop/test_mamba_selective_scan_fwd.py`.
- Implemented Float2 POD type, `ssm_scan_op`, and `SSMScanPrefixCallbackOp` plus CPU reference.
- Kernel uses kernel traits for block load/store; scan remains single-phase due to block_scan instance typing issues.
- Added test coverage and verified: `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba`.
- Follow-up: enable two-phase block_scan in traits (typing defaults for mode/scan_op/algorithm + kwargs) then switch scan call.

## 2026-01-20
- Goal: enable two-phase block_scan in kernel traits and get the selective-scan kernel compiling.
- Debugged typing failures: Numba CallableTemplate rejects keyword args for instance calls.
- Changes:
  - `cuda/coop/_decls.py`: reordered block-scan arglist to match signature; accept explicit `NoneType` placeholders in two-phase calls; cleaned debug hooks.
  - `tests/coop/mamba_selective_scan_fwd.py`: switch block_scan call to positional None placeholders and document why.
- Tests:
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (pass)
  - `pytest -q tests/coop/test_block_scan.py -k prefix_op_multi_items` (24 passed, 156 deselected)

## 2026-01-20
- Switched coop typing to AbstractTemplate so instance calls accept kwargs.
- Updated selective-scan kernel to call `traits.block_scan` with keyword args.
- Tests: `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba`.
