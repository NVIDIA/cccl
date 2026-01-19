# Mamba Selective Scan Log

## 2026-01-19
- Started the mamba selective-scan port after fixing kernel-traits gpu_dataclass crashes.
- Plan: implement a simplified forward kernel first (kNRows=1, non-complex, fixed B/C, single chunk), then expand.
- Files to add: `tests/coop/mamba_selective_scan_fwd.py`, `tests/coop/test_mamba_selective_scan_fwd.py`.
- Implemented Float2 POD type, `ssm_scan_op`, and `SSMScanPrefixCallbackOp` plus CPU reference.
- Kernel uses kernel traits for block load/store; scan remains single-phase due to block_scan instance typing issues.
- Added test coverage and verified: `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba`.
- Follow-up: enable two-phase block_scan in traits (typing defaults for mode/scan_op/algorithm + kwargs) then switch scan call.
