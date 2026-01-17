# Single-Phase TODO

- [x] Enable array-based scan inputs when `items_per_thread == 1` (single-phase).
- [x] Accept string-literal `mode`/`scan_op` in single-phase typing.
- [x] Convert core block scan tests to single-phase (`test_block_sum`, `test_block_scan_known_ops`).
- [x] Support prefix callback ops in single-phase scan.
- [x] Support callable scan ops in single-phase scan.
- [x] Support user-defined types in single-phase scan.
- [ ] Support `coop.ThreadData` inputs for scan (dtype inference).
- [ ] Decide how to handle scalar scan inputs (return semantics) if needed.
- [ ] Verify explicit `temp_storage` handling for scan (single-phase).
- [x] Re-enable/convert skipped block scan tests once features land.
- [x] Fix array-based known-op scan codegen (items_per_thread > 1 initial value handling).
- [x] Run targeted GPU tests for block scan (block_sum, block_scan_known_ops).
- [x] Run targeted GPU tests for block scan (prefix callbacks, callable ops, user-defined types).
