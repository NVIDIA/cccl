# Single-Phase TODO

- [x] Enable array-based scan inputs when `items_per_thread == 1` (single-phase).
- [x] Accept string-literal `mode`/`scan_op` in single-phase typing.
- [x] Convert core block scan tests to single-phase (`test_block_sum`, `test_block_scan_known_ops`).
- [x] Support prefix callback ops in single-phase scan.
- [x] Support callable scan ops in single-phase scan.
- [x] Support user-defined types in single-phase scan.
- [ ] Support `coop.ThreadData` inputs for scan (dtype inference).
- [ ] Support `coop.ThreadData` inputs for load/store/exchange (rewrite items_per_thread + dtype inference).
- [ ] Decide how to handle scalar scan inputs (return semantics) if needed.
- [ ] Verify explicit `temp_storage` handling for scan (single-phase).
- [ ] Implement `coop.TempStorage` placeholder handling (allocate shared uint8 buffer from size/alignment).
- [ ] Add tests for `coop.TempStorage` placeholder reuse across primitives.
- [x] Re-enable/convert skipped block scan tests once features land.
- [x] Fix array-based known-op scan codegen (items_per_thread > 1 initial value handling).
- [x] Run targeted GPU tests for block scan (block_sum, block_scan_known_ops).
- [x] Run targeted GPU tests for block scan (prefix callbacks, callable ops, user-defined types).
- [x] Port block reduce/sum to single-phase (typing, rewrite, tests).
- [x] Port block exchange StripedToBlocked to single-phase (typing, rewrite, tests).
- [x] Port block exchange BlockedToStriped variants.
- [x] Port block exchange BlockedToWarpStriped variants.
- [x] Port block exchange ScatterToBlocked variants.
- [x] Port block exchange ScatterToStriped / Flagged / Guarded variants.
- [x] Port block exchange WarpStripedToBlocked variants.
- [x] Port block run-length decode to single-phase (total_decoded_size handling).
- [x] Add single-phase tests for block run-length decode.
- [x] Port block merge sort to single-phase (typing, rewrite, tests).
- [x] Port block radix sort to single-phase (typing, rewrite, tests).
- [x] Add single-phase tests for BlockLoad/BlockStore shared-memory algorithms:
  TRANSPOSE, WARP_TRANSPOSE, WARP_TRANSPOSE_TIMESLICED.
- [x] Add single-phase tests for BlockLoad/BlockStore VECTORIZE (alignment
  requirements + fallback behavior).
- [x] Document BlockLoad/Store algorithm constraints and shared-memory
  behavior (no separate BlockLoadToShared API; algorithms handle smem).
- [ ] Evaluate numba-cuda `280-launch-config-v2` branch for coop launch config needs.
