# Single-Phase TODO

- [x] Enable array-based scan inputs when `items_per_thread == 1` (single-phase).
- [x] Accept string-literal `mode`/`scan_op` in single-phase typing.
- [x] Convert core block scan tests to single-phase (`test_block_sum`, `test_block_scan_known_ops`).
- [x] Support prefix callback ops in single-phase scan.
- [x] Support callable scan ops in single-phase scan.
- [x] Support user-defined types in single-phase scan.
- [x] Support `coop.ThreadData` inputs for scan (dtype inference).
- [x] Support `coop.ThreadData` inputs for load/store/exchange (rewrite items_per_thread + dtype inference).
- [x] Decide how to handle scalar scan inputs (return semantics) if needed.
- [x] Verify explicit `temp_storage` handling for scan (single-phase).
- [x] Implement `coop.TempStorage` placeholder handling (allocate shared uint8 buffer from size/alignment).
- [x] Add tests for `coop.TempStorage` placeholder reuse across primitives.
- [x] Re-enable/convert skipped block scan tests once features land.
- [x] Fix array-based known-op scan codegen (items_per_thread > 1 initial value handling).
- [x] Run targeted GPU tests for block scan (block_sum, block_scan_known_ops).
- [x] Run targeted GPU tests for block scan (prefix callbacks, callable ops, user-defined types).
- [x] Port block reduce/sum to single-phase (typing, rewrite, tests).
- [x] Support explicit temp_storage for single-phase block reduce/sum.
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
- [x] Add single-phase block adjacent difference (typing, rewrite, tests).
- [x] Add warp load/store/exchange invocables and targeted tests.
- [x] Add single-phase tests for BlockLoad/BlockStore shared-memory algorithms:
  TRANSPOSE, WARP_TRANSPOSE, WARP_TRANSPOSE_TIMESLICED.
- [x] Add single-phase tests for BlockLoad/BlockStore VECTORIZE (alignment
  requirements + fallback behavior).
- [x] Document BlockLoad/Store algorithm constraints and shared-memory
  behavior (no separate BlockLoadToShared API; algorithms handle smem).
- [x] Evaluate numba-cuda `280-launch-config-v2` branch for coop launch config needs.
- [x] Port block shuffle to single-phase (typing, rewrite, tests).
- [x] Port block radix rank to single-phase (typing, rewrite, tests).
- [x] Add warp exchange scatter-to-striped coverage/tests (including ranks dtype).
- [x] Add warp load/store num_valid + oob_default coverage/tests.
- [x] Fix gpu_dataclass kernel-traits argument handling and add test coverage.
- [x] Allow two-phase coop.block.scan instance calls to omit mode/scan_op/items_per_thread (kernel-traits defaulting).
- [x] Switch coop primitives to AbstractTemplate so two-phase instance calls accept kwargs.
- [x] Port warp primitives to single-phase (typing, rewrite, tests).
- [x] Add two-phase instance typing for remaining block/warp primitives.
- [x] Store instance constructor parameters needed for two-phase inference.
- [x] Remove `.create()`/`link=` usage from coop tests and examples.
- [x] Add missing two-phase tests for block primitives.
- [ ] Audit CUB block/warp overload coverage and fill gaps.
- [x] Add warp reduce/sum valid-items overloads.
- [x] Add WarpScan warp_aggregate/valid_items/temp_storage overloads and tests.
- [x] Add block load oob_default overload.
- [x] Add block discontinuity tile predecessor/successor overloads.
- [x] Add block shuffle prefix/suffix overloads (block_prefix/block_suffix).
- [x] Add block scan block-aggregate overloads (multi-output).
- [x] Add block merge/radix sort key/value + valid-items/oob_default/decomposer overloads.
- [x] Add block radix rank exclusive_digit_prefix output overloads.
- [ ] Enable BlockRadixSort decomposer for user-defined types (blocked: CUB expects tuple-of-references; need C++ adapter or alternate lowering).
- [x] Add warp merge sort key/value overloads.
- [x] Support block-aggregate scan out-params (no tuple-style multi-output return).
- [x] Expand single-phase `temp_storage=` support across all primitives and
      keep `TempStorage` getitem syntax compatible; add coverage.
- [x] Add GPU tests that use `gpu_dataclass` with multiple primitives sharing
      temp storage (load/scan/reduce/store pipelines, mixed parent/child).
- [x] Add/upgrade docstrings for every public primitive with
      `literalinclude`-based examples in `tests/coop/*_api.py`; remove any
      mention of `.create()` from public docs.
- [x] Fix coop FAQ indentation issues (Sphinx).
- [x] Add coop-local flexible data arrangement doc section and update docstring refs.
- [x] Add coop block/warp API doc stub modules and update coop_api docs.
- [ ] Improve kwarg validation and error messages for primitives with many
      overloads (match CUB API supersets; fail early with friendly errors).
- [ ] Extend ThreadData inference (alignment/shape/dtype propagation from
      inputs/outputs and `coop.(shared|local).array`) and add tests.

## Deferred / Not Planned

- BlockRadixSort decomposer support (requires a C++ tuple-of-references adapter).
- Multi-channel BlockHistogram outputs (not exposed via current CUB BlockHistogram API).
