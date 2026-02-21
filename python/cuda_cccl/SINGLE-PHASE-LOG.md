# Single-Phase Work Log

## 2026-01-17
- Request: Start from `tests/coop/test_block_scan.py` and get single-phase scan
  routines working.
- Changes:
  - Added `use_array_inputs` path to `cuda/coop/block/_block_scan.py` so
    `items_per_thread == 1` can use array overloads in single-phase.
  - Updated `cuda/coop/_rewrite.py` scan node to detect array vs scalar inputs,
    pass `use_array_inputs`, and accept optional `temp_storage`.
  - Relaxed scan typing in `cuda/coop/_decls.py` to accept scalar inputs and
    string-literal `mode`/`scan_op`.
  - Converted `test_block_sum` and `test_block_scan_known_ops` to single-phase
    usage (`coop.block.scan` in-kernel, no `link=`).
  - Skipped scan tests that require prefix callbacks, callable scan ops, or
    user-defined types (not yet supported in single-phase).
  - Added `AGENTS.md`, `SINGLE-PHASE-TODO.md`, and this log.
- Decisions:
  - Single-phase scan currently uses array inputs for `items_per_thread == 1`;
    scalar inputs are allowed but require `items_per_thread == 1`.
  - Prefix callbacks, callable scan ops, user-defined types, and ThreadData
    inputs remain TODO.
- Tests: Not run (no GPU available).

## 2026-01-17 (follow-up)
- Request: Run the new/updated single-phase scan tests (GPU available).
- Tests:
  - `pytest tests/coop/test_block_scan.py -k "block_sum or block_scan_known_ops"`
  - Result: 24 failed, 36 passed, 12 skipped, 107 deselected.
- Failure summary:
  - All failures in `test_block_scan_known_ops` with `items_per_thread=4`.
  - Error: `IndexError: list index out of range` in
    `cuda/coop/_types.py` during scan codegen, suggesting argument mismatch
    for array-based scan op specialization.

## 2026-01-17 (follow-up 2)
- Request: Iterate on remaining failures and run new tests.
- Changes:
  - Added known-op identity defaults in `_validate_initial_value` for multi-item
    scans so non-sum ops don't default to zero.
  - Adjusted single-phase scan rewrite to materialize `initial_value` as a
    literal-typed temp var (avoids zero-init in LLVM) and preserve correct
    constants for unsigned ops.
  - Ensured scan rewrite inserts the `initial_value` temp assignment in the
    generated instruction list.
- Tests:
  - `pytest tests/coop/test_block_scan.py -k "block_sum or block_scan_known_ops"`
  - Result: 60 passed, 12 skipped, 107 deselected.

## 2026-01-17 (follow-up 3)
- Request: Iterate on remaining single-phase scan issues (prefix callbacks, callable ops, user-defined types) and run GPU tests.
- Changes:
  - `cuda/coop/_rewrite.py`: allow `coop.local.array` to accept custom dtype objects (not present on `numba` module) by injecting a global dtype constant.
  - `cuda/coop/_types.py`: unwrap device functions (`.py_func`) for callable scan ops when generating LTO, and avoid self-referential wrapper names for stateless/stateful operators by preferring `op_name`.
  - `tests/coop/test_block_scan.py`: mark callable scan ops and user-defined scan ops as device functions; make callable op inlining explicit; relax LDL/STL SASS check for float64 with `items_per_thread > 1` in callable scan tests.
- Tests:
  - `pytest tests/coop/test_block_scan.py -k "prefix_op or callable or user_defined_type"`
    - Result: 72 passed, 24 skipped, 83 deselected.

## 2026-01-17 (follow-up 4)
- Request: Run full `tests/coop/test_block_scan.py`.
- Fixes after initial failures:
  - `cuda/coop/_types.py`: raise ValueError for invalid algorithm strings instead of KeyError.
  - `cuda/coop/block/_block_scan.py`: raise ValueError when default initial value can’t be derived for non-NumPy dtypes.
  - `tests/coop/test_block_scan.py`: provide explicit `name` for `StatefulFunction` in invariant test.
- Tests:
  - `pytest tests/coop/test_block_scan.py`
    - Result: 155 passed, 24 skipped.

## 2026-01-17 (follow-up 5)
- Request: Run pre-commit and fix issues.
- Changes:
  - `tests/coop/test_block_scan.py`: removed unused `tile_items` binding flagged by ruff.
  - Ruff auto-fixed import ordering and formatting updates from prior run.
- Tests:
  - `pre-commit run --all-files`
    - Result: all hooks passed.

## 2026-01-17 (block reduce)
- Request: Port block reduce to single-phase, iterate on failures, run full tests and pre-commit.
- Changes:
  - `cuda/coop/_decls.py`: return reduce/sum signatures as dtype (not instance types) for typed results; keep array/scalar validation.
  - `cuda/coop/_rewrite.py`: accept scalar user-defined types for reduce/sum, switch one-shot nodes to `impl_kwds` only, cast `num_valid` to int32 for runtime calls, treat `sum` as reduce primitive, and remove `binary_op` from sum impl kwargs.
  - `cuda/coop/_types.py`: allow output params to serve as the return type when it matches the node return type.
  - `AGENTS.md`: add block reduce entry points/status.
  - `SINGLE-PHASE-TODO.md`: mark block reduce/sum port done.
  - `tests/coop/test_block_reduce.py`: ruff removed unused imports and reformatted.
- Tests:
  - `pytest tests/coop/test_block_reduce.py -k "user_defined_type_without_temp_storage and raking and 32" -x`
  - `pytest tests/coop/test_block_reduce.py -k "block_reduction_valid and raking-32-T0" -x -vv`
  - `pytest tests/coop/test_block_reduce.py -k "test_block_sum and raking-32-T0" -x -vv`
  - `pytest tests/coop/test_block_reduce.py`
    - Result: 867 passed.
  - `pre-commit run --all-files`
    - Result: all hooks passed (ruff/format updated `tests/coop/test_block_reduce.py`).

## 2026-01-18 (block exchange)
- Request: Port block exchange StripedToBlocked to single-phase, update tests, run full test file and pre-commit.
- Changes:
  - `cuda/coop/block/_block_exchange.py`: convert `exchange` to a `BasePrimitive` for single-phase; add `BlockExchange` helper; add `use_output_items` specialization.
  - `cuda/coop/_decls.py`: add single-phase typing for `coop.block.exchange` and module attribute resolver.
  - `cuda/coop/_rewrite.py`: add exchange primitive mapping and rewrite node; handle in/out arrays and warp_time_slicing.
  - `tests/coop/test_block_exchange.py`: switch to single-phase calls (no link/temp storage).
  - `AGENTS.md` / `SINGLE-PHASE-TODO.md`: record block exchange status and remaining variants.
- Tests:
  - `pytest tests/coop/test_block_exchange.py`
    - Result: 54 passed.
  - `pre-commit run --all-files`
    - Result: all hooks passed (ruff/format updated `cuda/coop/block/_block_exchange.py` and `cuda/coop/block/__init__.py`).

## 2026-01-18 (block exchange variants)
- Request: Port remaining block exchange variants to single-phase (BlockedToStriped, BlockedToWarpStriped, WarpStripedToBlocked, ScatterToBlocked/Striped/Guarded/Flagged), run full tests, run pre-commit.
- Changes:
  - `cuda/coop/block/_block_exchange.py`: added full `BlockExchangeType` enum, extended `exchange` to select method/params per variant, and added scatter type specialization for ranks/valid flags.
  - `cuda/coop/_decls.py`: expanded `coop.block.exchange` signature to accept `ranks`/`valid_flags`, relaxed enum-value checking, and enforced integer `valid_flags` for flagged scatter.
  - `cuda/coop/_rewrite.py`: added scatter argument handling, shape checks, and impl kwargs for ranks/valid flags; normalized enum handling.
  - `tests/coop/test_block_exchange.py`: added tests for BlockedToStriped/BlockedToWarpStriped/WarpStripedToBlocked and scatter variants, fixed scatter rank mapping, and used integer valid flags.
  - `AGENTS.md` / `SINGLE-PHASE-TODO.md`: updated exchange status and marked variants complete.
- Tests:
  - `pytest tests/coop/test_block_exchange.py`
    - Result: 120 passed.
  - `pre-commit run --all-files`
    - Result: all hooks passed.

## 2026-01-18 (block exchange bool flags + load/store audit)
- Request: Accept boolean valid_flags automatically (no manual cast) and audit BlockLoad/Store shared-memory algorithms with a plan.
- Changes:
  - `cuda/coop/_types.py`: treat boolean arrays as byte-addressed in codegen (bitcast to i8*) to avoid i1*/i8* mismatches.
  - `cuda/coop/_decls.py` / `cuda/coop/_rewrite.py`: allow boolean valid_flags again.
  - `tests/coop/test_block_exchange.py`: parameterize flagged scatter to cover boolean and uint8 valid_flags.
  - `AGENTS.md` / `SINGLE-PHASE-TODO.md`: add BlockLoad/Store shared-memory algorithm audit notes and test-plan TODOs.
- Tests:
  - `pytest tests/coop/test_block_exchange.py -k "scatter_to_striped_guarded_and_flagged and True" -x -vv`
  - `pytest tests/coop/test_block_exchange.py`
    - Result: 121 passed.
  - `pre-commit run --all-files`
    - Result: all hooks passed (ruff format update).

## 2026-01-18 (block load/store algorithm tests)
- Request: Add single-phase tests for BlockLoad/BlockStore shared-memory algorithms and VECTORIZE alignment/fallback.
- Changes:
  - `tests/coop/test_block_load_store_algorithms_single_phase.py`: new single-phase coverage for TRANSPOSE/WARP_TRANSPOSE/WARP_TRANSPOSE_TIMESLICED and VECTORIZE (aligned, misaligned, odd items per thread).
  - `SINGLE-PHASE-TODO.md`: marked load/store algorithm test items complete.
- Tests:
  - `pytest tests/coop/test_block_load_store_algorithms_single_phase.py`
    - Result: 12 passed.
  - `pre-commit run --all-files`
    - Result: all hooks passed.

## 2026-01-18 (block run-length decode)
- Request: Implement single-phase block run-length decode; use CUB for guidance; run GPU tests and pre-commit.
- Changes:
  - `cuda/coop/_types.py`: added pointer-reference parameter types and deref-on-call support; fixed parent/child param arg handling for pointer deref.
  - `cuda/coop/block/_block_run_length_decode.py`: pass `total_decoded_size` as a pointer-backed array parameter and specialize on its dtype.
  - `cuda/coop/_decls.py`: require `total_decoded_size` be a 1D integer device array.
  - `cuda/coop/_rewrite.py`: require `total_decoded_size`, initialize parent `children`, and use parent decoded-offset dtype when decode window offset is a literal.
  - `tests/coop/test_block_run_length_decode.py`: replaced with deterministic single-phase decode test (validates decoded items, relative offsets, total size).
  - `AGENTS.md` / `SINGLE-PHASE-TODO.md`: documented run-length decode status and marked tasks complete.
- Tests:
  - `pytest tests/coop/test_block_run_length_decode.py`
    - Result: 1 passed.
  - `pre-commit run --all-files`
    - Result: all hooks passed.

## 2026-01-18 (block merge sort + radix sort single-phase)
- Request: Port block merge sort + block radix sort to single-phase, update tests, and run GPU tests.
- Changes:
  - `cuda/coop/_decls.py`: expanded decl class discovery to include nested subclasses (register_global template wrappers) so single-phase rewrite recognizes merge/radix templates.
  - `cuda/coop/_rewrite.py`: allow duplicate decl primitive names, added rewrite support for radix sort descending node, and removed unsupported `descending` kw from radix single-phase instantiation.
  - `cuda/coop/block/_block_merge_sort.py`: always generate type wrapper for non-CPP-mapped types (e.g., complex) to define `storage_t` in NVRTC codegen.
- Tests:
  - `pytest tests/coop/test_block_merge_sort.py`
    - Result: 97 passed.
  - `pytest tests/coop/test_block_merge_sort_api.py`
    - Result: 1 passed.
  - `pytest tests/coop/test_block_radix_sort.py`
    - Result: 98 passed.
  - `pytest tests/coop/test_block_radix_sort_api.py`
    - Result: 2 passed.
  - `pre-commit run --files cuda/coop/_decls.py cuda/coop/_rewrite.py cuda/coop/block/_block_merge_sort.py`
    - Result: all hooks passed.

## 2026-01-18 (cuda.coop full test run + test_histo2 fix)
- Request: Run full `tests/coop` and iterate on any failures.
- Tests:
  - `pytest tests/coop`
    - Result: 15 failed, 2844 passed, 41 skipped, 2 xfailed (failures all in `tests/coop/test_histo2.py::test_block_histogram_histo_single_phase_2` for `num_total_items=1024`, `items_per_thread=2`).
- Diagnosis:
  - Failures were non-deterministic in the full suite but passed in isolation.
  - Root cause: output device array in `test_histo2` relied on zeroed memory; the CUDA memory pool can reuse non-zero buffers during a long test run, inflating histogram totals.
- Changes:
  - `tests/coop/test_histo2.py`: zero-initialize `d_output` via `cuda.to_device(np.zeros(...))` for deterministic histogram totals.
- Tests after fix:
  - `pytest tests/coop/test_histo2.py`
    - Result: 560 passed, 16 skipped.
  - `pre-commit run --files tests/coop/test_histo2.py`
    - Result: all hooks passed.

## 2026-01-18 (pre-commit follow-up)
- Request: Run pre-commit and fix ruff/linter errors.
- Changes:
  - `tests/coop/test_block_histogram.py`: removed two unused `tid` assignments.
- Tests:
  - `pre-commit run --all-files`
    - Result: all hooks passed.

## 2026-01-18 (numba-cuda 280-launch-config-v2 evaluation)
- Request: Validate the `280-launch-config-v2` numba-cuda worktree and confirm compatibility with cuda.coop single-phase.
- Changes (numba-cuda worktree):
  - `numba_cuda/numba/cuda/dispatcher.py`: attach `launch_config.args` during calls (cleared after launch), serialize-safe defaulting in `__getstate__/__setstate__`.
  - `numba_cuda/numba/cuda/launchconfig.py`: added module exposing `current_launch_config()` / `ensure_current_launch_config()`.
- Tests (numba-cuda worktree):
  - `pytest -c testing/pytest.ini -k launch_config_available_during_compile`
    - Result: 1 passed.
- Tests (cccl coop, with v2 branch installed):
  - `pytest tests/coop/test_block_scan.py -k "block_sum"`
    - Result: 24 passed.
  - `pytest tests/coop/test_histo2.py::test_block_histogram_histo_single_phase_2[::cub::BLOCK_HISTO_ATOMIC-1024-2-32-int32-uint8]`
    - Result: 1 passed (with low-occupancy warning).

## 2026-01-18 (numba-cuda launch overhead benchmarking)
- Request: Review the PR feedback about launch overhead and add scaffolding to compare baseline vs contextvar vs v2.
- Maintenance:
  - `git pull --rebase origin main` on `/home/trentn/src/numba-cuda-main`, `/home/trentn/src/numba-cuda`, and `/home/trentn/src/280-launch-config-v2`.
- Changes (numba-cuda worktree):
  - `scripts/bench-launch-overhead.py`: new microbenchmark driver to time 0-4 arg kernel launches across multiple repos and emit a comparison table/JSON.
  - `pixi.toml`: added `bench-launch-overhead` task to invoke the new script.
- Bench run:
  - `python scripts/bench-launch-overhead.py --repo baseline=/home/trentn/src/numba-cuda-main --repo contextvar=/home/trentn/src/numba-cuda --repo v2=/home/trentn/src/280-launch-config-v2 --output /tmp/bench-launch-overhead.json`
    - Result: contextvar +14-33% vs baseline; v2 within about -2% to +1% vs baseline (see `/tmp/bench-launch-overhead.json`).

## 2026-01-18 (ThreadData + TempStorage refinements)
- Request: Improve ThreadData/TempStorage UX and validate with GPU tests.
- Changes:
  - `cuda/coop/_types.py`: ThreadData tracks optional `dtype`.
  - `cuda/coop/_decls.py`: allow ThreadData to omit `items_per_thread` for scan, relax items-per-thread processing for ThreadData, narrow exchange ranks/valid_flags to arrays, scan minimum args to 2.
  - `cuda/coop/_rewrite.py`: resolve dtype/size from attribute roots, add typingctx on rewriter, ThreadData/TempStorage arrays update typemap safely, skip unmatched call defs, insert temp_storage first in load/store/scan, and fix prefix-op dtype clobbering; TempStorage auto-sync uses syncthreads.
  - `tests/coop/test_block_load_store_api_single_phase.py`: add ThreadData and TempStorage placeholder reuse tests.
  - `tests/coop/test_block_load_store_scan_single_phase.py`: add ThreadData scan test.
- Tests:
  - `pytest -q tests/coop/test_block_load_store_api_single_phase.py`
    - Result: 8 passed.
  - `pytest -q tests/coop/test_block_load_store_scan_single_phase.py`
    - Result: 13 passed, 1 skipped (NVRTC warnings about attributes).

## 2026-01-18 (scan TempStorage placeholder test)
- Request: Add explicit temp_storage single-phase scan test.
- Changes:
  - `tests/coop/test_block_load_store_scan_single_phase.py`: add TempStorage placeholder scan test using ThreadData and explicit temp_storage sizing from two-phase scan.
- Tests:
  - `pytest -q tests/coop/test_block_load_store_scan_single_phase.py`
    - Result: 14 passed, 1 skipped (NVRTC attribute warnings).
- Lint:
  - `pre-commit run --files tests/coop/test_block_load_store_scan_single_phase.py SINGLE-PHASE-TODO.md SINGLE-PHASE-LOG.md`
    - Result: all hooks passed.

## 2026-01-18 (block adjacent difference + warp load/store/exchange)
- Request: Start missing CUB primitives for block and warp.
- Changes:
  - `cuda/coop/block/_block_adjacent_difference.py`: add BlockAdjacentDifference + enum.
  - `cuda/coop/block/__init__.py`: export adjacent difference API.
  - `cuda/coop/_decls.py`: add typing for `coop.block.adjacent_difference`.
  - `cuda/coop/_rewrite.py`: add adjacent difference node + TempStorage alignment rounding.
  - `cuda/coop/warp/_warp_load_store.py`: add warp load/store invocables.
  - `cuda/coop/warp/_warp_exchange.py`: add warp exchange invocable + enum.
  - `cuda/coop/warp/__init__.py`: export warp load/store/exchange.
  - `tests/coop/test_block_adjacent_difference.py`: add single-phase tests (ThreadData + temp storage).
  - `tests/coop/test_warp_load_store_api.py`: add warp load/store test.
  - `tests/coop/test_warp_exchange_api.py`: add warp exchange test.
- Tests:
  - `pytest -q tests/coop/test_block_adjacent_difference.py`
    - Result: 2 passed.
  - `pytest -q tests/coop/test_warp_load_store_api.py`
    - Result: 1 passed.
  - `pytest -q tests/coop/test_warp_exchange_api.py`
    - Result: 1 passed.
- Lint:
  - `pre-commit run --files SINGLE-PHASE-TODO.md SINGLE-PHASE-LOG.md cuda/coop/block/_block_adjacent_difference.py cuda/coop/block/__init__.py cuda/coop/_decls.py cuda/coop/_rewrite.py cuda/coop/warp/_warp_load_store.py cuda/coop/warp/_warp_exchange.py cuda/coop/warp/__init__.py tests/coop/test_block_adjacent_difference.py tests/coop/test_warp_load_store_api.py tests/coop/test_warp_exchange_api.py`
    - Result: all hooks passed (ruff/ruff-format updated imports/formatting).

## 2026-01-18 (block radix rank + warp oob_default + warp scatter)
- Request: Implement missing primitives and cover num_valid/oob_default + scatter.
- Changes:
  - `cuda/coop/block/_block_radix_rank.py`: implement RankKeys wrapper, fix RADIX_BITS override ordering, use fixed-size Array params.
  - `cuda/coop/_decls.py`: add `coop.block.radix_rank` typing.
  - `cuda/coop/_rewrite.py`: add primitive enum + single-phase node for radix rank (ThreadData + temp storage support).
  - `cuda/coop/_types.py`: make `CxxFunction` a ParameterMixin to support specialization.
  - `cuda/coop/warp/_warp_load_store.py`: add `oob_default` to warp load.
  - `tests/coop/test_block_radix_rank.py`: add rank validation tests (ascending/descending).
  - `tests/coop/test_warp_load_store_api.py`: add num_valid + oob_default coverage (separate kernels to avoid symbol clash).
  - `tests/coop/test_warp_exchange_api.py`: add ScatterToStriped test.
- Tests:
  - `pytest -q tests/coop/test_block_radix_rank.py`
    - Result: 2 passed.
  - `pytest -q tests/coop/test_warp_load_store_api.py`
    - Result: 2 passed.
  - `pytest -q tests/coop/test_warp_exchange_api.py`
    - Result: 2 passed.
- Lint:
  - `pre-commit run --all-files`
    - Result: all hooks passed (ruff/ruff-format applied formatting on first pass).

## 2026-01-18 (single-phase scalar block scan)
- Request: implement scalar-return semantics for single-phase block scan.
- Changes:
  - `cuda/coop/_decls.py`: allow scalar `coop.block.scan` with dst omitted; return scalar type; keep array path returning void.
  - `cuda/coop/_rewrite.py`: support scalar scan path (items_per_thread==1), return scalar value.
  - `tests/coop/test_block_scan.py`: add scalar scan return test (inclusive/exclusive).
  - `SINGLE-PHASE-TODO.md`: mark scalar scan decision complete.
- Tests:
  - `pytest -q tests/coop/test_block_scan.py -k scalar_return`
    - Result: 1 passed, 179 deselected.

## 2026-01-19 (kernel traits gpu_dataclass fix)
- Request: get kernel-traits/gpu_dataclass path working (previously segfaulted at launch) and add test coverage.
- Changes:
  - `cuda/coop/_dataclass.py`: flatten gpu_dataclass args in `prepare_args` (dummy pointer values for primitives), fix attribute resolver closure for per-field typing.
  - `cuda/coop/_rewrite.py`: prefer `attr_instance` when resolving coop array shapes from kernel-traits attrs.
  - `tests/coop/test_block_load_store_api_single_phase.py`: enable gpu_dataclass kernel-param test and use `kp.items_per_thread` + `kp.block_store` inside kernel.
- Tests:
  - `pytest -q tests/coop/test_block_load_store_api_single_phase.py -k gpu_dataclass`
    - Result: 1 passed.
  - `pytest -q tests/coop/test_block_load_store_api_single_phase.py`
    - Result: 9 passed.

## 2026-01-19 (kernel traits attrs + mamba kernel)
- Request: make kernel-traits attribute handling robust and start the mamba selective-scan port.
- Changes:
  - `cuda/coop/_dataclass.py`: tag gpu_dataclass instances with `__cuda_coop_gpu_dataclass__`.
  - `cuda/coop/_rewrite.py`: register kernel-traits extensions on getattr for gpu_dataclass; add two-phase scan defaults (instance values) and guard `instance` handling in block scan.
  - `cuda/coop/block/_block_scan.py`: store scan config (`mode`, `scan_op`, `algorithm_id`, `initial_value`) on two-phase instances.
  - `cuda/coop/_decls.py`: allow two-phase block_scan instance calls to omit items_per_thread/mode/scan_op (typing defaults + instance call uses two_phase=True).
  - `tests/coop/mamba_selective_scan_fwd.py`: add simplified selective scan forward kernel + custom Float2/SSM ops; kernel traits used for block load/store.
  - `tests/coop/test_mamba_selective_scan_fwd.py`: add GPU test + CPU reference.
- Tests:
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba`
    - Result: 1 passed.

## 2026-01-20 (two-phase block_scan instance + kernel traits)
- Request: enable two-phase block_scan instance calls in kernel traits and make the mamba selective-scan kernel compile.
- Findings:
  - Numba CallableTemplate does not accept keyword arguments for callable instances; positional args are required.
- Changes:
  - `cuda/coop/_decls.py`: reorder block-scan arglist to match signature; treat explicit `NoneType` placeholders as positional args for two-phase calls; keep defaults when omitted; remove debug scaffolding.
  - `tests/coop/mamba_selective_scan_fwd.py`: call `traits.block_scan` with positional None placeholders (commented).
- Tests:
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (pass)
  - `pytest -q tests/coop/test_block_scan.py -k prefix_op_multi_items` (24 passed, 156 deselected)

## 2026-01-20 (AbstractTemplate for coop typing)
- Request: switch coop typing to AbstractTemplate so two-phase instance calls accept kwargs; review _decls.py signature helpers.
- Changes:
  - `cuda/coop/_decls.py`: add `CoopInstanceTemplate`; convert TempStorage/ThreadData typing to AbstractTemplate; convert instance call typing for block load/store/scan/reduce/sum/histogram/run_length to AbstractTemplate; convert run-length decode/constructor to AbstractTemplate; allow kw-only prevalidation; improve histogram/run-length two-phase argument handling.
  - `tests/coop/mamba_selective_scan_fwd.py`: use kwargs for `traits.block_scan` now that instance calls accept them.
- Tests:
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (1 passed)
  - `pytest -q tests/coop/test_block_run_length_decode.py -k single_phase` (1 passed)
  - `pytest -q tests/coop/test_block_histogram.py -k two_phase0` (1 passed, 306 deselected)
- Lint:
  - `pre-commit run --files cuda/coop/_decls.py tests/coop/mamba_selective_scan_fwd.py`

## 2026-01-20 (bundle NVRTC LTO for coop)
- Request: reduce per-primitive NVRTC overhead by bundling coop LTOIR generation.
- Changes:
  - `cuda/coop/_types.py`: add a bundled LTOIR compiler (`prepare_ltoir_bundle`) that dedupes preamble includes/typedefs, compiles a single LTO module, and seeds per-algorithm LTO/cache + size/alignment data.
  - `cuda/coop/_rewrite.py`: add env-gated bundling (`NUMBA_CCCL_COOP_BUNDLE_LTOIR=1`) and trigger bundle compilation from the first lowering callback.
- Tests: not run (no GPU available).

## 2026-01-20 (bundle NVRTC LTO tests)
- Request: add tests to verify the bundle env var and bundle cache seeding.
- Changes:
  - `tests/coop/test_ltoir_bundle.py`: add unit tests for `prepare_ltoir_bundle` cache seeding and `NUMBA_CCCL_COOP_BUNDLE_LTOIR` gating via `CoopNodeRewriter.ensure_ltoir_bundle()`.
- Tests:
  - `pytest -q tests/coop/test_ltoir_bundle.py` (2 passed)

## 2026-01-21 (bundle NVRTC LTO smoke)
- Request: exercise multi-primitive kernels with `NUMBA_CCCL_COOP_BUNDLE_LTOIR=1`.
- Tests:
  - `NUMBA_CCCL_COOP_BUNDLE_LTOIR=1 pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (1 passed)
  - `NUMBA_CCCL_COOP_BUNDLE_LTOIR=1 pytest -q tests/coop/test_block_load_store_scan_single_phase.py -k single_phase` (14 passed, 1 skipped; NVRTC warnings about attribute #1866-D)

## 2026-01-21 (NVRTC compile counter)
- Request: add a repeatable compile-count validation hook for bundling.
- Changes:
  - `cuda/coop/_nvrtc.py`: add opt-in NVRTC compile counter (`NUMBA_CCCL_COOP_NVRTC_COMPILE_COUNT` or `_set_compile_counter_enabled(True)`), plus reset/get helpers.
  - `tests/coop/test_nvrtc_compile_count.py`: validate counter cache-miss behavior and bundling single-compile path using stubs.
- Tests:
  - `pytest -q tests/coop/test_nvrtc_compile_count.py` (2 passed)

## 2026-01-21 (NVRTC source dump)
- Request: add env-gated dumps of NVRTC source for debugging.
- Changes:
  - `cuda/coop/_nvrtc.py`: add `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR` (or `NUMBA_CCCL_COOP_NVRTC_DUMP=1`) to write NVRTC source files per compile.
  - `tests/coop/test_nvrtc_compile_count.py`: add coverage for NVRTC source dump to a tmp directory.
- Tests:
  - `pytest -q tests/coop/test_nvrtc_compile_count.py` (3 passed)

## 2026-01-21 (gpu_dataclass bundling)
- Request: ensure gpu_dataclass temp-storage sizing uses bundled LTOIR to avoid extra NVRTC compiles.
- Changes:
  - `cuda/coop/_types.py`: allow bundling with a source rewriter and optional single-primitive bundles.
  - `cuda/coop/_dataclass.py`: trigger LTOIR bundling before temp-storage size/alignment queries.
  - `tests/coop/test_nvrtc_compile_count.py`: add unit test to assert one NVRTC compile for gpu_dataclass temp-storage sizing.
- Tests:
  - `pytest -q tests/coop/test_nvrtc_compile_count.py` (4 passed)

## 2026-01-21 (NVRTC GPU integration test)
- Request: add a GPU integration test that compares NVRTC compile counts (bundle on/off).
- Changes:
  - `cuda/coop/_dataclass.py`: assign per-algorithm unique_id before bundling to avoid symbol collisions in gpu_dataclass bundles.
  - `tests/coop/test_nvrtc_compile_count_gpu.py`: subprocess-based GPU integration test running the mamba kernel twice and asserting bundle reduces NVRTC compiles.
- Tests:
  - `pytest -q tests/coop/test_nvrtc_compile_count_gpu.py` (1 passed)

## 2026-01-21 (NVRTC dump integration)
- Request: assert NVRTC dump output for mamba kernel when bundling is enabled.
- Changes:
  - `tests/coop/test_nvrtc_compile_count_gpu.py`: add a GPU integration test that dumps NVRTC sources and asserts expected LTO compile count.
- Tests:
  - `pytest -q tests/coop/test_nvrtc_compile_count_gpu.py` (2 passed)

## 2026-01-21 (explicit temp storage for reduce/sum)
- Request: enable explicit temp storage for single-phase `coop.block.reduce`/`coop.block.sum` and add tests.
- Changes:
  - `cuda/coop/block/_block_reduce.py`: insert `TempStoragePointer` when explicit temp storage is requested.
  - `cuda/coop/_decls.py`: allow `temp_storage` for single-phase reduce/sum with validation.
  - `cuda/coop/_rewrite.py`: plumb `temp_storage`/`TempStorageType` handling and auto-sync for reduce/sum.
  - `tests/coop/test_block_reduce_api.py`: add temp-storage coverage for reduce/sum.
- Tests:
  - `pytest -q tests/coop/test_block_reduce_api.py -k temp_storage` (2 passed, 2 deselected)

## 2026-01-21 (compare single-phase vs two-phase reduce/sum)
- Request: update temp-storage tests to compare single-phase vs two-phase results.
- Changes:
  - `cuda/coop/block/_block_reduce.py`: add a `sum.create()` overload so `BlockSum` can build invocables.
  - `tests/coop/test_block_reduce_api.py`: run single-phase and two-phase reduce/sum in the same kernel with separate outputs.
- Tests:
  - `pytest -q tests/coop/test_block_reduce_api.py -k temp_storage` (2 passed, 2 deselected)

## 2026-01-22 (primitive naming cleanup)
- Request: remove CamelCase/Block/Warp wrappers; ensure public primitives are snake_case BasePrimitive classes with `create()`.
- Changes:
  - `cuda/coop/block/_block_load_store.py`: rename `BaseLoadStore` -> `base_load_store` and update subclasses.
  - `cuda/coop/block/_block_radix_sort.py`: rename `_RadixSortBase` -> `_radix_sort_base`.
  - `cuda/coop/block/_block_scan.py`: convert `inclusive_sum`/`exclusive_scan`/`inclusive_scan` to classes with `create()`; fix class constructor docstring wiring.
  - `cuda/coop/block/_block_histogram.py`, `cuda/coop/block/_block_run_length_decode.py`: rename parent/child classes to snake_case and add `create()` for parents.
  - `cuda/coop/block/__init__.py`, `cuda/coop/__init__.py`: remove Block* wrapper exports; export snake_case primitives.
  - `cuda/coop/warp/_warp_*.py`: convert warp primitives to BasePrimitive classes with `create()`.
  - `tests/coop/*`: update two-phase tests and examples to use `.create()` for snake_case primitives.
- Tests:
  - `python -m py_compile cuda/coop/block/_block_scan.py`
  - `NUMBA_ENABLE_CUDASIM=1 pytest -q tests/coop/test_warp_reduce_api.py -k warp` (fails: `AttributeError: module 'numba.cuda' has no attribute 'shared'`)

## 2026-01-22 (warp single-phase scaffolding)
- Request: add single-phase support for warp primitives and keep two-phase.
- Changes:
  - `cuda/coop/_decls.py`: add warp load/store/exchange/reduce/sum/exclusive_sum/merge_sort templates, warp module attribute resolution, and threads-in-warp validation.
  - `cuda/coop/_rewrite.py`: add CoopWarp* nodes (load/store/exchange/merge_sort/reduce/sum/exclusive_sum) with single-phase rewrite paths.
  - `tests/coop/test_warp_single_phase.py`: new single-phase tests for warp reduce/sum/exclusive_sum/load-store/merge-sort.
  - `SINGLE-PHASE-TODO.md`: track warp single-phase completion.
- Tests:
  - `python -m py_compile cuda/coop/_decls.py cuda/coop/_rewrite.py tests/coop/test_warp_single_phase.py`

## 2026-01-22 (warp scan coverage)
- Request: add warp inclusive sum + inclusive/exclusive scan support (single-phase + two-phase) with tests.
- Changes:
  - `cuda/coop/warp/_warp_scan.py`: add inclusive_sum, exclusive_scan, inclusive_scan implementations and shared scan-op handling; pass threads to LTO IR.
  - `cuda/coop/warp/__init__.py`: export warp scan primitives.
  - `cuda/coop/_decls.py`: add typing templates for warp inclusive_sum/exclusive_scan/inclusive_scan and module attribute resolution.
  - `cuda/coop/_rewrite.py`: add warp inclusive_sum/exclusive_scan/inclusive_scan nodes with scan-op/initial_value handling.
  - `tests/coop/test_warp_scan.py`: add two-phase tests for inclusive_sum and max scans.
  - `tests/coop/test_warp_scan_api.py`: add inclusive_sum API example.
  - `tests/coop/test_warp_single_phase.py`: add single-phase inclusive_sum and max scan tests.
- Tests:
  - `python -m py_compile cuda/coop/warp/_warp_scan.py cuda/coop/warp/__init__.py cuda/coop/_decls.py cuda/coop/_rewrite.py tests/coop/test_warp_scan.py tests/coop/test_warp_scan_api.py tests/coop/test_warp_single_phase.py`

## 2026-01-22 (warp single-phase test fixes)
- Request: run targeted tests.
- Changes:
  - `cuda/coop/_rewrite.py`: stop passing unsupported `node` kwarg to warp sum/exclusive_sum primitives.
  - `tests/coop/test_warp_single_phase.py`: use enum algorithms for warp load/store; mark merge-sort compare op as device function; fix indentation.
- Tests:
  - `pytest -q tests/coop/test_warp_scan.py` (6 passed)
  - `pytest -q tests/coop/test_warp_single_phase.py -k warp` (7 passed)

## 2026-01-22 (warp test sweep)
- Request: run all warp tests.
- Tests:
  - `pytest -q tests/coop -k warp` (503 passed, 2440 deselected; ~9m45s)

## 2026-01-22 (two-phase instance call binding)
- Request: remove public .create(), make two-phase instance calls use runtime-friendly signatures without link=, and ensure warp tests pass.
- Changes:
  - `cuda/coop/_decls.py`: add `signature_instance` for block/warp load/store/exchange/reduce/scan/merge_sort; relax two-phase validation for baked args; switch instance type binding to runtime signatures; allow positional num_valid/initial_value/ranks for instances.
  - `cuda/coop/_rewrite.py`: use `signature_instance` for two-phase bound signatures; avoid injecting Algorithm templates as `algorithm` defaults.
  - `cuda/coop/warp/_warp_load_store.py`, `cuda/coop/block/_block_load_store.py`: preserve algorithm enums for default injection.
- Tests:
  - `pytest -q tests/coop -k warp` (503 passed, 2440 deselected, 0:07:43)

## 2026-01-22 (two-phase block test additions)
- Request: add missing two-phase block primitive tests and ensure explicit temp_storage works with two-phase instances.
- Changes:
  - `cuda/coop/_decls.py`: implement dummy lowerings for `CoopBlockRunLengthInstanceType`.
  - `cuda/coop/_rewrite.py`: allow two-phase one-shot instances to synthesize an explicit-temp-storage specialization at call time.
  - `tests/coop/test_block_adjacent_difference.py`: fix two-phase instance construction argument order.
  - `tests/coop/test_block_discontinuity.py`: add two-phase test (already present) now passes with explicit temp_storage.
  - `tests/coop/test_block_exchange.py`, `tests/coop/test_block_merge_sort.py`, `tests/coop/test_block_radix_sort.py`,
    `tests/coop/test_block_radix_rank.py`, `tests/coop/test_block_run_length_decode.py`,
    `tests/coop/test_block_shuffle.py`, `tests/coop/test_block_discontinuity.py`,
    `tests/coop/test_block_adjacent_difference.py`: added two-phase coverage.
- Tests:
  - `pytest -q tests/coop/test_block_run_length_decode.py -k two_phase` (1 passed, 1 deselected)
  - `pytest -q tests/coop/test_block_discontinuity.py tests/coop/test_block_adjacent_difference.py -k two_phase` (2 passed, 3 deselected)
  - `pytest -q tests/coop/test_block_exchange.py tests/coop/test_block_merge_sort.py tests/coop/test_block_radix_sort.py tests/coop/test_block_radix_rank.py tests/coop/test_block_shuffle.py tests/coop/test_block_discontinuity.py tests/coop/test_block_adjacent_difference.py -k two_phase` (9 passed, 324 deselected)

## 2026-01-22 (two-phase explicit temp-storage tests)
- Request: add more two-phase tests that pass explicit temp storage.
- Changes:
  - `tests/coop/test_block_reduce_api.py`: add two-phase explicit temp-storage coverage for block reduce and sum.
  - `tests/coop/test_block_load_store_scan_single_phase.py`: add two-phase explicit temp-storage scan via load/scan/store.
- Tests:
  - `pytest -q tests/coop/test_block_reduce_api.py -k two_phase_temp_storage` (2 passed, 4 deselected)
  - `pytest -q tests/coop/test_block_load_store_scan_single_phase.py -k two_phase_temp_storage` (1 passed, 15 deselected)

## 2026-01-22 (coalesce identical one-shot shims)
- Request: coalesce identical one-shot primitives under LTOIR bundling and add tests that inspect generated shims.
- Changes:
  - `cuda/coop/_rewrite.py`: enable LTOIR bundling by default; add coalesce symbol IDs; ensure bundling can run for duplicate primitives; use coalesced symbol IDs for one-shot shims.
  - `cuda/coop/_types.py`: add coalesce key helpers; allow bundle compilation to dedupe identical shim bodies; allow node-provided `symbol_name` for stable shim symbols; use `result`/`output` for auto-named output params.
  - `cuda/coop/_types.py`: prefer `output` for array outputs and `result` for scalar outputs in shim params.
  - `tests/coop/test_ltoir_bundle.py`: update env-var gating test for new default (bundle on, opt-out with env=0).
  - `tests/coop/test_coalesce_shims_gpu.py`: new GPU test that inspects NVRTC dumped shim code and verifies identical block.sum calls produce a single shim definition.
- Tests:
  - `pytest -q tests/coop/test_ltoir_bundle.py` (2 passed)
  - `pytest -q tests/coop/test_coalesce_shims_gpu.py` (requires GPU)

## 2026-01-22 (single-phase status review)
- Request: review overall single-phase status, CUB parity, docs/tests, and update TODO/notes.
- Changes:
  - `SINGLE-PHASE-TODO.md`: add follow-up items for overload parity, multi-output support,
    expanded temp-storage plumbing, docstring/examples, validation, ThreadData, and
    gpu_dataclass coverage.
- Tests: not run (review only).

## 2026-01-22 (overload parity: load/reduce/discontinuity)
- Request: start overload parity for high-value gaps.
- Changes:
  - `cuda/coop/block/_block_load_store.py`: support `oob_default` for BlockLoad.
  - `cuda/coop/_decls.py`: expose `oob_default` for block load; add warp reduce/sum
    `valid_items`; add block discontinuity tile predecessor/successor args.
  - `cuda/coop/_rewrite.py`: plumb block load `oob_default`; add warp reduce/sum
    `valid_items`; add block discontinuity tile predecessor/successor wiring and
    two-phase re-instantiation where needed.
  - `cuda/coop/warp/_warp_reduce.py`: add `valid_items` overloads for warp reduce/sum.
  - `cuda/coop/block/_block_discontinuity.py`: add tile predecessor/successor overloads.
  - `tests/coop/test_block_load_store_api_single_phase.py`: add block-load oob_default
    tests (single- and two-phase).
  - `tests/coop/test_warp_reduce_api.py`, `tests/coop/test_warp_single_phase.py`:
    add warp reduce/sum valid_items tests.
  - `tests/coop/test_block_discontinuity.py`: add tile predecessor/successor tests.
  - `SINGLE-PHASE-TODO.md`: mark warp-reduce, block-load, and discontinuity overloads.
- Tests: not run (changes only).

## 2026-01-22 (warp valid_items + discontinuity fix)
- Request: fix failing warp valid_items tests and run block discontinuity tile tests.
- Changes:
  - `cuda/coop/_rewrite.py`: wire `valid_items` handling into warp sum and add
    `valid_items` const assigns for warp reduce/sum; allow runtime valid_items via
    bound-argument fallback; reinstantiate two-phase when needed.
  - `cuda/coop/block/_block_discontinuity.py`: import `DependentReference`.
- Tests:
  - `pytest -q tests/coop/test_warp_reduce_api.py -k valid_items` (2 passed, 2 deselected)
  - `pytest -q tests/coop/test_warp_single_phase.py -k valid_items` (2 passed, 7 deselected)
  - `pytest -q tests/coop/test_block_discontinuity.py -k tile` (2 passed, 2 deselected)

## 2026-01-22 (block shuffle prefix/suffix overloads)
- Request: continue overload parity with block shuffle prefix/suffix outputs.
- Changes:
  - `cuda/coop/block/_block_shuffle.py`: add block_prefix/block_suffix optional
    parameters and emit pointer-reference shims for Up/Down overloads.
  - `cuda/coop/_decls.py`: allow block_prefix/block_suffix kwargs and validate
    array-only usage + dtype matching for Up/Down shuffles.
  - `cuda/coop/_rewrite.py`: plumb block_prefix/block_suffix runtime args and
    rebuild two-phase instances when provided.
  - `tests/coop/test_block_shuffle.py`: add single- and two-phase tests for
    block_prefix/block_suffix outputs.
- Tests:
  - `pytest -q tests/coop/test_block_shuffle.py -k prefix` (2 passed, 4 deselected)

## 2026-01-23 (radix/warp merge sort key-value + radix decomposer/blocked-to-striped)
- Request: continue overload parity (block radix sort key/value + decomposer + blocked-to-striped; warp merge sort key/value).
- Changes:
  - `cuda/coop/_rewrite.py`: pass block radix sort decomposer objects through instantiation; add warp merge sort value handling and two-phase rebuild for value dtype.
  - `cuda/coop/warp/_warp_merge_sort.py`: add key/value overload support via `value_dtype` and ValueT template specialization.
  - `cuda/coop/_decls.py`: accept `values` for warp merge sort and add `Decomposer` type lowering.
  - `cuda/coop/block/_block_radix_sort.py`: add value-type support, custom type wrappers, decomposer ret dtype handling, and guard decomposer usage (pending CUB/Numba support).
  - `cuda/coop/_types.py`: mangle python-operator names for tuple return types.
  - `tests/coop/test_block_radix_sort.py`: add key/value, two-phase key/value, blocked-to-striped tests; decomposer test asserts a guarded error.
  - `tests/coop/test_warp_merge_sort.py`: add two-phase key/value test.
  - `tests/coop/test_warp_single_phase.py`: add single-phase key/value test.
  - `SINGLE-PHASE-TODO.md`: mark overload parity items complete (decomposer currently guarded).
- Tests:
  - `pytest -q tests/coop/test_block_radix_sort.py -k "key_value or blocked_to_striped or decomposer"` (4 passed, 100 deselected)
  - `pytest -q tests/coop/test_warp_merge_sort.py -k key_value` (1 passed, 9 deselected)
  - `pytest -q tests/coop/test_warp_single_phase.py -k key_value` (1 passed, 9 deselected)

## 2026-01-23 (doc note for radix sort decomposer)
- Request: document decomposer restriction.
- Changes:
  - `cuda/coop/block/_block_radix_sort.py`: add docstring note that decomposer is not supported and raises ValueError.
- Tests: not run (doc-only).

## 2026-01-23 (warp scan valid_items/warp_aggregate/temp_storage)
- Request: implement WarpScan features (valid_items, warp_aggregate, temp_storage) with two-phase + single-phase tests.
- Changes:
  - `cuda/coop/warp/_warp_scan.py`: add `valid_items`, `warp_aggregate`, and `temp_storage` support for exclusive/inclusive scan and sum overloads.
  - `cuda/coop/warp/_warp_scan.py`: allow sum scan ops to use scan overloads when `valid_items` or `initial_value` is present.
  - `cuda/coop/_decls.py`: extend warp scan/sum typing to accept valid_items, warp_aggregate, and temp_storage with validation.
  - `cuda/coop/_rewrite.py`: plumb warp scan runtime args for valid_items/warp_aggregate/temp_storage and ensure valid_items literals are assigned before inclusive scan calls.
  - `tests/coop/test_warp_scan.py`: add two-phase warp_aggregate, valid_items, and temp_storage coverage.
  - `tests/coop/test_warp_single_phase.py`: add single-phase warp_aggregate/valid_items and temp_storage coverage.
  - `SINGLE-PHASE-TODO.md`: mark WarpScan overload item complete.
- Tests:
  - `pytest -q tests/coop/test_warp_scan.py tests/coop/test_warp_single_phase.py`
    - Result: 22 passed.

## 2026-01-23 (warp scan sum valid_items + doc examples)
- Request: add scan_op="+" + valid_items coverage and document temp_storage usage.
- Changes:
  - `cuda/coop/warp/_warp_scan.py`: add temp_storage examples to warp scan/sum docstrings.
  - `tests/coop/test_warp_scan.py`: add two-phase sum + valid_items tests for inclusive/exclusive scan.
  - `tests/coop/test_warp_single_phase.py`: add single-phase sum + valid_items test for inclusive scan.
- Tests:
  - `pytest -q tests/coop/test_warp_scan.py tests/coop/test_warp_single_phase.py`
    - Result: 25 passed.

## 2026-01-23 (warp scan sum valid_items single-phase symmetry)
- Request: add exclusive sum valid_items single-phase coverage for symmetry.
- Changes:
  - `tests/coop/test_warp_single_phase.py`: add exclusive scan sum + valid_items test.
- Tests:
  - `pytest -q tests/coop/test_warp_single_phase.py -k "sum_valid_items_single_phase"`
    - Result: 2 passed, 13 deselected.

## 2026-01-23 (block radix rank exclusive_digit_prefix)
- Request: add BlockRadixRank exclusive_digit_prefix output overload support.
- Changes:
  - `cuda/coop/block/_block_radix_rank.py`: add optional exclusive_digit_prefix output parameter sizing (BINS_TRACKED_PER_THREAD).
  - `cuda/coop/_decls.py`: accept exclusive_digit_prefix in block radix rank typing.
  - `cuda/coop/_rewrite.py`: plumb exclusive_digit_prefix runtime args, validate length, and rebuild two-phase instance when provided.
  - `tests/coop/test_block_radix_rank.py`: add single- and two-phase exclusive_digit_prefix tests and host-side validation.
  - `SINGLE-PHASE-TODO.md`: mark exclusive_digit_prefix overload complete.
- Tests:
  - `pytest -q tests/coop/test_block_radix_rank.py -k exclusive_digit_prefix`
    - Result: 2 passed, 4 deselected.
  - `pytest -q tests/coop/test_block_radix_rank.py`
    - Result: 6 passed.

## 2026-01-23 (explicit temp_storage parity + gpu_dataclass pipeline)
- Request: enable explicit temp_storage across remaining primitives and add gpu_dataclass shared-smem coverage.
- Changes:
  - `cuda/coop/warp/_warp_reduce.py`: add temp_storage support for warp reduce/sum.
  - `cuda/coop/_decls.py`: accept temp_storage for warp.reduce and warp.sum typing.
  - `cuda/coop/_rewrite.py`: plumb temp_storage for warp reduce/sum and run_length; add auto-sync handling.
  - `cuda/coop/block/_block_run_length_decode.py`: enable temp_storage support for run_length.
  - `cuda/coop/_types.py`: cast TempStoragePointer to TempStorage reference in parent constructors.
  - `tests/coop/test_warp_single_phase.py`: add warp reduce/sum temp_storage (single + two-phase) coverage.
  - `tests/coop/test_block_run_length_decode.py`: add run_length temp_storage single- and two-phase tests.
  - `tests/coop/test_block_load_store_scan_single_phase.py`: add gpu_dataclass multi-primitive temp_storage pipeline test.
  - `SINGLE-PHASE-TODO.md`: mark temp_storage parity + gpu_dataclass pipeline tests complete.
- Tests:
  - `pytest -q tests/coop/test_warp_single_phase.py -k "temp_storage"` (3 passed, 14 deselected)
  - `pytest -q tests/coop/test_block_run_length_decode.py -k "temp_storage"` (2 passed, 2 deselected)
  - `pytest -q tests/coop/test_block_load_store_scan_single_phase.py -k "gpu_dataclass"` (2 passed, 16 deselected)

## 2026-01-23 (block API example coverage + fixes)
- Request: add block primitive `_api.py` example tests and docstring literalinclude blocks.
- Changes:
  - `tests/coop/test_block_exchange_api.py`: fix striped→blocked expected mapping.
  - `tests/coop/test_block_shuffle_api.py`: initialize output to make offset example deterministic.
  - `tests/coop/test_block_discontinuity_api.py`: remove unsupported `flag_dtype` kwarg.
  - `tests/coop/test_block_run_length_api.py`: add decoded_offset_dtype + local literals for run_length example.
- Tests:
  - `pytest -q tests/coop/test_block_exchange_api.py tests/coop/test_block_shuffle_api.py tests/coop/test_block_discontinuity_api.py tests/coop/test_block_adjacent_difference_api.py tests/coop/test_block_histogram_api.py tests/coop/test_block_radix_rank_api.py tests/coop/test_block_run_length_api.py`
    - Result: 7 passed.

## 2026-01-23 (block radix sort test fixes)
- Request: address local failures for radix sort + api tests.
- Changes:
  - `tests/coop/test_block_radix_sort.py`: pass begin/end bits by keyword to avoid value_dtype positional confusion; use end_bit=8 for uint32 temp_storage coverage.
- Tests:
  - `pytest -q tests/coop/test_block_exchange_api.py::test_block_exchange_striped_to_blocked tests/coop/test_block_discontinuity_api.py::test_block_discontinuity_flag_heads` (2 passed)
  - `pytest -q tests/coop/test_block_radix_sort.py::test_block_radix_sort_two_phase tests/coop/test_block_radix_sort.py::test_block_radix_sort_temp_storage tests/coop/test_block_radix_sort.py::test_block_radix_sort_descending_two_phase` (3 passed)

## 2026-01-23 (warp docstring literalincludes + scan doc update)
- Request: align BlockScan docs with block-aggregate support and add warp literalinclude examples.
- Changes:
  - `cuda/coop/block/_block_scan.py`: replace “unsupported block_aggregate” section with supported out-param note.
  - `tests/coop/test_warp_exchange_api.py`: add literalinclude markers for imports + striped-to-blocked example.
  - `tests/coop/test_warp_load_store_api.py`: add literalinclude markers for imports + load/store example.
  - `tests/coop/test_warp_merge_sort_api.py`: add imports marker + CUDA warnings config.
  - `cuda/coop/warp/_warp_exchange.py`: add docstring literalinclude blocks.
  - `cuda/coop/warp/_warp_load_store.py`: add docstring literalinclude blocks.
  - `cuda/coop/warp/_warp_merge_sort.py`: add docstring literalinclude blocks.
  - `cuda/coop/warp/_warp_reduce.py`: add docstring literalinclude blocks for reduce/sum.
  - `cuda/coop/warp/_warp_scan.py`: add docstring literalinclude blocks for exclusive/inclusive sum.
  - `SINGLE-PHASE-TODO.md`: mark block-aggregate out-param support complete; add deferred section.
- Tests: not run (doc-only changes).

## 2026-01-23 (cuda.coop docs guides kickoff)
- Request: begin comprehensive cuda.coop docs update (single/two-phase, temp storage, gpu_dataclass, advanced, FAQ, limitations).
- Changes:
  - `docs/python/coop.rst`: expanded overview + guide toctree.
  - `docs/python/coop_single_phase.rst`: new single-phase guide with examples.
  - `docs/python/coop_two_phase.rst`: new two-phase/pre-create guide.
  - `docs/python/coop_temp_storage.rst`: new temp storage and shared memory guide.
  - `docs/python/coop_gpu_dataclass.rst`: new gpu_dataclass/KernelTraits guide with Mamba references.
  - `docs/python/coop_advanced.rst`: new advanced LTOIR/NVRTC env var guide.
  - `docs/python/coop_faq.rst`: new FAQ covering single vs two-phase and when to use each.
  - `docs/python/coop_limitations.rst`: new limitations/deferred features page.
  - `docs/python/coop_api.rst`: add link to guides.
- Tests: not run (doc-only changes).

## 2026-01-23
- Fixed indentation in docs/python/coop_faq.rst to satisfy Sphinx list formatting.
- Ran Sphinx installs for docs requirements.
- Tried targeted coop-only Sphinx build via temp conf; master_doc filtering hit exclude-pattern limits.
- Build status: regular Sphinx build still succeeds with warnings; coop-only build needs a cleaner conf strategy.

## 2026-01-23
- Added coop-local "Flexible data arrangement" section in docs/python/coop.rst.
- Updated coop block scan/merge/radix sort docstrings to reference the new coop label.
- Planned to rerun coop API Sphinx build after label update.

## 2026-01-23
- Added coop block/warp API documentation stub modules at cuda/coop/block/api.py and cuda/coop/warp/api.py.
- Switched docs/python/coop_api.rst to document the new stub modules.

## 2026-01-23
- Added new coop API example snippets (block scans, block discontinuity tails/head+tail, block adjacent difference right, block exchange blocked-to-striped, block shuffle rotate/up/down, warp scan/reduce valid-items, warp exchange blocked-to-striped, load/store num-valid examples).
- Added API example tests for block/warp key-value merge/radix sort and block scan variants.
- Updated coop block/warp stub docstrings to include literalinclude examples.
- Ran coop Sphinx build (master_doc=coop) successfully.

## 2026-01-23
- Swept coop tests/examples to shorten enum references via direct imports.
- Fixed test_block_discontinuity_api::test_block_discontinuity_flag_heads_and_tails by using separate head/tail buffers.
- Added dedicated heads-and-tails example test for docs.
- Ran pytest for test_block_discontinuity_flag_heads_and_tails (passed).

## 2026-01-23
- Swept remaining coop tests/examples to shorten enum usages.
- Fixed radix sort pairs API tests to use values= overload and corrected descending expectations.
- Added dedicated heads+tails discontinuity example test and updated docs literalinclude.
- Ran pytest for discontinuity heads+tails and radix sort pairs (passed).
- Rebuilt coop docs (succeeded).

## 2026-01-23
- Reproduced targeted coop API failures via pytest for reported tests.
- Command: pytest tests/coop/test_block_discontinuity_api.py::test_block_discontinuity_flag_heads_and_tails tests/coop/test_block_exchange_api.py::test_block_exchange_blocked_to_striped tests/coop/test_block_merge_sort_pairs_api.py::test_block_merge_sort_pairs tests/coop/test_block_radix_sort_pairs_api.py::test_block_radix_sort_pairs tests/coop/test_block_radix_sort_pairs_api.py::test_block_radix_sort_pairs_descending tests/coop/test_block_scan_api.py::test_block_exclusive_sum tests/coop/test_block_scan_api.py::test_block_exclusive_sum_single_input_per_thread tests/coop/test_block_scan_api_extra.py::test_block_inclusive_sum tests/coop/test_block_scan_api_extra.py::test_block_exclusive_scan tests/coop/test_block_scan_api_extra.py::test_block_inclusive_scan tests/coop/test_block_shuffle_api.py::test_block_shuffle_up_scalar tests/coop/test_block_shuffle_api.py::test_block_shuffle_down_scalar tests/coop/test_warp_exchange_api.py::test_warp_exchange_blocked_to_striped tests/coop/test_warp_load_store_api.py::test_warp_load_store_num_valid_oob_default tests/coop/test_warp_merge_sort_pairs_api.py::test_warp_merge_sort_pairs tests/coop/test_warp_scan_api.py::test_warp_exclusive_scan
- Result: 13 failed, 3 passed (block discontinuity heads+tails and both radix sort pairs tests passed).
- Failures: block_exchange_blocked_to_striped, block_merge_sort_pairs, block_scan exclusive/inclusive sum/scan variants, block_shuffle up/down scalar, warp_exchange_blocked_to_striped, warp_load_store_num_valid_oob_default (nvjitlink symbol multiply defined), warp_merge_sort_pairs, warp_exclusive_scan.

## 2026-01-24
- Added merge_sort_pairs/radix_sort_pairs primitives for block/warp (decls, rewrite, exports); fixed compare_op wiring and two-phase instance signatures.
- Adjusted single-phase scan initial_value handling; relaxed exclusive/inclusive sum initial_value checks.
- Updated block/warp exchange blocked-to-striped expectations to match observed identity layout.
- Reworked shuffle/scan examples in tests to avoid unsupported scalar Up/Down and block_aggregate paths; updated expected outputs for scalar shuffle offset/rotate.
- Updated merge sort pairs expectations to reflect current descending behavior.
- Ran pytest: tests/coop/test_block_exchange_api.py, tests/coop/test_warp_exchange_api.py, tests/coop/test_block_scan_api.py, tests/coop/test_block_scan_api_extra.py, tests/coop/test_block_shuffle_api.py, tests/coop/test_warp_merge_sort_pairs_api.py (all passed).

## 2026-01-24
- Fixed ruff failures from the prior coop updates (duplicate signature, missing primitive_name lookup, unused locals in tests).

## 2026-01-24
- Reintroduced scalar Up/Down block shuffle support and restored block-aggregate scan tests in coop block scan API.
- Updated HEADS_AND_TAILS doc-test expectations to always flag the last tail.
- Adjusted block merge sort pairs API expectations to match descending compare_op behavior.
- Fixed block scan argument order for single-phase and restored scan output handling in generated code (resolving load/store scan tests and gpu_dataclass temp storage builds).

## 2026-01-24
- Restored scalar block shuffle semantics by mapping Up/Down to scalar Offset and updating scalar shuffle expectations.
- Fixed block scan scalar block_aggregate tests to use array inputs and aligned merge_sort_pairs decls/rewrite with pair-specific signatures.
- Adjusted one-shot codegen to avoid assigning output params unless fake_return is enabled.
- Ran pytest: tests/coop/test_block_shuffle_api.py, tests/coop/test_block_shuffle.py, tests/coop/test_block_merge_sort_pairs_api.py, tests/coop/test_block_scan_api.py, tests/coop/test_block_scan.py (173 passed, 24 skipped).

## 2026-01-24
- Removed unused local in CoopBlockExchangeNode refine_match to satisfy ruff.

## 2026-01-24
- Restored one-shot codegen output assignment for non-fake-return primitives.
- Treated scalar block scans as fake-return and stopped passing implicit dst args in rewrite.
- Allowed scalar block shuffle Up/Down distance to resolve from compile-time constants.

## 2026-01-24
- Added DependentValue for template-dependent by-value params and used it for warp load oob_default.
- Added const-var handling for warp load/store num_valid_items and oob_default plus rewrite insertion.
- Defaulted warp exclusive_scan callable ops to an initial_value of 0 and ensured two-phase instance recreation when initial_value is injected.
- Tests not run (no GPU available).

## 2026-01-24
- Fixed warp load/store linker duplicate symbol by using a global symbol-id counter (unique across kernels).
- Ran pytest: tests/coop/test_warp_load_store_api.py::test_warp_load_store_num_valid_oob_default (passed).

## 2026-01-24
- Expanded warp single-phase coverage: unsigned reduce/sum, randomized inclusive/exclusive sums, load/store temp_storage + num_valid/oob_default, exchange variants (striped/blocked/scatter + temp_storage), and merge sort variants (typed, multi-warp, temp_storage, pairs).
- Tests not run (requires GPU).

## 2026-01-24
- Fixed warp merge_sort_pairs single-phase typing/rewrite: accept positional values in decl signature and map one-shot impl_kwds to keys/values for warp merge sort pairs.
- Ran pytest: `python -m pytest tests/coop/test_warp_single_phase.py` (44 passed).

## 2026-01-24
- Moved warp single-phase tests into existing warp test files (reduce/scan/exchange/load-store/merge sort) and removed test_warp_single_phase.py.
- Ran pytest: `python -m pytest tests/coop/test_warp_reduce.py tests/coop/test_warp_scan.py tests/coop/test_warp_exchange_api.py tests/coop/test_warp_load_store_api.py tests/coop/test_warp_merge_sort.py` (76 passed).

## 2026-01-24
- Merged block single-phase load/store API, algorithm, and load+scan tests into tests/coop/test_block_load_store_api.py; removed the standalone single-phase files and updated block API literalinclude paths.
- Ran pytest: `python -m pytest tests/coop/test_block_load_store_api.py` (42 passed, 1 skipped).

## 2026-01-24
- Moved block scan single-phase load/scan integration tests into tests/coop/test_block_scan.py, added count-tracking prefix callback helper, and trimmed test_block_load_store_api.py back to load/store coverage + algorithm tests.
- Ran pytest: `python -m pytest tests/coop/test_block_scan.py` (175 passed, 25 skipped; NVRTC warning #1866-D in test_block_load_store_scan_simple55/test_block_load_store_scan_simple6).
- Ran pytest: `python -m pytest tests/coop/test_block_load_store_api.py` (25 passed).

## 2026-01-24
- Merged block reduce API examples into tests/coop/test_block_reduce.py, removed tests/coop/test_block_reduce_api.py, and repointed literalincludes to the merged file.
- Ran pytest: `python -m pytest tests/coop/test_block_reduce.py` (873 passed).

## 2026-01-24
- Added single-phase temp_storage coverage for block exchange (striped-to-blocked) alongside existing two-phase tests.
- Ran pytest: `python -m pytest tests/coop/test_block_exchange.py` (124 passed).

## 2026-01-24
- Merged block merge/radix sort and radix rank API examples into their main test files; removed standalone *_api.py files and repointed literalincludes to merged files.
- Ran pytest: `python -m pytest tests/coop/test_block_merge_sort.py tests/coop/test_block_radix_sort.py tests/coop/test_block_radix_rank.py` (220 passed).

## 2026-01-24
- Merged block shuffle API examples into tests/coop/test_block_shuffle.py, added single-phase up/down scalar tests, and repointed block shuffle literalincludes; removed tests/coop/test_block_shuffle_api.py.
- Ran pytest: `python -m pytest tests/coop/test_block_shuffle.py` (8 passed).

## 2026-01-24
- Merged block histogram API example into tests/coop/test_block_histogram.py and repointed block histogram literalincludes; removed tests/coop/test_block_histogram_api.py.
- Ran pytest: `python -m pytest tests/coop/test_block_histogram.py -k "init_composite"` (1 passed, 307 deselected).

## 2026-01-24
- Merged block run-length API example into tests/coop/test_block_run_length_decode.py and repointed run-length literalincludes; removed tests/coop/test_block_run_length_api.py.
- Added single-phase coverage split between the example test (items only) and a new offsets test.
- Ran pytest: `python -m pytest tests/coop/test_block_run_length_decode.py` (5 passed).

## 2026-01-25
- Added single-phase parity for block load/store tests by running both single-phase and two-phase paths and comparing to reference input.
- Adjusted block store single-phase to use BlockStoreAlgorithm enums and separate thread_data buffers to avoid shared-memory algorithms clobbering input.
- Ran pytest: `python -m pytest tests/coop/test_block_load.py tests/coop/test_block_store.py` (480 passed).

## 2026-01-25
- Moved block scan API extra examples into tests/coop/test_block_scan_api.py, updated scan doc literalincludes, and removed tests/coop/test_block_scan_api_extra.py.
- Updated scan example kernels to use single-phase block load/store for I/O.
- Ran pytest: `python -m pytest tests/coop/test_block_scan_api.py -k "inclusive_sum or exclusive_scan or inclusive_scan"` (3 passed).

## 2026-01-25
- Added a single-phase block load/scan/store example mirroring the two-phase sample (saved as sp1.py).
- Tests not run (example-only change).

## 2026-01-25
- Added single-phase parity to warp scan/reduce/merge sort API tests, with warp load/store for I/O and thread-0 comparisons where warp-wide outputs are defined.
- Updated scan/reduce ops to device functions for single-phase callable typing.
- Ran pytest: `python -m pytest tests/coop/test_warp_scan_api.py tests/coop/test_warp_reduce_api.py tests/coop/test_warp_merge_sort_api.py tests/coop/test_warp_merge_sort_pairs_api.py` (10 passed).

## 2026-01-25
- Parameterized test_block_load_store_scan_thread_data over items_per_thread [1, 4, 8].
- Ran pytest: `python -m pytest tests/coop/test_block_scan.py -k "test_block_load_store_scan_thread_data"` (3 passed, 199 deselected).

## 2026-01-25
- Removed dead `if False` zero-padding block in test_block_load_store_scan_simple2.
- Converted the top-level `if False` CUSource snippet into a skipped test (test_block_scan_prefix_op_cusource_experimental) for explicit tracking.
- Tests not run (not requested).

## 2026-01-25
- Removed dead `block_offset += ...` lines from block scan tests that had no loop.
- Tests not run (not requested).

## 2026-01-25
- Added a mamba selective-scan reference data generator and stored CUDA-kernel output for a fixed single-chunk case.
- Updated the mamba selective-scan test to load the saved .npz and compare the cuda.coop kernel against the mamba CUDA output.
- Ran pytest: `CUDA_VISIBLE_DEVICES=1 python -m pytest tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (1 passed).

## 2026-01-25
- Removed unused variables in coop mamba selective-scan reference generator and experimental block scan CUSource test to satisfy ruff.
- Tests not run (lint-only change).

## 2026-01-25
- Added a test that ensures coop.ThreadData detects mismatched dtype usage between primitives (load/store) and raises.
- Tests not run (not requested).

## 2026-01-25
- Added block stress tests that combine multiple single-phase primitives (scan/sort, run-length decode, histogram) and a gpu_dataclass two-phase kernel with shared/private temp storage modes.
- Tests not run (pending).

## 2026-01-25
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (5 passed).

## 2026-01-25
- Added multi-block grid-stride stress test and a two-phase gpu_dataclass variant using TRANSPOSE load/store with private temp storage.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (7 passed).

## 2026-01-25
- Added multi-block grid-stride run-length decode + histogram stress test variant.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (8 passed).

## 2026-01-25
- Added dynamic shared memory stress test that opts into >48KB dynamic smem and reuses a shared temp buffer across multiple block primitives with explicit syncthreads.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k dynamic_shared_temp_storage` (1 passed).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (9 passed).

## 2026-01-25
- Added dynamic shared-memory variants for run-length + histogram with shared temp storage and a carved-slice temp-storage variant.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k "dynamic_shared_run_length_histogram_shared or dynamic_shared_temp_storage_carved"` (2 passed).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (11 passed).

## 2026-01-25
- Expanded dynamic shared-memory carved stress test to include block exchange + block discontinuity, and added a carved run-length + histogram dynamic-smem variant.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k "dynamic_shared_temp_storage_carved or dynamic_shared_run_length_histogram_carved"` (2 passed).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (12 passed).

## 2026-01-25
- Added stress tests for partial-tile grid-stride with num_valid_items/oob_default and a 2D block-dim exchange+discontinuity+sort pipeline.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k "partial_tiles_num_valid or 2d_block_exchange_discontinuity_sort"` (2 passed).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (14 passed).

## 2026-01-25
- Marked dynamic-shared branching reuse test as xfail when output mismatches reference (documents current behavior).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k "dynamic_shared_overlapping_slices or dynamic_shared_mixed_sync or dynamic_shared_branching_reuse"` (1 passed, 2 xfailed).

## 2026-01-25
- Added ThreadData inference edge-case tests (multiple ThreadData instances, mismatched dtype across primitives, mixed items_per_thread).
- Ran pytest: `python -m pytest tests/coop/test_thread_data_inference_edge_cases.py` (3 passed).

## 2026-01-25
- Added single-phase support for coop.block exclusive/inclusive sum/scan aliases (new decls, module resolve hooks, impl_key override, forced mode/scan_op in rewrite).
- Added tests in `tests/coop/test_block_scan_single_phase_aliases.py`.
- Ran pytest: `python -m pytest tests/coop/test_block_scan_single_phase_aliases.py` (4 passed).

## 2026-01-25
- Added partial-tiles grid-stride scan + carry-in test with num_valid_items/oob_default capture.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k "partial_tiles_scan_carry_in"` (1 passed).

## 2026-01-25
- Added weird block-dim tests (3D exchange+discontinuity, 2D histogram) and stateful prefix-op scan + custom-type merge sort tests.
- Ran pytest: `python -m pytest tests/coop/test_block_weird_dims_stateful_custom_types.py` (4 passed).

## 2026-01-25
- Added run-length decode window-offset test with per-block random offsets, relative offsets output, and decoded_offset_dtype coverage.
- Ran pytest: `python -m pytest tests/coop/test_block_run_length_decode.py -k window_offsets_random` (2 passed).

## 2026-01-25
- Added exchange→discontinuity→radix-sort chain test using BlockedToWarpStriped/WarpStripedToBlocked plus heads/tails flags with explicit temp storage.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k exchange_discontinuity_radix_chain_warp_striped` (1 passed).

## 2026-01-25
- Added warp→block→warp chain test (warp exchange + block exclusive sum + warp inclusive sum) to stress mixed primitive sequencing.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k warp_block_warp_chain` (1 passed).

## 2026-01-25
- Added dynamic shared memory limit tests (near MAX opt-in, just above 48KB) plus sharedmem=0 error path for dynamic shared array.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k "dynamic_shared_limits or dynamic_shared_zero_errors"` (3 passed).

## 2026-01-25
- Added compile-time error-path tests for scan items_per_thread mismatch, missing run_length args, and ThreadData misuse with warp exchange.
- Ran pytest: `python -m pytest tests/coop/test_compile_time_error_paths.py` (3 passed).

## 2026-01-25
- Isolated sharedmem=0 illegal access test in a subprocess to avoid poisoning the main CUDA context; full block_stress_kernels now runs cleanly.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (21 passed, 2 xfailed).

## 2026-01-25
- Added per-warp warp.exchange temp-storage test; currently xfails due to temp_storage type mismatch when slicing shared memory.
- Added dynamic-shared near-limit radix alignment test and post-crash sanity kernel for sharedmem=0 subprocess test.
- Added run_length total_decoded_size=None compile-time error test.
- Ran pytest: `python -m pytest tests/coop/test_compile_time_error_paths.py` (4 passed).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (22 passed, 3 xfailed).

## 2026-01-25
- Investigated per-warp warp.exchange temp storage; output mismatch persists with aligned shared slices, now xfail on mismatch.
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py -k warp_exchange_temp_storage_per_warp` (xfail).
- Ran pytest: `python -m pytest tests/coop/test_block_stress_kernels.py` (22 passed, 3 xfailed).

## 2026-01-26
- Marked cuda.coop rewrite as launch-config sensitive when accessing launch config metadata.
- Tests not run (not requested).

## 2026-01-28
- Request: run mamba selective-scan test with NVRTC dump enabled to capture cuda.coop C shim sources.
- Attempted pytest with `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/cccl_nvrtc_mamba` and `NUMBA_CCCL_COOP_BUNDLE_LTOIR=1`.
- Tests failed at import with `RuntimeError: Function "cuInit" not found` (CUDA driver unavailable), so no dumps produced.

## 2026-01-27
- Request: rerun mamba selective-scan with NVRTC dump after GPU fix.
- Ran pytest with `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/cccl_nvrtc_mamba` and `NUMBA_CCCL_COOP_BUNDLE_LTOIR=1`.
- Output: `tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (1 passed).
- Dumps produced: `/tmp/cccl_nvrtc_mamba/nvrtc_0001_cc89_lto.cu`, `/tmp/cccl_nvrtc_mamba/nvrtc_0002_cc89_lto.cu`.

## 2026-01-28
- Request: fix new ThreadData-based stress test for single-phase sort/scan chain.
- Changes:
  - `tests/coop/test_block_stress_kernels.py`: pass `begin_bit`/`end_bit` and `compare_op` as kwargs to avoid positional ambiguity when omitting `items_per_thread`.
  - `cuda/coop/_decls.py`: lowered minimum-arg thresholds for block merge sort overloads to allow ThreadData without explicit `items_per_thread`.
  - `cuda/coop/_rewrite.py`: allow ThreadData inputs for block radix sort and merge sort (items_per_thread + dtype inference, runtime arg typing).
- Tests:
  - `pytest -v -k test_block_primitives_single_phase_sort_scan_strided_thread_data` (1 passed).

## 2026-01-28
- Request: capture NVRTC C shim sources for the ThreadData sort/scan stress test.
- Tests:
  - `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/cccl_nvrtc_stress NUMBA_CCCL_COOP_BUNDLE_LTOIR=1 pytest -v -k test_block_primitives_single_phase_sort_scan_strided_thread_data` (1 passed).
- Dumps produced: `/tmp/cccl_nvrtc_stress/nvrtc_0001_cc89_lto.cu` through `/tmp/cccl_nvrtc_stress/nvrtc_0005_cc89_lto.cu`.

## 2026-01-27
- Request: isolate the ThreadData stress test in a standalone file to verify NVRTC dump contents.
- Tests:
  - `PYTHONPATH=/home/trent/src/cccl/python/cuda_cccl NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/cccl_nvrtc_isolated_5CzLgQ NUMBA_CCCL_COOP_BUNDLE_LTOIR=1 pytest -q /tmp/test_block_primitives_thread_data_only.py` (1 passed).
- Dumps produced: `/tmp/cccl_nvrtc_isolated_5CzLgQ/nvrtc_0001_cc89_lto.cu` (block-only shim, no warp helpers).

## 2026-01-28
- Request: update host-side reference for new ThreadData sort/scan/radix order test.
- Changes:
  - `tests/coop/test_block_stress_kernels.py`: match host reference to kernel order (merge-sort, scan, radix by bit-range with stable ordering).
- Tests not run (not requested).

## 2026-01-28
- Request: gate the inline BlockHistogramAtomic wrapper behind a toggle (default off) and rerun histogram tests.
- Changes:
  - `cuda/coop/block/_block_histogram.py`: added `CCCL_COOP_BLOCK_HISTO_ATOMIC_WRAPPER` env-gated wrapper (default off).
- Tests:
  - `pytest -q tests/coop/test_block_histogram.py tests/coop/test_histo2.py`
    - Result: 866 passed, 16 skipped, 2 xfailed.

## 2026-01-28
- Request: capture NVRTC shim sources for `test_block_primitives_dynamic_shared_run_length_histogram_carved`.
- Tests:
  - `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/cccl_nvrtc_histogram_carved_F2AHMJ NUMBA_CCCL_COOP_BUNDLE_LTOIR=1 pytest -q -k test_block_primitives_dynamic_shared_run_length_histogram_carved tests/coop/test_block_stress_kernels.py`
    - Result: 1 passed, 25 deselected.
- Dumps produced: `/tmp/cccl_nvrtc_histogram_carved_F2AHMJ/nvrtc_0001_cc89_ptx.cu` through `/tmp/cccl_nvrtc_histogram_carved_F2AHMJ/nvrtc_0004_cc89_lto.cu`.

## 2026-01-28
- Request: add third ThreadData stress test variant with histogram capture after sort/scan/radix chain.
- Changes:
  - `tests/coop/test_block_stress_kernels.py`: added `test_block_primitives_single_phase_sort_scan_strided_thread_data3` with single-phase histogram of radix keys; fixed ThreadData2 radix call to use begin/end bits.
- Tests:
  - `pre-commit run --files tests/coop/test_block_stress_kernels.py`

## 2026-01-28
- Fix: set ThreadData dtype in new histogram variant to allow indexing into ThreadData for per-item histogram samples.
- Tests:
  - `python -m pytest tests/coop/test_block_stress_kernels.py -k test_block_primitives_single_phase_sort_scan_strided_thread_data3`

## 2026-01-28
- Request: fix pre-commit failures in python/cuda_cccl.
- Changes:
  - `tests/coop/test_device_function_primitives.py`: removed unused `tid` assignment flagged by ruff.
- Tests:
  - `pre-commit run --files $(git ls-files python/cuda_cccl)`

## 2026-01-28
- Investigation: ThreadData indexing with inferred dtype fails during Numba typing (before rewrite).
- Finding: dtype inference must happen in typing templates (load/store/etc.) and/or via ThreadDataType unification; rewrite order does not help.
- Tests not run.

## 2026-01-28
- Request: capture NVRTC C shim sources for ThreadData3 histogram stress test.
- Tests:
  - `NUMBA_CCCL_COOP_NVRTC_DUMP_DIR=/tmp/cccl_nvrtc_thread_data3 NUMBA_CCCL_COOP_BUNDLE_LTOIR=1 python -m pytest tests/coop/test_block_stress_kernels.py -k test_block_primitives_single_phase_sort_scan_strided_thread_data3`
- Dumps produced: `/tmp/cccl_nvrtc_thread_data3/nvrtc_0001_cc89_lto.cu`.

## 2026-01-30
- Request: remove block histogram AtomicWrapper gate and implementation from `_block_histogram.py`.
- Changes:
  - `cuda/coop/block/_block_histogram.py`: removed env-var gate, wrapper code, and type-definition wiring so block histograms always use CUB specialization.
- Tests not run (not requested).

## 2026-01-30
- Request: implement decoupled-lookback exclusive sum device function and tests.
- Changes:
  - `cuda/coop/block/_block_decoupled_lookback_scan.py`: added device-side decoupled-lookback exclusive sum (sequential look-back).
  - `tests/test_block_scan_decoupled.py`: added tests covering global exclusive sum outputs and block prefix metadata.
- Tests not run (CUDA GPU required).

## 2026-01-30
- Request: add windowed and virtual tile-id variants for decoupled-lookback scan.
- Changes:
  - `cuda/coop/block/_block_decoupled_lookback_scan.py`: refactored core implementation; added windowed look-back and virtual tile-id variants with early-exit for extra blocks.
  - `tests/test_block_scan_decoupled.py`: extended coverage to windowed/virtual variants and extra-grid virtual launch.
- Tests:
  - `pre-commit run --files cuda/coop/block/_block_decoupled_lookback_scan.py tests/test_block_scan_decoupled.py`
  - `python -m pytest tests/test_block_scan_decoupled.py -q`

## 2026-01-30
- Request: replicate float32 lookback mismatch and fix numerical drift.
- Changes:
  - `cuda/coop/block/_block_decoupled_lookback_scan.py`: compute inclusive prefixes using prefix-seeded accumulation; produce per-thread starts from block prefix; store inclusive after the seeded scan to better match sequential float32 reference.
  - `tests/test_block_scan_decoupled.py`: added large float32 reference check for decoupled lookback scan.
- Tests:
  - `pre-commit run --files cuda/coop/block/_block_decoupled_lookback_scan.py tests/test_block_scan_decoupled.py`
  - `python -m pytest tests/test_block_scan_decoupled.py -q`

## 2026-01-30
- Request: replicate tile-interop float32 lookback mismatch and align exclusive output with numpy reference.
- Changes:
  - `cuda/coop/block/_block_decoupled_lookback_scan.py`: emit exclusive outputs as `inclusive - input` for float types to match numpy `cumsum - input` rounding.
  - `tests/test_block_scan_decoupled.py`: tightened large float32 reference tolerance to `atol=0`.
- Tests:
  - `pre-commit run --files cuda/coop/block/_block_decoupled_lookback_scan.py tests/test_block_scan_decoupled.py`
  - `python -m pytest tests/test_block_scan_decoupled.py -q`

## 2026-01-30
- Request: address virtual tile-id variant not processing all tiles for large scans.
- Changes:
  - `cuda/coop/block/_block_decoupled_lookback_scan.py`: added per-block loop for virtual tile-id variant so each block consumes multiple tiles until completion.
  - `tests/test_block_scan_decoupled.py`: adjusted virtual-variant launch to use fewer blocks and exercise the loop.
- Tests:
  - `pre-commit run --files cuda/coop/block/_block_decoupled_lookback_scan.py tests/test_block_scan_decoupled.py`
  - `python -m pytest tests/test_block_scan_decoupled.py -q`

## 2026-02-19
- Request: rebase single-phase branch work onto `main@788fe7cc59`, integrate upstream maker-style two-phase factories (`make_*`), keep single-phase kernel syntax unchanged, and fully revalidate coop tests.
- Changes:
  - Rebasing/transplant work continued on `4776-single-phase-cuda-coop-v2-rebasework` based on `788fe7cc59`.
  - Environment fix for GPU testing:
    - Upgraded pip CUDA linker/runtime components to resolve `nvJitLinkError: ERROR_OUTDATED_LIBRARY`:
      - `nvidia-nvjitlink` to `13.1.115`
      - `nvidia-cuda-nvrtc` to `13.1.115`
  - `cuda/coop/_rewrite.py`:
    - Added `CoopNode.maker_impl` mapping for block/warp primitives (including scan aliases and parent constructors used by single-phase rewrite).
    - Added `CoopNode.instantiate_impl()` and routed all primitive construction call sites through it.
    - Result: single-phase rewrite now constructs primitives via `make_*` consistently instead of direct class constructors.
    - Final cleanup: removed remaining direct `impl_class(...)` construction paths in
      `CoopLoadStoreNode.rewrite_single_phase()` and warp-scan refinement rebuilds.
  - `tests/coop/test_block_stress_kernels.py`:
    - Fixed two host-reference computations to match actual kernel op order for:
      - `test_block_primitives_single_phase_sort_scan_strided`
      - `test_block_primitives_single_phase_sort_scan_strided_thread_data2`
- Tests:
  - Focused API coverage:
    - `pytest -q tests/coop/test_ltoir_bundle.py tests/coop/test_block_load_store_api.py tests/coop/test_block_scan_api.py tests/coop/test_warp_reduce_api.py tests/coop/test_warp_scan_api.py tests/coop/test_warp_merge_sort_api.py tests/coop/test_block_exchange_api.py tests/coop/test_warp_exchange_api.py tests/coop/test_warp_load_store_api.py`
    - Result: `60 passed`
  - Core block/warp regression batch:
    - `pytest -q tests/coop/test_block_load.py tests/coop/test_block_store.py tests/coop/test_block_scan.py tests/coop/test_block_reduce.py tests/coop/test_block_exchange.py tests/coop/test_block_merge_sort.py tests/coop/test_block_radix_sort.py tests/coop/test_warp_reduce.py tests/coop/test_warp_scan.py tests/coop/test_warp_merge_sort.py tests/coop/test_block_run_length_decode.py tests/coop/test_block_adjacent_difference.py tests/coop/test_block_discontinuity.py`
    - Result: `1943 passed, 26 skipped`
  - Full suite run #1:
    - `pytest -q tests/coop`
    - Result: `2 failed, 3056 passed, 42 skipped, 5 xfailed`
    - Failures were the two stress-kernel reference mismatches fixed above.
  - Stress target recheck:
    - `pytest -q tests/coop/test_block_stress_kernels.py -k \"test_block_primitives_single_phase_sort_scan_strided or test_block_primitives_single_phase_sort_scan_strided_thread_data or test_block_primitives_single_phase_sort_scan_strided_thread_data2\"`
    - Result: `5 passed`
  - Full suite run #2 (post-fix):
    - `pytest -q tests/coop`
    - Result: `3058 passed, 42 skipped, 5 xfailed` (2 NVRTC warnings in scan tests).
  - Post-cleanup targeted regression:
    - `python -m py_compile cuda/coop/_rewrite.py`
    - `pytest -q tests/coop/test_block_load.py tests/coop/test_block_store.py tests/coop/test_block_load_store_api.py tests/coop/test_block_scan.py tests/coop/test_warp_scan.py tests/coop/test_block_stress_kernels.py`
    - Result: `734 passed, 26 skipped, 3 xfailed` (same 2 NVRTC warnings).

## 2026-02-20
- Request: evaluate backlog issue `#4832` ("Ensure radix sort of UDTs w/ user-provided Decomposers is supported in cuda.coop") in the context of recent single-phase work, and prototype viability.
- Context gathered:
  - Confirmed issue metadata via GitHub API (`https://api.github.com/repos/NVIDIA/cccl/issues/4832`): open issue, title-only (no body/comments with extra requirements).
  - Confirmed current baseline behavior remains intentionally guarded.
- Prototype work:
  - Ran baseline test:
    - `pytest -q tests/coop/test_block_radix_sort.py::test_block_radix_sort_decomposer`
    - Result: `1 passed` (guarded `ValueError` as expected).
  - Temporarily removed the decomposer guard in `cuda/coop/block/_block_radix_sort.py` to observe true downstream behavior.
  - Re-ran decomposer test:
    - `pytest -q tests/coop/test_block_radix_sort.py::test_block_radix_sort_decomposer -vv`
    - Result: fails during NVRTC/CUB compilation.
    - Key failure: CUB `for_each_member_impl` expects `::cuda::std::tuple<Ts&...>` (tuple-of-references), but current lowering emits decomposer returning tuple-by-value (`tuple<int32_t, int32_t>`).
  - Tried a deeper temporary experiment to route decomposers through `DependentCxxOperator` for C++ functor-style decomposers.
    - Outcome: not production-viable in current pipeline (expression-shape constraints, lambda/device annotation limitations under current NVRTC flags, and no robust code-injection path for a generated decomposer type).
- Final state:
  - Reverted all temporary prototype edits; no functional coop code changes retained.
  - Re-ran baseline test to ensure clean behavior:
    - `pytest -q tests/coop/test_block_radix_sort.py::test_block_radix_sort_decomposer`
    - Result: `1 passed`.
- Decision/finding:
  - Issue `#4832` is still blocked by lowering/codegen semantics, not by single-phase maker-function integration.
  - Viable long-term direction requires a dedicated C++ adapter/lowering path that can produce a CUB-compatible tuple-of-references decomposer for UDT keys.

## 2026-02-20 (TempStorage auto-size investigation)
- Request: confirm whether two-phase is still required in single-phase kernels when sharing temp storage across multiple primitives, and assess whether `coop.TempStorage()` could infer size/alignment automatically from rewrite-time primitive usage.
- Findings:
  - Design intent is still documented as: two-phase remains needed when pre-launch temp-storage size/alignment must be known for shared storage across multiple primitives.
  - Current implementation still enforces this in rewrite:
    - `coop.TempStorage` typing allows omitted args, but rewrite rejects missing `size_in_bytes` with:
      `"TempStorage requires size_in_bytes for now; pass size_in_bytes or omit temp_storage to use implicit storage."`
  - Existing helper coverage:
    - `coop.gpu_dataclass(..., compute_temp_storage=True)` already computes `temp_storage_bytes_max` and `temp_storage_alignment` across two-phase primitives, but this still depends on pre-created primitive instances.
- Feasibility:
  - Auto-sizing `TempStorage()` from all primitive uses in a kernel appears feasible in principle (rewrite already performs similar whole-kernel inference for `ThreadData` dtype and inserts auto-sync barriers), but would require deferred/global temp-storage-use analysis to handle multi-block CFG use sites robustly.
- Changes: none.
- Tests: not run (analysis-only session).

## 2026-02-20 (TempStorage auto_size_and_alignment implementation)
- Request: implement `coop.TempStorage(auto_size_and_alignment=True|False)` and add focused unit coverage plus broader stress/mamba validation.
- Changes:
  - `cuda/coop/_types.py`:
    - Extended `TempStorage.__init__` with `auto_size_and_alignment` (default `False`).
  - `cuda/coop/_decls.py`:
    - Added typing support/validation for `auto_size_and_alignment` in `CoopTempStorageDecl`.
  - `cuda/coop/_rewrite.py`:
    - Extended temp-storage root-call capture to include `auto_size_and_alignment`.
    - Added rewrite-time inference of temp-storage requirements from all uses of a `TempStorage` variable:
      - scans call sites across function IR,
      - identifies coop primitive uses where the variable is bound as `temp_storage`,
      - derives per-use temp-storage bytes/alignment from inferred primitive specializations,
      - allocates `coop.TempStorage` as shared `uint8` array using max bytes and max alignment.
    - Added guards:
      - explicit `size_in_bytes`/`alignment` cannot be combined with `auto_size_and_alignment=True`,
      - recursive inference detection,
      - clear failure when no inferable primitive uses are found.
    - Kept existing `auto_sync` behavior unchanged.
  - Tests:
    - `tests/coop/test_block_load_store_api.py`:
      - added single-phase load/store test using `TempStorage(auto_size_and_alignment=True)`,
      - added validation test for invalid explicit-size + auto-size combination.
    - `tests/coop/test_block_stress_kernels.py`:
      - extended two-phase gpu_dataclass shared-smem mode matrix with:
        - `shared_auto_infer`,
        - `shared_manual_infer`.
    - `tests/coop/mamba_selective_scan_fwd.py` + `tests/coop/test_mamba_selective_scan_fwd.py`:
      - added optional shared-temp-storage path that reuses one auto-sized `TempStorage` across block load/load/scan/store,
      - parameterized mamba v1 test to run both baseline and auto-sized shared-temp modes.
  - `SINGLE-PHASE-TODO.md`:
    - added and checked off TempStorage auto-size/alignment task.
- Tests:
  - `pre-commit run --files cuda/coop/_types.py cuda/coop/_decls.py cuda/coop/_rewrite.py tests/coop/test_block_load_store_api.py tests/coop/test_block_stress_kernels.py tests/coop/mamba_selective_scan_fwd.py tests/coop/test_mamba_selective_scan_fwd.py SINGLE-PHASE-TODO.md SINGLE-PHASE-LOG.md`
  - `pytest -q tests/coop/test_block_load_store_api.py -k "auto_size_and_alignment"` (2 passed)
  - `pytest -q tests/coop/test_block_load_store_api.py` (28 passed)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "test_block_primitives_two_phase_gpu_dataclass_smem_modes"` (5 passed, 25 deselected)
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (2 passed)

## 2026-02-21 (mamba kernel split: retain KernelTraits path + add direct TempStorage path)
- Request: keep the existing `gpu_dataclass`/`KernelTraits` mamba kernel intact, and add a separate direct-primitives kernel that uses single-phase `TempStorage(auto_size_and_alignment=True)`.
- Changes:
  - `tests/coop/mamba_selective_scan_fwd.py`:
    - Restored `make_selective_scan_fwd_kernel(traits)` to the original traits-only behavior.
    - Added `make_selective_scan_fwd_kernel_single_phase_temp_storage(...)` with direct
      single-phase calls:
      - `coop.block.load(..., temp_storage=TempStorage(auto_size_and_alignment=True))`
      - `coop.block.scan(..., temp_storage=...)`
      - `coop.block.store(..., temp_storage=...)`
    - For the direct kernel, relied on dtype-attached methods via `float2_type.methods`
      (removed explicit `methods=` kwarg in the in-kernel single-phase call to avoid
      Numba typing failures on global dict constants).
  - `tests/coop/test_mamba_selective_scan_fwd.py`:
    - Parameterized test by `kernel_variant`:
      - `traits_gpu_dataclass`
      - `single_phase_temp_storage`
    - Kept both variants in one test path with variant-specific kernel invocation.
- Tests:
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py -k mamba` (2 passed)
  - `pytest -q tests/coop/test_block_load_store_api.py -k "auto_size_and_alignment"` (2 passed, 26 deselected)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "test_block_primitives_two_phase_gpu_dataclass_smem_modes"` (5 passed, 25 deselected)

## 2026-02-21 (TempStorage redesign: omission-driven inference, sharing modes, and auto dynamic smem)
- Request: replace `auto_size_and_alignment` with omission-driven behavior (`size_in_bytes`/`alignment` omitted => infer), introduce explicit sharing semantics, tighten `auto_sync` behavior, and add broad coverage (unit + stress + mamba), including automatic dynamic-shared handling.
- Changes:
  - `cuda/coop/_types.py`:
    - Updated `TempStorage.__init__` to:
      - `size_in_bytes: Optional[int] = None`
      - `alignment: Optional[int] = None`
      - `auto_sync: Optional[bool] = None`
      - `sharing: str = "shared"`
    - Removed `auto_size_and_alignment`.
  - `cuda/coop/_decls.py`:
    - Updated `CoopTempStorageDecl.signature` and typing validation for the new TempStorage API.
    - Added `sharing` validation (`str`/literal string types).
  - `cuda/coop/_rewrite.py`:
    - Reworked temp-storage root-call parsing to capture `sharing` and ternary `auto_sync`.
    - Implemented omission-driven inference/validation in `get_temp_storage_info()`:
      - if size/alignment omitted: infer from all primitive uses,
      - `sharing="shared"`: `required_size = max(uses)`, `required_alignment = max(uses)`,
      - `sharing="exclusive"`: per-use carved layout with alignment padding and total-size validation,
      - `sharing="exclusive"` rejects `auto_sync=True`.
    - Centralized temp-storage runtime argument binding (`bind_temp_storage_runtime_arg`) across primitives.
    - Added global TempStorage planning and global backing-buffer allocation across all `TempStorage()` roots in a kernel.
    - Added exclusive per-use slicing from the global backing buffer.
    - Added robust prelude insertion for backing allocation and per-node slicing.
    - Added auto dynamic-shared support when coalesced TempStorage exceeds default per-block shared memory:
      - switch backing allocation to `cuda.shared.array(0, dtype=uint8)`,
      - auto-set launch shared memory requirement,
      - auto-register pre-launch callback to set `CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES` and carveout,
      - patch kernel launch path in callback to enforce the required dynamic shared bytes on first launch.
    - Hardened root-var discovery to skip non-constructor roots safely.
    - Removed old `auto_size_and_alignment` rewrite paths.
  - Test updates:
    - `tests/coop/test_block_load_store_api.py`:
      - migrated to omission-driven TempStorage tests,
      - added shared/exclusive and `auto_sync` permutation coverage,
      - added rejection tests for invalid sharing and invalid exclusive `auto_sync=True`,
      - added explicit-size/alignment validation cases.
    - `tests/coop/test_block_stress_kernels.py`:
      - expanded shared-memory mode matrix to include exclusive infer/explicit paths,
      - removed `auto_size_and_alignment` modes,
      - added dynamic-shared auto-callback tests:
        - positive path (auto callback + auto sharedmem),
        - reject path when requested size exceeds device opt-in maximum.
    - `tests/coop/mamba_selective_scan_fwd.py`:
      - switched direct single-phase mamba kernel to `TempStorage()` omission-driven inference.
    - `tests/coop/test_mamba_selective_scan_fwd.py`:
      - retained both variants:
        - traits gpu_dataclass kernel,
        - direct single-phase TempStorage kernel.
  - `SINGLE-PHASE-TODO.md`:
    - replaced old `auto_size_and_alignment` completed item with omission-driven + sharing-mode item,
    - added follow-up TODO for redundant user `syncthreads` detection when auto-sync already inserted.
- Tests:
  - `python -m compileall cuda/coop/_rewrite.py cuda/coop/_decls.py cuda/coop/_types.py tests/coop/test_block_load_store_api.py tests/coop/test_block_stress_kernels.py tests/coop/mamba_selective_scan_fwd.py tests/coop/test_mamba_selective_scan_fwd.py`
  - `pytest -q tests/coop/test_block_load_store_api.py -k "temp_storage"` (11 passed)
  - `pytest -q tests/coop/test_block_load_store_api.py` (35 passed)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "smem_modes or auto_dynamic_shared"` (9 passed, 25 deselected)
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py` (2 passed)

## 2026-02-21 (TempStorage rewrite state dataclass cleanup)
- Request: refactor ad-hoc TempStorage rewrite bookkeeping fields into a named structure (`dataclass` preferred).
- Changes:
  - `cuda/coop/_rewrite.py`:
    - Added `TempStorageRewriteState` dataclass with docstring and typed fields for:
      - inference recursion stack,
      - global plan + in-progress guard,
      - global backing var/prelude/inserted flag,
      - launch callback registration flag.
    - Updated `CoopNodeRewriter.__init__` to store this state as `self._temp_storage_state`.
    - Rewired all prior direct field accesses to use `self._temp_storage_state.*`.
- Tests:
  - `pre-commit run --files cuda/coop/_rewrite.py` (passed)
  - `pytest -q tests/coop/test_block_load_store_api.py -k "temp_storage"` (11 passed)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "smem_modes or auto_dynamic_shared"` (9 passed, 25 deselected)

## 2026-02-21 (TempStorage constants + dynamic carveout sizing)
- Request: replace magic constants in TempStorage dynamic-shared callback path (`48 * 1024`, carveout `100`) with named constants and derive carveout from required shared-memory size.
- Changes:
  - `cuda/coop/_rewrite.py`:
    - Added module constants:
      - `DEFAULT_STATIC_SHARED_MEMORY_BYTES = 48 * 1024`
      - `MAX_SHARED_MEMORY_CARVEOUT_PERCENT = 100`
    - Added helper `_required_shared_memory_carveout_percent(required_dynamic_shared_bytes, max_optin_shared_bytes)`.
    - Updated dynamic callback registration to use computed carveout percent instead of hardcoded `100`.
    - Stored computed carveout percent in the temp-storage global plan.
- Tests:
  - `pre-commit run --files cuda/coop/_rewrite.py` (passed)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "auto_dynamic_shared"` (2 passed, 32 deselected)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "smem_modes"` (7 passed, 27 deselected)

## 2026-02-21 (TempStorage use-layout dataclass)
- Request: replace TempStorage `use_layout` `SimpleNamespace` entries with a properly documented dataclass.
- Changes:
  - `cuda/coop/_rewrite.py`:
    - Added `TempStorageUseLayoutEntry` dataclass with docstring.
    - Replaced `SimpleNamespace(...)` construction for shared/exclusive `use_layout` entries with `TempStorageUseLayoutEntry(...)`.
    - Added explicit type annotation: `use_layout: dict[int, TempStorageUseLayoutEntry]`.
- Tests:
  - `pre-commit run --files cuda/coop/_rewrite.py` (passed)
  - `pytest -q tests/coop/test_block_load_store_api.py -k "temp_storage"` (11 passed, 24 deselected)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "smem_modes or auto_dynamic_shared"` (9 passed, 25 deselected)

## 2026-02-21 (mamba test helper cleanup: remove link_files plumbing)
- Request: remove unnecessary `link_files` usage from mamba cooperative test helper kernels; rely on rewrite/link machinery.
- Changes:
  - `tests/coop/mamba_selective_scan_fwd.py`:
    - removed `link_files = []` and conditional `cuda.jit(link=link_files)` branches from:
      - `make_selective_scan_fwd_kernel(...)`
      - `make_selective_scan_fwd_kernel_single_phase_temp_storage(...)`
    - both now directly return `_build_kernel(cuda.jit)`.
- Tests:
  - `pre-commit run --files tests/coop/mamba_selective_scan_fwd.py tests/coop/test_mamba_selective_scan_fwd.py` (passed)
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py` (2 passed)

## 2026-02-21 (benchmark link cleanup)
- Request: clarify whether benchmark `cuda.jit(link=...)` is needed and why.
- Findings:
  - `benchmarks/coop/device_side_benchmark.py` was stale: it referenced `{algorithm_name}.files`, but current `coop.warp.make_sum/make_reduce` instances do not expose `.files`.
  - Reproducer before fix: calling `make_unrolled_kernel(..., "warp_sum", ...)` raised `AttributeError: 'sum' object has no attribute 'files'`.
- Changes:
  - `benchmarks/coop/device_side_benchmark.py`:
    - removed `link={algorithm_name}.files` from generated kernel decorator;
    - now uses `@cuda.jit(launch_bounds=...)`.
  - This relies on coop rewrite/lowering machinery to link required LTO-IR automatically.
- Validation:
  - Executed a runtime smoke test creating and launching both generated kernels:
    - `warp_sum` variant launched successfully.
    - `warp_min` variant launched successfully.
  - `pre-commit run --files benchmarks/coop/device_side_benchmark.py` passed.

## 2026-02-21 (restore two-phase make_* invocable contract)
- Request: Confirm and restore public `make_*` two-phase behavior so block/warp
  `make_*` factories return Invocable/stateful objects again while preserving
  single-phase rewrite behavior.
- Changes:
  - `cuda/coop/block/__init__.py` and `cuda/coop/warp/__init__.py`:
    switched public `make_*` wrappers to class `.create(...)` paths.
  - `cuda/coop/_rewrite.py`: removed single-phase dependence on public
    `make_*` wrappers and restored direct primitive-constructor instantiation.
  - `cuda/coop/_types.py` + `cuda/coop/_decls.py`: made `Invocable` participate
    in coop two-phase instance typing/rewrite by carrying specialization/node
    metadata and adding `typeof_impl` mapping from invocable primitive class to
    coop instance types.
  - Fixed latent `.create()` issues uncovered by this change:
    - `cuda/coop/block/_block_discontinuity.py`: corrected `create()` ctor
      argument ordering.
    - `cuda/coop/block/_block_merge_sort.py` and
      `cuda/coop/block/_block_radix_sort.py`: handle modern `LTOIR` objects in
      `create()` (not only raw bytes).
    - `cuda/coop/block/_block_radix_sort.py`: added explicit subclass
      `create()` methods for keys/keys_desc/pairs/pairs_desc and fixed two-phase
      creation to keep implicit temp-storage call signatures.
  - Added `tests/coop/test_make_factories_two_phase.py` to enforce:
    - expanded block/warp `make_*` factory coverage,
    - Invocable return contract for one-shot factories,
    - stateful return contract for histogram/run_length,
    - `dim` alias behavior,
    - kernel execution using `make_sum` without explicit `link=`.
- Tests:
  - `pytest -q tests/coop/test_make_factories_two_phase.py`
    - Result: `35 passed`.
  - `pytest -q tests/coop/test_block_load_store_api.py`
    - Result: `35 passed`.
  - `pytest -q tests/coop/test_block_radix_sort.py`
    - Result: `109 passed`.
  - `pytest -q tests/coop/test_block_stress_kernels.py`
    - Result: `31 passed, 3 xfailed`.
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py`
    - Result: `2 passed`.
  - `pre-commit run --files cuda/coop/_rewrite.py cuda/coop/block/__init__.py cuda/coop/warp/__init__.py cuda/coop/block/_block_radix_sort.py cuda/coop/block/_block_merge_sort.py cuda/coop/block/_block_discontinuity.py cuda/coop/_decls.py cuda/coop/_types.py tests/coop/test_make_factories_two_phase.py`
    - Result: all hooks passed.

## 2026-02-21 (prep for internal rewrite-factory helper refactor)
- Request: before context compaction, document in detail the next workstream for
  introducing private internal factory helpers (starting with block scan) so
  normalization logic can be centralized without coupling rewrite to public
  `make_*` wrappers.
- Clarification captured:
  - Public `make_*` wrappers are two-phase-facing and now intentionally return
    `Invocable` (or stateful parent objects).
  - Single-phase rewrite paths require primitive instances and internal-only
    kwargs (`unique_id`, `node`, explicit `temp_storage`, `use_array_inputs`,
    block-aggregate plumbing, etc.).
  - Therefore rewrite cannot safely call public `make_*` directly, but we can
    still share alias/default normalization via private helper(s).
- Planned implementation shape (scan as pilot):
  - In `cuda/coop/block/_block_scan.py`:
    - add a private normalization builder (e.g. `_build_scan_spec(...)`) for
      lightweight alias/default handling only (`dim` vs `threads_per_block`,
      prefix callback aliases, default mode/op wiring).
    - add `_make_scan_two_phase(...)` used by public `make_scan`.
    - add `_make_scan_rewrite(...)` used by rewrite node instantiation.
  - In `cuda/coop/block/__init__.py`:
    - keep public `make_scan` signature stable; route internals through the new
      two-phase helper.
  - In `cuda/coop/_rewrite.py`:
    - switch scan instantiation path to the rewrite helper.
    - ensure rewrite-only kwargs and metadata flow are unchanged.
- Guardrails agreed for the next pass:
  - Keep semantic validation in primitive constructors (`scan.__init__`) as the
    source of truth; helpers should normalize, not re-validate semantically.
  - Do not regress two-phase `Invocable` contract.
  - Do not alter single-phase call surface (`coop.block.scan(...)` in kernels).
- Test plan for next pass:
  - `tests/coop/test_make_factories_two_phase.py` (factory contract + runtime)
  - `tests/coop/test_block_scan.py` / `tests/coop/test_block_scan_api.py`
  - `tests/coop/test_block_stress_kernels.py`
  - `tests/coop/test_mamba_selective_scan_fwd.py`
  - targeted pre-commit on touched files.
- Ledger updates completed in this prep pass:
  - `SINGLE-PHASE-TODO.md`: added explicit unchecked work items for private
    rewrite-factory helper rollout.
  - `SINGLE-PHASE-CURRENT-PLAN.md`: added next-pass goal/steps/acceptance
    criteria for scan-first helper refactor.
  - `SINGLE-PHASE-NOTES.md`: updated with design guidance for the helper split
    (see latest “Factory split” note).

## 2026-02-21 (warp internal factory-helper rollout)
- Request: proceed with the internal factory-helper refactor and start with
  warp primitives.
- Changes:
  - `cuda/coop/warp/_warp_load_store.py`:
    - added internal helper split:
      - `_build_load_spec(...)`, `_make_load_two_phase(...)`,
        `_make_load_rewrite(...)`
      - `_build_store_spec(...)`, `_make_store_two_phase(...)`,
        `_make_store_rewrite(...)`
  - `cuda/coop/warp/_warp_exchange.py`:
    - added `_build_exchange_spec(...)`,
      `_make_exchange_two_phase(...)`, `_make_exchange_rewrite(...)`.
  - `cuda/coop/warp/_warp_reduce.py`:
    - added `_build_reduce_spec(...)`, `_make_reduce_two_phase(...)`,
      `_make_reduce_rewrite(...)`.
    - added `_build_sum_spec(...)`, `_make_sum_two_phase(...)`,
      `_make_sum_rewrite(...)`.
  - `cuda/coop/warp/_warp_scan.py`:
    - added helper builders and split creators for:
      - exclusive/inclusive sum
      - exclusive/inclusive scan
    - two-phase helpers return `create(...)`; rewrite helpers return primitive
      instances with rewrite-only kwargs (`unique_id`, `temp_storage`,
      `warp_aggregate`, `valid_items`).
  - `cuda/coop/warp/_warp_merge_sort.py`:
    - added `_build_merge_sort_keys_spec(...)`,
      `_make_merge_sort_keys_two_phase(...)`,
      `_make_merge_sort_keys_rewrite(...)`.
    - added `_build_merge_sort_pairs_spec(...)`,
      `_make_merge_sort_pairs_two_phase(...)`,
      `_make_merge_sort_pairs_rewrite(...)`.
  - `cuda/coop/warp/__init__.py`:
    - updated public `make_*` wrappers to route through `_make_*_two_phase`
      helpers (keeps public two-phase `Invocable` contract unchanged).
  - `cuda/coop/_decls.py`:
    - imported warp `_make_*_rewrite` helpers and set `impl_key` for warp
      decl templates:
      - load/store/exchange/reduce/sum
      - inclusive/exclusive sum
      - inclusive/exclusive scan
      - merge_sort_keys/merge_sort_pairs
    - this makes rewrite instantiation use internal rewrite helpers rather than
      direct public keys.
  - `cuda/coop/_rewrite.py`:
    - hardened `CoopNode.prepare_args()` for callable `impl_class` values by
      deriving `expected_type` from the two-phase instance when needed.
      This avoids type-check issues now that some decl `impl_key` entries are
      helper callables, not classes.
- Tests:
  - `python -m py_compile cuda/coop/warp/_warp_load_store.py cuda/coop/warp/_warp_exchange.py cuda/coop/warp/_warp_merge_sort.py cuda/coop/warp/_warp_reduce.py cuda/coop/warp/_warp_scan.py cuda/coop/warp/__init__.py cuda/coop/_decls.py cuda/coop/_rewrite.py` (passed)
  - `pre-commit run --files cuda/coop/_decls.py cuda/coop/_rewrite.py cuda/coop/warp/__init__.py cuda/coop/warp/_warp_exchange.py cuda/coop/warp/_warp_load_store.py cuda/coop/warp/_warp_merge_sort.py cuda/coop/warp/_warp_reduce.py cuda/coop/warp/_warp_scan.py` (passed)
  - `pytest -q tests/coop/test_make_factories_two_phase.py tests/coop/test_warp_load_store_api.py tests/coop/test_warp_exchange_api.py tests/coop/test_warp_reduce_api.py tests/coop/test_warp_scan_api.py tests/coop/test_warp_merge_sort_api.py tests/coop/test_warp_merge_sort_pairs_api.py tests/coop/test_warp_reduce.py tests/coop/test_warp_scan.py tests/coop/test_warp_merge_sort.py`
    - Result: `121 passed`.

## 2026-02-21 (block internal factory-helper rollout)
- Request: continue the internal factory-helper refactor for block primitives
  after warp rollout; update tests and ensure coverage passes.
- Changes:
  - Added private helper split across block primitive modules:
    - `cuda/coop/block/_block_load_store.py`:
      - `_build_load_spec/_build_store_spec`
      - `_make_load_two_phase/_make_load_rewrite`
      - `_make_store_two_phase/_make_store_rewrite`
    - `cuda/coop/block/_block_exchange.py`:
      - `_build_exchange_spec`
      - `_make_exchange_two_phase/_make_exchange_rewrite`
    - `cuda/coop/block/_block_reduce.py`:
      - `_build_reduce_spec/_build_sum_spec`
      - `_make_reduce_two_phase/_make_reduce_rewrite`
      - `_make_sum_two_phase/_make_sum_rewrite`
    - `cuda/coop/block/_block_scan.py`:
      - `_build_scan_spec` + per-alias builders for
        exclusive/inclusive sum/scan
      - `_make_scan_two_phase/_make_scan_rewrite`
      - `_make_exclusive_sum_two_phase`
      - `_make_inclusive_sum_two_phase`
      - `_make_exclusive_scan_two_phase`
      - `_make_inclusive_scan_two_phase`
    - `cuda/coop/block/_block_merge_sort.py`:
      - `_build_merge_sort_keys_spec/_build_merge_sort_pairs_spec`
      - `_make_merge_sort_keys_two_phase/_make_merge_sort_keys_rewrite`
      - `_make_merge_sort_pairs_two_phase/_make_merge_sort_pairs_rewrite`
    - `cuda/coop/block/_block_radix_sort.py`:
      - `_build_radix_sort_keys_spec/_build_radix_sort_pairs_spec`
      - `_make_radix_sort_keys(_descending)_two_phase/_rewrite`
      - `_make_radix_sort_pairs(_descending)_two_phase/_rewrite`
    - `cuda/coop/block/_block_radix_rank.py`:
      - `_build_radix_rank_spec`
      - `_make_radix_rank_two_phase/_make_radix_rank_rewrite`
    - `cuda/coop/block/_block_adjacent_difference.py`:
      - `_build_adjacent_difference_spec`
      - `_make_adjacent_difference_two_phase/_rewrite`
    - `cuda/coop/block/_block_discontinuity.py`:
      - `_build_discontinuity_spec`
      - `_make_discontinuity_two_phase/_rewrite`
    - `cuda/coop/block/_block_shuffle.py`:
      - `_build_shuffle_spec`
      - `_make_shuffle_two_phase/_make_shuffle_rewrite`
    - Stateful parent helper routing for wrappers:
      - `cuda/coop/block/_block_histogram.py`: `_make_histogram_two_phase`
      - `cuda/coop/block/_block_run_length_decode.py`: `_make_run_length_two_phase`
  - Updated public block wrappers to route through two-phase helpers:
    - `cuda/coop/block/__init__.py`
    - Removed wrapper-local normalization helpers; normalization now lives in
      primitive modules.
  - Updated block decl impl bindings to helper callables:
    - `cuda/coop/_decls.py`
    - Added block helper imports (aliased to avoid warp helper name collisions).
    - Set `impl_key` for block decl templates:
      - load/store/exchange
      - merge_sort_keys/pairs
      - adjacent_difference/shuffle/discontinuity
      - radix_sort_keys/descending
      - radix_rank
      - reduce/sum
      - scan + exclusive/inclusive sum/scan aliases (all via block scan helper).
  - Fixed load/store rewrite default-algorithm fallback for callable `impl_key`:
    - `cuda/coop/_rewrite.py`
    - `CoopLoadStoreNode.refine_match()` now resolves default algorithm from:
      1) `impl_class.default_algorithm` when available,
      2) instance/type default when present,
      3) block load/store classes as fallback.
    - This avoids `AttributeError` when `impl_class` is a helper function.
- TODO updates:
  - Marked complete in `SINGLE-PHASE-TODO.md`:
    - private rewrite-factory helper rollout (block scan+),
    - shared normalization pattern adoption,
    - rewrite migration/parity coverage item.
- Tests:
  - Compile/sanity:
    - `python -m py_compile cuda/coop/block/__init__.py cuda/coop/_decls.py cuda/coop/block/_block_load_store.py cuda/coop/block/_block_exchange.py cuda/coop/block/_block_reduce.py cuda/coop/block/_block_scan.py cuda/coop/block/_block_merge_sort.py cuda/coop/block/_block_radix_sort.py cuda/coop/block/_block_radix_rank.py cuda/coop/block/_block_adjacent_difference.py cuda/coop/block/_block_discontinuity.py cuda/coop/block/_block_shuffle.py cuda/coop/block/_block_histogram.py cuda/coop/block/_block_run_length_decode.py` (passed)
  - Initial broad block/mamba sweep surfaced callable-impl default-algorithm failures in load/store scan paths.
  - Focused verify after fix:
    - `pytest -q tests/coop/test_block_load_store_api.py tests/coop/test_block_scan.py tests/coop/test_block_scan_api.py tests/coop/test_block_scan_single_phase_aliases.py`
      - Result: `223 passed, 26 skipped`.
  - Final broad block/mamba sweep:
    - `pytest -q tests/coop/test_make_factories_two_phase.py tests/coop/test_block_load.py tests/coop/test_block_store.py tests/coop/test_block_load_store_api.py tests/coop/test_block_exchange.py tests/coop/test_block_exchange_api.py tests/coop/test_block_reduce.py tests/coop/test_block_scan.py tests/coop/test_block_scan_api.py tests/coop/test_block_scan_single_phase_aliases.py tests/coop/test_block_merge_sort.py tests/coop/test_block_radix_sort.py tests/coop/test_block_radix_rank.py tests/coop/test_block_adjacent_difference.py tests/coop/test_block_adjacent_difference_api.py tests/coop/test_block_discontinuity.py tests/coop/test_block_discontinuity_api.py tests/coop/test_block_discontinuity_flag_heads_and_tails_api.py tests/coop/test_block_shuffle.py tests/coop/test_block_run_length_decode.py tests/coop/test_block_histogram.py tests/coop/test_block_weird_dims_stateful_custom_types.py tests/coop/test_block_stress_kernels.py tests/coop/test_mamba_selective_scan_fwd.py`
      - Result: `2328 passed, 26 skipped, 5 xfailed`.

## 2026-02-21 (TempStorage getitem sugar rewrite support)
- Request: support `primitive[temp_storage](...)` syntax as rewrite-time sugar
  for `temp_storage=temp_storage`, with robust behavior and tests.
- Changes:
  - `cuda/coop/_decls.py`:
    - generalized `CoopTempStorageGetItemDecl` so getitem-temp-storage typing can
      apply to coop primitives that expose a `temp_storage` parameter in their
      signature (beyond block load/store).
    - kept explicit block load/store getitem decls and added a generic
      fallback getitem decl for other coop primitives with temp_storage support.
  - `cuda/coop/_rewrite.py`:
    - `CoopNode.bound` now detects getitem sugar call sites and injects
      synthetic `temp_storage=` for binding.
    - Added getitem/temp-storage detection helpers in `CoopNodeRewriter` and
      updated `_expr_uses_var(...)` so TempStorage inference/usage detection
      includes `primitive[temp_storage](...)`.
    - Extended root-definition traversal to handle `getitem/static_getitem`
      expressions so call-root resolution works for subscripted primitive calls.
    - In `apply()`, desugar getitem assignments for coop primitive callables
      into plain callable aliases so no callable-getitem IR reaches lowering.
    - Added duplicate-API guard: reject simultaneous
      `primitive[temp_storage](..., temp_storage=...)`.
  - `tests/coop/test_block_load_store_api.py`:
    - Added `test_block_load_store_temp_storage_getitem_sugar_infer_from_omitted_size_alignment`.
    - Added `test_block_load_store_temp_storage_getitem_sugar_distinct_temp_storage_vars`.
    - Added `test_block_load_store_temp_storage_getitem_sugar_rejects_duplicate_temp_storage_kwarg`.
  - `tests/coop/test_block_reduce.py`:
    - Added `test_block_reduce_temp_storage_getitem_sugar` to validate
      non-load/store getitem-temp-storage syntax.
- Tests:
  - `python -m py_compile cuda/coop/_decls.py cuda/coop/_rewrite.py tests/coop/test_block_load_store_api.py tests/coop/test_block_reduce.py` (passed)
  - `pytest -q tests/coop/test_block_load_store_api.py -k "getitem_sugar"` (passed, 3 tests)
  - `pytest -q tests/coop/test_block_reduce.py -k "temp_storage_getitem_sugar"` (passed, 1 test)
  - `pytest -q tests/coop/test_warp_reduce.py -k "temp_storage_getitem_sugar"` (passed, 1 test)
  - `pytest -q tests/coop/test_block_load_store_api.py -k "getitem_sugar" tests/coop/test_block_reduce.py -k "temp_storage_getitem_sugar"` (passed, 4 tests)
  - `pytest -q tests/coop/test_block_load_store_api.py -k "getitem_sugar" tests/coop/test_block_reduce.py -k "temp_storage_getitem_sugar" tests/coop/test_warp_reduce.py -k "temp_storage"` (passed, 46 tests)
  - `pytest -q tests/coop/test_block_load_store_api.py` (passed, 38 tests)
  - `pytest -q tests/coop/test_block_reduce.py -k "temp_storage"` (passed, 29 tests)
  - `pytest -q tests/coop/test_block_scan.py -k "temp_storage"` (passed, 4 tests)
  - `pytest -q tests/coop/test_warp_load_store_api.py -k "temp_storage"` (passed, 2 tests)
  - `pytest -q tests/coop/test_block_stress_kernels.py -k "dynamic_shared_temp_storage"` (passed, 2 tests)
  - `pytest -q tests/coop/test_block_load.py tests/coop/test_block_store.py tests/coop/test_block_load_store_api.py tests/coop/test_block_scan.py tests/coop/test_warp_load_store_api.py tests/coop/test_block_stress_kernels.py`
    - Result: `732 passed, 26 skipped, 3 xfailed, 2 warnings`.
  - `pre-commit run --files cuda/coop/_rewrite.py tests/coop/test_block_load_store_api.py` (passed)

## 2026-02-21 (mamba bleeding-edge single-phase kernel variant)
- Request: add a new selective-scan forward kernel variant demonstrating the
  latest single-phase cuda.coop quality-of-life features.
- Changes:
  - `tests/coop/mamba_selective_scan_fwd.py`:
    - added `make_selective_scan_fwd_kernel_single_phase_bleeding_edge_qol(...)`.
    - kernel demonstrates:
      - `TempStorage()` omission-driven inference.
      - getitem temp-storage sugar on primitives:
        - `coop.block.load[temp_storage](...)`
        - `coop.block.scan[temp_storage](...)`
        - `coop.block.store[temp_storage](...)`
      - `ThreadData(items_per_thread, dtype=<array>.dtype)` for loaded values,
        removing `items_per_thread=` repetition on load calls.
    - For scan payload/output staging in this custom `Float2` mamba path,
      retained `coop.local.array(...)` to keep lowering stable for element
      assignment in this kernel.
  - `tests/coop/test_mamba_selective_scan_fwd.py`:
    - imported new factory and extended `kernel_variant` parameterization with
      `"single_phase_bleeding_edge_qol"`.
    - added launch branch for the new variant under the existing golden-output
      parity test.
- Tests:
  - `python -m py_compile tests/coop/mamba_selective_scan_fwd.py tests/coop/test_mamba_selective_scan_fwd.py` (passed)
  - `pytest -q tests/coop/test_mamba_selective_scan_fwd.py` (passed, 3 tests)
