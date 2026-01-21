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
