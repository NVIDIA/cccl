# Single-Phase Current Plan (2026-02-19)

## Goal
Rebase `4776-single-phase-cuda-coop-v2` onto `origin/main@788fe7cc59` and
fully integrate maker-style factory functions (`make_*`) while keeping
single-phase kernel usage unchanged.

## Execution Plan

- [x] 1. Snapshot and safety setup
  - Create a backup branch/tag before history surgery.
  - Record current HEAD and merge-base context.

- [x] 2. Rebase/transplant onto `788fe7cc59`
  - Apply the full branch delta onto the new base in one pass to avoid
    repetitive per-commit conflicts.
  - Resolve conflicts with priority:
    - Keep mainline maker-function API surface.
    - Preserve single-phase implementation and coverage from this branch.

- [x] 3. Maker-function integration for single-phase internals
  - Retire primitive `.create()` from active usage paths.
  - Ensure two-phase creation path is maker-only (`make_*`).
  - Keep single-phase call sites unchanged (`coop.block.scan(...)`, etc.).
  - Ensure rewrite constructs primitives via maker functions, not direct
    primitive constructors.

- [x] 4. `_decls.py` and `_rewrite.py` simplification/update
  - Remove dual-interface assumptions where `coop.block.<primitive>(...)`
    could mean two-phase creation.
  - Keep support for invoking two-phase invocables in kernels.
  - Verify parent/child primitives (histogram/run_length) and temp-storage
    behavior remain correct.

- [x] 5. Tests and validation
  - Update tests for maker-style two-phase creation.
  - Run targeted coop GPU tests (block/warp + parent/child + stress + mamba).
  - Run `pre-commit` on changed files.

- [x] 6. Final audit + ledgers
  - Audit for API consistency and obvious regressions.
  - Update `SINGLE-PHASE-TODO.md` and append detailed session notes to
    `SINGLE-PHASE-LOG.md`.
  - Provide final summary with exact test results and any residual risks.

## Execution Notes

- Rebase/transplant work was performed onto `788fe7cc59` using a squash-merge
  conflict resolution strategy for this branch.
- Follow-up (2026-02-21): public `make_*` wrappers were restored to two-phase
  Invocable/stateful behavior; single-phase rewrite now instantiates direct
  primitive constructors again via centralized `CoopNode.instantiate_impl()`.
- Full coop validation completed:
  - `3058 passed, 42 skipped, 5 xfailed` on `pytest -q tests/coop`.

## Follow-Up (2026-02-20)

- [x] Evaluate issue `#4832` (BlockRadixSort UDT decomposer support) against
  current single-phase architecture and run a compile prototype.
- Finding:
  - The blocker is still active and architectural, not just API plumbing:
    current decomposer lowering produces `tuple<T...>` values, while CUB radix
    paths require `tuple<T&...>` and fail template deduction in
    `for_each_member_impl`.

## Follow-Up (2026-02-21, next implementation pass)

### Goal
Introduce private internal factory helpers for rewrite construction so we can
share argument normalization logic with public `make_*` without binding rewrite
to the two-phase `Invocable` contract.

### Why
- Public `make_*` now intentionally returns two-phase objects (`Invocable` or
  stateful parent objects).
- Rewrite needs internal-only kwargs (`unique_id`, `node`, explicit
  `temp_storage`, `use_array_inputs`, etc.) and must produce primitive
  instances, not `Invocable`.
- We still want to avoid duplicating alias/default normalization logic.

### Planned Steps
- [ ] 1. Add block-scan internal helpers in `cuda/coop/block/_block_scan.py`:
  - `_build_scan_spec(...)` for shared normalization only.
  - `_make_scan_two_phase(...)` used by public `make_scan`.
  - `_make_scan_rewrite(...)` used by rewrite paths.
- [ ] 2. Update public wrappers in `cuda/coop/block/__init__.py`:
  - route `make_scan` through `_make_scan_two_phase`.
  - preserve existing public signature and semantics.
- [ ] 3. Update rewrite scan node(s) in `cuda/coop/_rewrite.py`:
  - use `_make_scan_rewrite` for instantiation.
  - verify current rewrite-only kwargs are passed through unchanged.
- [ ] 4. Add parity and regression coverage:
  - two-phase make-scan contract tests (`Invocable` + kernel execution).
  - single-phase scan stress paths (ThreadData, temp storage, prefix callbacks).
  - ensure no regressions in mamba/stress kernels.
- [ ] 5. If pattern works, roll out incrementally to adjacent families:
  - block exclusive/inclusive sum/scan wrappers,
  - then other complex primitives (load/store, merge/radix families).

### Acceptance Criteria
- Public `make_scan(...)` behavior unchanged for users.
- Single-phase scan rewrite behavior unchanged.
- No additional semantic validation duplication introduced in wrappers/helpers
  (constructor remains source of truth for validation).
- Targeted test matrix passes.

## Follow-Up (2026-02-22, PR #7214 `@codex` thread triage)

### Goal
Capture a concrete, thread-by-thread implementation plan for all unresolved
`@codex` review comments on `NVIDIA/cccl#7214` before making code changes.

### Thread-by-Thread Plan
- [ ] 1. `python/cuda_cccl/cuda/coop/block/_block_exchange.py:116`
  - Expand `exchange.__init__` docs with parameter/behavior details for
    in-place vs out-of-place calls, scatter-only arguments, and
    `warp_time_slicing`.
  - Add short docstrings for `_build_exchange_spec`,
    `_make_exchange_two_phase`, and `_make_exchange_rewrite`.

- [ ] 2. `python/cuda_cccl/cuda/coop/block/_block_histogram.py:232`
  - Reconcile `temp_storage` docs with CUB reality (`BlockHistogram` does have
    `TempStorage`) and current Python behavior (`NotImplementedError`).
  - Decide whether this PR should keep explicit temp storage unsupported (with
    clear messaging/tests) or wire support end-to-end.

- [ ] 3. `python/cuda_cccl/cuda/coop/_numba_extension.py:12`
  - Rename module globals from `NUMBA_CCCL_*` to `CUDA_CCCL_*` for internal
    consistency, with compatibility aliases as needed.

- [ ] 4. `python/cuda_cccl/cuda/coop/_rewrite.py:943`
  - Evaluate the `Primitive`/`primitive_type` TODO and either implement the
    rename or remove the TODO if churn outweighs value.

- [ ] 5. `python/cuda_cccl/cuda/coop/_rewrite.py:1573`
  - Replace the current return-type reconciliation comment with a concrete
    explanation of typemap mismatch after rewrite-time call substitution.

- [ ] 6. `python/cuda_cccl/cuda/coop/_rewrite.py:1593`
  - Remove stale commented debug/breakpoint code or convert it to a
    non-disruptive debug assertion with actionable failure text.

- [ ] 7. `python/cuda_cccl/cuda/coop/_rewrite.py:1666`
  - Replace `SimpleNamespace` rewrite payload with a typed dataclass (fields:
    `g_var`, `g_assign`, `new_call`, `new_assign`, `sig`, `func_ty`,
    `prelude_instrs`) and migrate immediate call sites.

- [ ] 8. `python/cuda_cccl/cuda/coop/_rewrite.py:2063`
  - Reword the cache-priming comment to explain why
    `func_ty.get_call_type(...)` is required (populate `_impl_keys` after
    rewrite-time registration).

- [ ] 9. `python/cuda_cccl/cuda/coop/_rewrite.py:2094`
  - Reword the two-phase wrapper rationale in neutral, technical terms and
    note what would be needed to unify code paths later.

- [ ] 10. `python/cuda_cccl/cuda/coop/_rewrite.py:2155`
  - Remove stray/no-op comment text.

- [ ] 11. `python/cuda_cccl/cuda/coop/_rewrite.py:4435`
  - Replace broad `except Exception` fallback around `ThreadDataType` lookup
    with a deterministic helper and tighter error mode.

- [ ] 12. `python/cuda_cccl/cuda/coop/_rewrite.py:6533`
  - Validate whether `CoopBlockRunLengthDecodeNode.rewrite_details` can defer
    to `CoopNode.do_rewrite()` and remove duplicate logic if parity holds.

- [ ] 13. `python/cuda_cccl/cuda/coop/_rewrite.py:6292`
  - Validate whether `CoopBlockRunLengthNode.rewrite_details` can defer to
    `CoopNode.do_rewrite()` (or a small shared helper) while preserving
    parent-instance return typing and TempStorage prelude insertion behavior.

### Validation Plan (after implementation)
- Run targeted coop tests for touched primitives:
  - `pytest -q tests/coop/test_block_exchange.py`
  - `pytest -q tests/coop/test_block_histogram.py tests/coop/test_histo2.py`
  - `pytest -q tests/coop/test_block_run_length_decode.py`
  - `pytest -q tests/coop/test_block_scan.py tests/coop/test_block_reduce.py`
  - `pytest -q tests/coop/test_warp_*`
- Run `pre-commit run --files <changed files>`.
