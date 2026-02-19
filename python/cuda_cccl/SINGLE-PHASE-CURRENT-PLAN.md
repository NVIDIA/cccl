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
- Single-phase rewrite now consistently instantiates primitives through
  maker functions (`make_*`) via centralized `CoopNode.instantiate_impl()`.
- Full coop validation completed:
  - `3058 passed, 42 skipped, 5 xfailed` on `pytest -q tests/coop`.
