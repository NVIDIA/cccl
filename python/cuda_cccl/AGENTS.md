# Agent Notes for cuda.coop Single-Phase Work

This directory tracks the CUDA cooperative single-phase effort. Start with
`SINGLE-PHASE-NOTES.md` for background and design context.

## Working Agreements
- Keep `SINGLE-PHASE-TODO.md` up to date (check off completed items).
- Append to `SINGLE-PHASE-LOG.md` each session (what was requested, what changed,
  decisions made, and tests run).
- GPU tests are available; prefer targeted runs and note any skips.

## Single-Phase Entry Points
- Rewriter: `cuda/coop/_rewrite.py` (CoopNodeRewriter + per-primitive nodes).
- Typing: `cuda/coop/_decls.py` (templates, validation, instance types).
- Primitives: `cuda/coop/block/_block_scan.py`, `cuda/coop/block/_block_load_store.py`,
  `cuda/coop/block/_block_histogram.py`, `cuda/coop/block/_block_reduce.py`,
  `cuda/coop/block/_block_exchange.py`, `cuda/coop/block/_block_run_length_decode.py`,
  `cuda/coop/block/_block_merge_sort.py`, `cuda/coop/block/_block_radix_sort.py`.
- Examples/tests: `tests/coop/test_histo2.py` (single-phase histogram example),
  `tests/coop/test_block_load_store_scan_single_phase.py`,
  `tests/coop/test_block_reduce.py`, `tests/coop/test_block_exchange.py`.

## Scan Status (Single-Phase)
- Use `coop.block.scan` inside kernels; pass `items_per_thread` explicitly.
- Array inputs are supported for `items_per_thread == 1` via the
  `use_array_inputs` path in `_block_scan.py`.
- Known ops via string literals are supported (e.g., "max", "bit_xor").
- Prefix callbacks, callable scan ops, and user-defined types are supported.
- ThreadData inputs are still unsupported (see TODO).

## Reduce Status (Single-Phase)
- Use `coop.block.reduce` / `coop.block.sum` inside kernels.
- Scalar and array inputs are supported (`items_per_thread > 1` requires arrays).
- `num_valid` is supported for scalar inputs only.

## Exchange Status (Single-Phase)
- `coop.block.exchange` supports `StripedToBlocked`, `BlockedToStriped`,
  `BlockedToWarpStriped`, `WarpStripedToBlocked`, `ScatterToBlocked`,
  `ScatterToStriped`, `ScatterToStripedGuarded`, and `ScatterToStripedFlagged`.
- In-place and separate output arrays are supported (scatter variants require
  `ranks`; flagged scatter requires `valid_flags`).
- `warp_time_slicing` is supported as a compile-time boolean.

## Run Length Decode Status (Single-Phase)
- Use `coop.block.run_length` inside kernels; `total_decoded_size` must be a
  1-element local array (integer dtype).
- `run_length.decode` supports optional `relative_offsets` and window offsets.

## Merge/Radix Sort Status (Single-Phase)
- `coop.block.merge_sort_keys` sorts per-thread key arrays (items_per_thread
  literal) using a provided comparison op.
- `coop.block.radix_sort_keys` / `coop.block.radix_sort_keys_descending` sort
  key arrays; optional `begin_bit`/`end_bit` are supported.

## Reminders
- Prefer `coop.block.scan` in kernels for single-phase.
- For two-phase compatibility, `BlockLoad`/`BlockStore` and other capitalized
  constructors still return invocables with `.files` when needed.
- BlockLoad/Store algorithms that use shared memory in CUB: TRANSPOSE,
  WARP_TRANSPOSE, WARP_TRANSPOSE_TIMESLICED (same for store). There is no
  separate BlockLoadToShared API; shared memory is internal to the algorithm.
- After any Python changes, run `pre-commit` checks (e.g. `pre-commit run --files <changed files>`).
