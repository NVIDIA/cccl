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
  `cuda/coop/block/_block_histogram.py`.
- Examples/tests: `tests/coop/test_histo2.py` (single-phase histogram example),
  `tests/coop/test_block_load_store_scan_single_phase.py`.

## Scan Status (Single-Phase)
- Use `coop.block.scan` inside kernels; pass `items_per_thread` explicitly.
- Array inputs are supported for `items_per_thread == 1` via the
  `use_array_inputs` path in `_block_scan.py`.
- Known ops via string literals are supported (e.g., "max", "bit_xor").
- Prefix callbacks, callable scan ops, and user-defined types are supported.
- ThreadData inputs are still unsupported (see TODO).

## Reminders
- Prefer `coop.block.scan` in kernels for single-phase.
- For two-phase compatibility, `BlockLoad`/`BlockStore` and other capitalized
  constructors still return invocables with `.files` when needed.
