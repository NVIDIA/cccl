<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# cuda.coop.cutlass contributor and review guidance

Apply the repository's root `AGENTS.md` and this guidance when editing or
reviewing this package, `docs/python/coop_cutlass/`, `docs/_static/coop-*`, or
`docs/_ext/coop_visualization.py`. Paths below are relative to the repository
root. Read
[Keeping documentation in sync](../../docs/python/coop_cutlass/developer_guide.rst#keeping-documentation-in-sync)
for the source map and maintenance workflow.

- When API semantics change, update affected public docstrings, guide sections,
  runnable examples and their runtime tests. Check participation, layouts,
  result ownership, storage lifetime and synchronization. Preserve the markers
  used by Sphinx `literalinclude` directives.
- When compiler phases, class/function names or ownership boundaries depicted
  in the Developer Guide change, update the corresponding phase and swimlane
  SVGs under `docs/python/coop_cutlass/visualizations/`, including overview,
  Python-class and mobile variants.
- When an operation visualization changes, keep its RST explanation, static
  SVG and `docs/_static/coop-*.js` explorer consistent with the implementation.
  Check the mapping in `docs/_ext/coop_visualization.py`.
- Reviewers should flag specific stale documentation or diagrams caused by the
  diff, including when the diff changes only implementation files. Update only
  affected material; add or revise an FAQ only when it answers a relevant user
  question.
- Build affected docs with Sphinx warnings treated as errors. Run relevant
  examples or runtime tests when their code or documented behavior changes.
  Inspect changed diagrams and pages at desktop and narrow widths, including
  the static fallback for an interactive visualization.
