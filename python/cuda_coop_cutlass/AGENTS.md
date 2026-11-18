<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# cuda.coop.cutlass contributor and review guidance

Apply the repository's root `AGENTS.md` and this guidance when editing or
reviewing this package, `docs/python/coop_cutlass/`, `docs/_static/coop-*`, or
`docs/_ext/coop_visualization.py`. Paths below are relative to the repository
root. Read
[Finding the implementation](../../docs/python/coop_cutlass/developer_guide.rst#finding-the-implementation)
for the source map and
[Keeping documentation in sync](../../docs/python/coop_cutlass/developer_guide.rst#keeping-documentation-in-sync)
for the maintenance workflow.

- When API semantics change, update the docstring that states the rule, the
  pages that link to it, the runnable examples and the runtime tests. Each
  rule is stated once, in the docstring of the function, class or module it
  governs. Other pages link to that entry in the API reference. Check
  participation, layouts, result ownership, storage lifetime and
  synchronization.
- The module docstrings of `cuda.coop.cutlass` and of its modules
  `_thread_group`, `_thread_data`, `_temp_storage`, each `_group_*` and
  `_device_operator` are user documentation.
  `docs/python/coop_cutlass/api.rst` renders each one in its section with
  `automodule` and the options `:no-members:`, `:no-special-members:` and
  `:no-index:`, followed by `.. currentmodule:: cuda.coop.cutlass`. Notes for
  maintainers go in an `# Implementation notes.` comment under the docstring.
- Every complete kernel shown in a user page or docstring comes from a
  tested file under `python/cuda_coop_cutlass/examples/` or
  `python/cuda_coop_cutlass/tests/runtime/` through `literalinclude`, with
  markers when it shows part of the file. The kernel and launcher in
  `docs/python/coop_cutlass/visualizations/index.rst` hold no primitive
  call and are written in the page. Preserve the markers. The Sphinx
  build drops an include whose file or marker is missing and still passes,
  so after changing a marker, an include or `api.rst`, run
  `python -m pytest python/cuda_coop_cutlass/tests/test_docs_sources.py`. It
  checks the file and markers of each include, the GitHub link above each
  include in a docstring, and the `currentmodule` reset. It needs no GPU.
- When compiler phases, class/function names or ownership boundaries depicted
  in the Developer Guide change, update the corresponding phase and swimlane
  SVGs under `docs/python/coop_cutlass/visualizations/`, including overview,
  Python-class and mobile variants.
- When an operation visualization changes, keep its RST explanation, static
  SVG and `docs/_static/coop-*.js` explorer consistent with the implementation.
  Update the `:alt:` text and caption where a module docstring shows the
  same SVG. Check the mapping in `docs/_ext/coop_visualization.py`.
- Reviewers should flag specific stale documentation or diagrams caused by the
  diff, including when the diff changes only implementation files. Update only
  affected material. Add or revise an FAQ only when it answers a relevant user
  question.
- Build affected docs with Sphinx warnings treated as errors. Run relevant
  examples or runtime tests when their code or documented behavior changes.
  Inspect changed diagrams and pages at desktop and narrow widths, including
  the static fallback for an interactive visualization.
- Documentation pages and public docstrings follow these prose rules:
  - Open with one sentence, then a table, a figure or an example.
  - Say what happens in short sentences, one fact each. Keep a "does not"
    only for a mistake readers make, next to the right way.
  - Where the positive statement stands by itself, write it without
    "X, not Y" or "rather than". Name the exception, with no "most"
    and no "unless ... says otherwise".
  - No em-dashes. Semicolons only inside lists.
  - Test each general sentence on the other case: block and warp, payload
    and scalar, every mode and algorithm, with and without `valid_items`.
  - User pages and public docstrings say primitive, group, lane, payload, item,
    element type, tile and scratch. Provider, lowering, planner, wrapper,
    descriptor, backend, adapter and trace time are for the contributor pages:
    the Developer Guide, the primer and the integration example.
