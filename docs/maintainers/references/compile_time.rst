Compile-time benchmark reference
================================

The compile-time benchmark builds generated one-include CUDA translation units
(TUs) and summarizes NVCC ``--fdevice-time-trace`` output. It is a TU
compile-time benchmark, even when the current input set comes from public
include-check targets.

Entry point
-----------

The top-level entry point is:

.. code-block:: bash

  ci/build_compile_time_bench.sh

The wrapper configures a caller-selected CMake preset with compile-time
instrumentation, builds selected targets, prepares Perfetto-friendly trace
copies, writes a generated-TU summary CSV, and emits one or more event summary
CSVs.

Generated outputs
-----------------

By default, outputs are written under:

- ``build/<infix>/<preset>/compile_time/tu_summary.csv``
- ``build/<infix>/<preset>/compile_time/event_reports/``
- ``build/<infix>/<preset>/compile_time/perfetto_traces/``

Raw NVCC traces are generated under:

- ``build/<infix>/<preset>/compile_time/raw_traces/``

When ``-baseline-ref`` is used, baseline raw traces are copied out of the
temporary baseline worktree before cleanup and preserved under:

- ``build/<infix>/<preset>/compile_time/baseline_raw_traces/``

Build controls
--------------

The wrapper accepts build-shape parameters so it behaves like other
``ci/build*.sh`` entry points:

.. code-block:: bash

  PARALLEL_LEVEL=16 ci/build_compile_time_bench.sh \
    -preset all-dev \
    -target libcudacxx.test.public_headers \
    -cmake-options "-DCMAKE_CUDA_ARCHITECTURES=native"

Useful build options include:

- ``-preset <name>``
- ``-cmake-options <args>``
- ``-target <name>`` (repeatable; replaces the default public include-check target set)
- ``-baseline-ref <commit-ish>`` (build a temporary baseline worktree for comparison)
- ``-skip-configure``
- ``-skip-build``

The default target set is the current public include-check target set:

- ``cub.headers.base``
- ``thrust.cpp.cuda.headers.base``
- ``libcudacxx.test.public_headers``

The one-include source TUs are created by those CMake targets themselves. The
benchmark wrapper only enables extra compile options on generated CUDA TUs:
``CCCL_COMPILE_TIME_GENERATE_DEVICE_TIME_TRACES`` writes NVCC device-time trace
JSON, and ``CCCL_COMPILE_TIME_SAVE_PREPROCESSED_TUS`` preserves compiler
preprocessed/temporary TU artifacts for the TU summary.

Event summaries
---------------

Arguments after ``--`` are forwarded to
``ci/compile_time/summarize_events.py`` after the raw trace directory. If no
event-summary arguments are provided, the wrapper runs:

.. code-block:: bash

  -f file-processing -e -n 15

For CI-style multi-slice reports, pass a JSON slice file:

.. code-block:: bash

  ci/build_compile_time_bench.sh -baseline-ref origin/main -- \
    --slices /path/to/slices.json

The slice file contains a ``slices`` array. Each slice has a stable ``id``,
display ``title``, ``filter``, ``timing`` (``inclusive`` or ``exclusive``),
``sort``, ``top``, and ``threshold`` in seconds. Multi-slice mode writes each
slice under ``event_reports/<slice-id>/`` and writes a normalized
``event_reports/summary.json`` manifest for PR comment rendering.

Empty slices are represented in the manifest and CSVs. Slices that match no
events, have no matching trace files, or have no comparable event keys record
warnings so they are visible in PR comments instead of looking like ordinary
no-change results. Empty slices with no warnings are omitted recursively by the
PR comment renderer.

Single-slice examples:

.. code-block:: bash

  ci/build_compile_time_bench.sh -skip-build -- -f scanning-function-body -i -n 20
  ci/build_compile_time_bench.sh -skip-build -- -f template-instantiation -e -n 15 --tag templates
  ci/build_compile_time_bench.sh -skip-build -- -f 'Scanning|Instantiating' -i -n 25
  ci/build_compile_time_bench.sh -skip-build -- -f code-generation -i --scope-filter ""

Baseline comparisons
--------------------

Pass ``-baseline-ref <commit-ish>`` to compare the current tree state against a
baseline commit. The wrapper creates a temporary detached worktree for the
baseline, builds both the current tree and the baseline with the same preset,
targets, and common build options, then runs the requested event slice as a
baseline/current comparison:

.. code-block:: bash

  ci/build_compile_time_bench.sh \
    -baseline-ref origin/main \
    -- -f file-processing -e --sort total -n 25 --threshold 0.2

Comparison mode writes three subdirectories under
``<preset-build-dir>/compile_time/event_reports/``:

- ``baseline/``: the normal report for the baseline traces
- ``current/``: the normal report for the current traces
- ``comparison/``: ``worse`` and ``better`` CSVs for the requested filter,
  timing, exclusivity, sort, and top-N slice

In multi-slice comparison mode, the same layout appears under each
``event_reports/<slice-id>/`` directory.

Trace files are matched by relative path. Delta CSVs only compare event keys
that appear in both sides of the same matched trace file. Unmatched child events
are still counted in their matched parent event's exclusive cost, so disappearing
or newly appearing nested work remains visible as a parent cost change instead
of being subtracted away. If there are no comparable event keys, the comparison
CSVs are still written with headers and no rows.

Pass ``--threshold <seconds>`` after the wrapper's ``--`` separator in
comparison mode to omit ``worse`` / ``better`` rows whose total impact change is
not greater than that threshold. Baseline/current reports use ``--sort`` for
their own top-N ordering, but comparison reports are ranked by total impact
across all matched traces so repeated small movements outrank a larger movement
in only one trace.

When ``-baseline-ref`` is used, the wrapper treats the invocation as an event
comparison and skips the generated-TU CSV unless ``-tu-csv`` is provided
explicitly. To compare arbitrary trace directories outside the wrapper layout,
run ``ci/compile_time/summarize_events.py`` directly.

Pull-request reporting
----------------------

Compile-time PR reporting is configured in ``ci/matrix.yaml`` under
``compile_time.pull_request``. Each config selects the GPU runner, devcontainer
launch arguments, baseline ref, preset, targets, wrapper arguments, and report
slices. ``ci/compile_time/parse_matrix.py`` validates that section and emits the
GitHub Actions matrix for the reusable compile-time benchmark workflow.

The reusable workflow uploads:

- event report CSVs and ``summary.json``
- current raw traces
- baseline raw traces
- Perfetto-friendly traces
- the rendered PR comment body

``ci/compile_time/render_pr_comment.py`` renders the comment from
``summary.json``. Regressions and improvements are rendered in separate
``<details>`` sections and are never mixed in one table. Slice warnings are
rendered separately. Empty sections with no warnings are omitted recursively.
Sticky comments are keyed by ``compile-time-bench-<config-id>``; previous
comments for the same config are archived as outdated when a new one is posted.

This reporting is informational and is not part of the aggregate branch
protection ``CI`` job. Commit messages containing ``[skip-compile-time-bench]``
skip compile-time benchmark dispatch.

Filters and scope filtering
---------------------------

Built-in filter names include:

- ``file-processing``
- ``scanning-function-body``
- ``template-instantiation``
- ``template-class-instantiation``
- ``template-function-instantiation``
- ``pending-instantiations``
- ``frontend``
- ``host-compiler``
- ``code-generation``
- ``optimizer``
- ``total-compilation``
- ``all``

Unknown filters are interpreted as case-insensitive regular expressions over
event names and event details.

Symbol-like events, such as function parsing, template instantiation, function
IR generation, and optimizer-function events, are scope-filtered by default to
top-level CCCL-owned namespaces:

- ``cuda::``
- ``thrust::``
- ``cub::``
- ``cccl::``

This keeps reports focused on CCCL symbols instead of system library symbols
pulled into the same generated TU. The filter applies to demangled trace details
and to decoded namespace prefixes from Itanium-mangled symbols in trace details.
It does not filter path/phase events such as file processing, host compiler
phases, or total compilation time.

Pass ``--scope-filter <regex>`` after the wrapper's ``--`` separator to choose
a different case-sensitive symbol-scope regex. Pass an empty string to disable
symbol-scope filtering:

.. code-block:: bash

  ci/build_compile_time_bench.sh -skip-build -- \
    -f template-instantiation -i --scope-filter ""

``host-compiler`` matches the host compiler preprocessing / compiling events
that appear in the device-time-trace output. ``total-compilation`` is a
synthetic per-trace event whose inclusive time is the wall-clock span from the
first timed trace event to the last timed trace event. This includes host
compiler, cudafe, NVVM, fatbinary, and gaps visible inside the trace timeline,
but it is not a separate external wall-clock measurement of untraced driver,
Ninja, or process-launch overhead.

Compared to Ninja log timings, ``total-compilation`` generally undercounts each
TU by a small, consistent amount because the trace span starts at the first
timed event and ends at the last timed event rather than at process launch/exit.
That makes it a good relative-comparison and ranking metric, but not an exact
replacement for external wall-clock command duration.

Use ``--sort`` to choose the selected ranking metric:

- ``total``
- ``avg``
- ``avg-root-tu``
- ``max``

Perfetto trace preparation
--------------------------

The wrapper prepares Perfetto-friendly trace copies by default. To prepare an
existing trace directory manually:

.. code-block:: bash

  ci/compile_time/prepare_traces.py \
    --input build/<infix>/<preset>/compile_time/raw_traces \
    --output /tmp/compile_time_perfetto

CSV output
----------

``summarize_tus.py`` writes:

- ``tu_input``
- ``transitive_loc``
- ``tu_source``
- ``preprocessed_tu``

This CSV is a generated-TU input/LOC summary. Use the ``total-compilation``
event filter when you need per-TU compile-time rankings from trace data.

``summarize_events.py`` writes stable event keys and both inclusive and
exclusive metrics, including:

- ``event_name``
- ``event_key`` (repo-root-relative path for project file-processing events)
- ``selected_total_s``
- ``selected_avg_per_event_s``
- ``selected_avg_per_root_tu_s``
- ``total_inclusive_s`` / ``total_exclusive_s``
- ``event_count``
- ``trace_count``
- ``root_tu_count``

Comparison CSVs additionally include total-impact columns
(``baseline_impact_s``, ``current_impact_s``, ``impact_delta_s``,
``impact_magnitude_s``) plus selected-metric columns
(``baseline_selected_s``, ``current_selected_s``, ``selected_delta_s``,
``selected_magnitude_s``).

Notebook workflow
-----------------

For exploratory analysis, use ``ci/compile_time/analytics.ipynb``.
