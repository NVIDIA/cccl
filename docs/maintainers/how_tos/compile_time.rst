Run compile-time benchmarks
===========================

Use the compile-time benchmark when you want to compare how a change affects
CUDA TU compile time. The common workflow compares the current tree against
``origin/main`` and reports the most important movements in generated public
include-check TUs.

For the full option, CSV, filter, and CI contract reference, see
:doc:`../references/compile_time`.

Run the common comparison
-------------------------

From the repository root, run:

.. code-block:: bash

  ci/build_compile_time_bench.sh \
    -baseline-ref origin/main \
    -- --slices /path/to/slices.json

The CI slices live in ``ci/matrix.yaml`` under
``compile_time.pull_request[].slices``. To reproduce the PR shape locally, copy
those slice definitions to a JSON file shaped as:

.. code-block:: json

  {
    "slices": [
      {
        "id": "file-processing",
        "title": "Direct file processing",
        "filter": "file-processing",
        "timing": "exclusive",
        "sort": "total",
        "top": 15,
        "threshold": 0.2
      }
    ]
  }

The wrapper builds both the current tree and the baseline commit with the same
preset, target set, architecture, and other build arguments. It then writes
baseline reports, current reports, comparison CSVs, a ``summary.json`` manifest,
raw traces, and Perfetto-friendly traces under the preset build directory:

.. code-block:: text

  build/<infix>/<preset>/compile_time/

Run a quick single-slice report
-------------------------------

For iteration after traces already exist, skip the build and regenerate only an
event report:

.. code-block:: bash

  ci/build_compile_time_bench.sh -skip-build -- \
    -f file-processing -e --sort total -n 25

Other useful built-in filters include:

- ``total-compilation``
- ``file-processing``
- ``scanning-function-body``
- ``template-instantiation``
- ``host-compiler``
- ``code-generation``
- ``all``

Interpret the PR comment
------------------------

The PR comment contains one section per configured slice. Within each slice,
regressions and improvements are intentionally separated. Rows are ranked by
total impact across matched traces, not just by the largest single-TU movement.

Important columns:

- ``Regression impact`` / ``Improvement impact``: absolute total-impact change
  across matched traces, in seconds.
- ``Selected Δ``: signed movement in the selected metric for the event.
- ``Baseline`` / ``Current``: selected metric values on each side.
- ``Matched traces``: number of generated TUs where the event key was comparable.

Small movements are filtered by per-slice thresholds from ``ci/matrix.yaml``.
Those thresholds are intentionally non-zero to hide ordinary run-to-run noise.

Inspect traces in Perfetto
--------------------------

The wrapper prepares trace copies whose event names include the useful file or
symbol detail. Open files under:

.. code-block:: text

  build/<infix>/<preset>/compile_time/perfetto_traces/

in Perfetto or another Chrome-trace-compatible viewer.

Skip compile-time benchmark telemetry
-------------------------------------

Compile-time benchmark jobs are informational and do not gate the aggregate PR
``CI`` job. For early iterations where the compile-time benchmark is unrelated,
append this case-sensitive tag to the commit message:

.. code-block:: text

  [skip-compile-time-bench]

Remove the tag before requesting final review if the compile-time benchmark
scripts, workflow, or configuration changed.
