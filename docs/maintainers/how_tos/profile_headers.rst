Profile headers
===============

Optimizing compile time in CCCL headers can be just as important as optimizing runtime performance.

This workflow profiles headers using generated one-header translation units (TUs) and
NVCC device-time-trace output. It supports two modes:

- ``public``: profile direct per-public-header costs (compile time + transitive LOC)
- ``all``: aggregate exclusive header-processing time across all
  generated public-header TUs

For each public header in CCCL, generated TUs have the form:

.. code-block:: c++

  #include <HEADER_NAME>
  int main() {
    return 0;
  }

- per-header TU compile time is computed from build timing logs
- transitive LOC is computed with ``cloc`` on preprocessed output
- transitive include/time metrics are computed from device-time-trace events;
  ``all`` mode subtracts nested header-processing spans from each header event
  before ranking headers
- Perfetto-ready trace copies are written next to the raw traces under
  ``build/profile-headers/header_testing/device_time_trace_for_perfetto``

How to run
----------

.. code-block:: bash

  ci/profile_headers.sh \
    --output-csv /tmp/profile_headers.csv

Run ``public`` mode explicitly:

.. code-block:: bash

  ci/profile_headers.sh \
    --mode public \
    --output-csv /tmp/profile_headers_public.csv

Run ``all`` mode explicitly:

.. code-block:: bash

  ci/profile_headers.sh \
    --mode all \
    --output-csv /tmp/profile_headers_all.csv

To also print an aggregate expensive-header report using ``ctadvisor``:

.. code-block:: bash

  ci/profile_headers.sh \
    --mode all \
    --output-csv /tmp/profile_headers.csv \
    --ctadvisor

``ci/profile_headers.sh`` prepares Perfetto-friendly trace copies automatically.
To prepare an existing raw trace directory manually:

.. code-block:: bash

  ci/profile_headers_prepare_trace.py \
    --input build/profile-headers/header_testing/device_time_trace \
    --output /tmp/profile_header_traces_for_perfetto

CSV output
----------

``--mode public`` writes:

- ``header_path``
- ``compile_time_ms``
- ``transitive_loc``

``--mode all`` writes:

- ``header_path``
- ``weighted_exclusive_process_time_s``:
  ``public_tu_count * avg_exclusive_process_time_s``, sorted descending
- ``public_tu_count``: number of distinct generated public-header TUs where
  the header appears at least once
- ``avg_exclusive_process_time_s``: average exclusive processing time per
  including public-header TU
- ``total_exclusive_process_time_s``: total processing time after
  subtracting nested header-processing spans
- ``total_inclusive_process_time_s``: raw inclusive processing time from the
  trace events
- ``event_count``: number of processing events aggregated for the header


Notebook workflow
-----------------

For exploratory analysis, use ``ci/profile_headers_analytics.ipynb``.
