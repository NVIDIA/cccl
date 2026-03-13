Profile headers
===============

Optimizing compile time in CCCL headers can be just as important as optimizing runtime performance.

This workflow profiles headers using generated one-header translation units (TUs) and
NVCC device-time-trace output. It supports two modes:

- ``public``: profile direct per-public-header costs (compile time + transitive LOC)
- ``all``: aggregate transitive header behavior across all generated public-header TUs
  (how many public-header TUs include each header + processing time metrics)

For each public header in CCCL, generated TUs have the form:

.. code-block:: c++

  #include <HEADER_NAME>
  int main() {
    return 0;
  }

- per-header TU compile time is computed from build timing logs
- transitive LOC is computed with ``cloc`` on preprocessed output
- transitive include/time metrics are computed from device-time-trace events

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

CSV output
----------

``--mode public`` writes:

- ``header_path``
- ``compile_time_ms``
- ``transitive_loc``

``--mode all`` writes:

- ``header_path``
- ``include_tu_count``: number of distinct generated public-header TUs where the header appears at least once
- ``avg_process_time_s``: average processing time per including TU
- ``total_process_time_s``: total processing time summed across all including TUs


Notebook workflow
-----------------

For exploratory analysis, use ``ci/profile_headers_analytics.ipynb``.
