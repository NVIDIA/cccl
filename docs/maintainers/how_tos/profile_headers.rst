Profile headers
===============

Optimizing compile time in CCCL headers can be just as important as optimizing runtime performance.

To help reason about the baseline compile time cost of each header in CCCL, this guide describes
how to generate a profile of compile time, transitive LOC, and include reference count for CCCL public headers.

For every public header in CCCL, it measures the compile time and expanded LOC of the following file:

.. code-block:: c++

  #include <HEADER_NAME>
  int main() {
    return 0;
  }

- per-header TU compile time is computed from the ``.ninja_log``
- transitive LOC is computed with ``cloc`` on the preprocessed output from a generated file that only includes the header
- include reference count is computed as direct ``#include`` references from other scanned CCCL headers

How to run
------------------

.. code-block:: bash

  ci/profile_headers.sh \
    --output-csv /tmp/profile_headers.csv

Outputs
-------

For ``--output-csv /tmp/profile_headers.csv``, the script writes:

- ``/tmp/profile_headers.csv`` with columns:

  - ``header_path``
  - ``compile_time_ms``
  - ``transitive_loc``
  - ``included_by_header_count``
