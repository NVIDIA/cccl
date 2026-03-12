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

- per-header TU compile time is computed from the build's compile timing logs
- transitive LOC is computed with ``cloc`` on preprocessed output for generated one-header translation units
- include reference count is computed as direct ``#include`` references from other scanned CCCL headers

How to run
----------

.. code-block:: bash

  ci/profile_headers.sh \
    --output-csv /tmp/profile_headers.csv

To also print an aggregate expensive-header report using ``ctadvisor``:

.. code-block:: bash

  ci/profile_headers.sh \
    --output-csv /tmp/profile_headers.csv \
    --ctadvisor


Notebook workflow
-----------------

For exploratory analysis, use ``ci/profile_headers_analytics.ipynb``.
