.. _cccl-python-coop-cutlass:

Getting Started with ``cuda.coop.cutlass``
===========================================

The CUTLASS backend lets CUTLASS CuTe DSL kernels use the portable
:mod:`cuda.coop` operations. A compatible CUTLASS compiler context activates
the backend automatically; the qualified :mod:`cuda.coop.cutlass` API selects
the same implementation explicitly. The backend renders each requested
collective as a CUB or CUDAX C++ shim, compiles it to LTO-IR with NVRTC
against the wheel's private CCCL header bundle, and attaches the artifact
during CUTLASS finalization. The backend supports CUDA 13.

All portable operation families are served: load/store, reduce/sum, the scan
family, exchange, shuffle, adjacent difference, discontinuity, merge sort,
radix sort, radix rank, histogram, run-length decode, and top-k at block
scope, plus the warp-scoped load/store, reduce, scan, exchange, and merge
sort forms.

Installation
------------

Install the ``cuda-coop`` distribution with the CUTLASS dependencies:

.. code-block:: bash

   pip install 'cuda-coop[cutlass,examples]'

The ``examples`` extra installs PyTorch for the executable examples. Code
that only uses ``cuda.coop`` from CUTLASS kernels can install
``cuda-coop[cutlass]`` instead.

An operation must be traced from a compatible compiler context. If no provider
can serve that context, ``cuda.coop`` reports the structured error below
instead of silently choosing another backend:

.. code-block:: text

   cuda.coop.<feature> requires an active compiler backend; install or import a compatible backend before compiling a kernel

Portable load, reduce, and store
--------------------------------

This executable example traces one complete CUDA thread block through the
portable root API: it loads a block tile into per-thread registers, stores a
copy, and reduces the loaded values to a single total. The same source runs
unchanged on any registered backend; here the CUTLASS compiler context serves
it.

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/portable_root_sum.py
   :language: python
   :caption: Portable root Load, Reduce, and Store traced through CUTLASS.

Run it from the repository root or an installed wheel:

.. code-block:: bash

   python -m examples.cutlass.portable_root_sum

The package ships many further executable examples under
``python/cuda_coop/examples/cutlass``: per-primitive array-path examples
(``prims_vector_*.py``), CuTe-tensor examples (``cute_*.py``) including fused
GEMM/MMA + top-k selection pipelines, and mixed-payload compositions. The
package README lists the full inventory with run commands.

Use :mod:`cuda.coop.cutlass` directly when a kernel should be explicitly tied
to CUTLASS. Its exports have the same signatures and behavior as the
automatically activated portable root API, and add deferred ``make_*``
factories.

See :doc:`coop_api` for the complete API.
