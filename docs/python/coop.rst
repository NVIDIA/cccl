.. _cccl-python-coop:

Cooperative Algorithms (``cuda.coop``)
======================================

.. warning::
   ``cuda.coop`` is an experimental API and is subject to change.

``cuda.coop`` provides cooperative Load and Store operations for Python GPU
kernels. Its shared API describes the participating threads, per-thread data,
and algorithm. A compiler backend lowers those calls to CUDA device code.

Installation
------------

.. code-block:: bash

   python -m pip install cuda-coop

The base package contains the shared API, Block and Warp Load/Store planning,
and the CCCL headers needed by compiler integrations. Backend integrations are
added separately. The base package has no Python package dependencies. Importing the base package does not require a CUDA device.

Backend registration
--------------------

``coop.register("numba-cuda-mlir")`` explicitly selects the Numba-CUDA-MLIR
integration; ``"numba_cuda_mlir"`` is an accepted spelling. Call it on the
host before compiling kernels. This core-only package does not include the
adapter yet, so requesting it raises an informative ``ImportError``.

Groups and algorithms
---------------------

Use ``coop.this_block()`` for a thread block, ``coop.this_warp()`` for a
physical warp of 32 threads, and ``coop.this_warp().group_by(width)`` for a
logical warp of 1, 2, 4, 8, 16, or 32 threads.

Block Load and Store accept ``direct``, ``striped``, ``vectorize``,
``transpose``, ``warp_transpose``, and ``warp_transpose_timesliced``. Physical
and logical Warp operations accept ``direct``, ``striped``, ``vectorize``, and
``transpose``. The planner selects the corresponding CUB specialization and
records storage and synchronization requirements for the compiler backend.

``load(group, source, output)`` fills a ``ThreadData`` object in place and
returns ``None``. ``store(group, destination, value)`` writes a per-thread
value or ``ThreadData`` payload. Both accept a nonnegative element ``offset``
and a ``valid_items`` count within the selected group tile. Load also accepts
``oob_default`` with ``valid_items`` to fill invalid output slots.

Data layout
-----------

For four threads holding two items each, a blocked layout assigns consecutive
items to each thread: thread 0 holds ``[0, 1]``, thread 1 holds ``[2, 3]``, and
so on. A striped layout assigns consecutive items across threads: thread 0
holds ``[0, 4]``, thread 1 holds ``[1, 5]``, and so on.

The ``direct`` and ``vectorize`` algorithms produce blocked per-thread data.
``striped`` produces striped per-thread data. Transpose algorithms use shared
memory to combine coalesced accesses with blocked per-thread data.

Temporary storage
-----------------

``direct``, ``striped``, and ``vectorize`` require no shared temporary storage.
Transpose algorithms require storage and synchronization. The planner lets the
backend allocate storage automatically. Block operations can also accept a
``TempStorage`` object when a kernel needs to control its allocation or reuse.
Warp operations require backend-managed storage with a separate slice for
each participating group.

The shared API functions are compiler markers; invoking an operation outside
a supported kernel compiler raises an error. Group descriptors can be created
and inspected from ordinary Python.

See :doc:`coop_api` for the shared API reference.
