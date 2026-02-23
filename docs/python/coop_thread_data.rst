.. _cccl-python-coop-thread-data:

ThreadData
==========

``coop.ThreadData`` is a cooperative helper for per-thread tiles. It lets
single-phase and two-phase primitive calls share one tile object instead of
repeating ``items_per_thread`` and dtype information at each call site.

C++ mental model
----------------

Think of ``coop.ThreadData`` as the Python analog of the idiomatic CUB/C++
per-thread array:

* Python: ``thread_data = coop.ThreadData(items_per_thread)``
* C++: ``T thread_data[ITEMS_PER_THREAD]``

In CUB block APIs, this aligns with overloads that accept parameters like
``const T (&items)[ITEMS_PER_THREAD]``.

Why use it
----------

* Reduces argument repetition for load/store/exchange/scan/reduce pipelines.
* Helps rewrite infer a consistent tile shape across primitive calls.
* Works with explicit :class:`coop.TempStorage` (including getitem sugar).
* Mirrors familiar CUB dataflow while keeping Python kernel code concise.

Basic usage
-----------

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin imports
   :end-before: example-end imports

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin load-store-thread-data
   :end-before: example-end load-store-thread-data

Equivalent C++ shape (for intuition):

.. code-block:: cpp

   template <typename T, int BLOCK_DIM_X, int ITEMS_PER_THREAD>
   __global__ void kernel(const T* d_in, T* d_out) {
     T thread_data[ITEMS_PER_THREAD];
     // BlockLoad(...).Load(d_in, thread_data);
     // BlockScan(...).ExclusiveSum(thread_data, thread_data);
     // BlockStore(...).Store(d_out, thread_data);
   }

How it lowers
-------------

``coop.ThreadData`` is a rewrite-time placeholder. During kernel compilation,
rewrite replaces it with a per-thread ``cuda.local.array(...)`` allocation
using the inferred/explicit dtype and ``items_per_thread``.

This is thread-private storage. It is not a global-memory allocation path.
As with regular CUDA local arrays, backend codegen may keep values in registers
or spill to local memory depending on register pressure.

ThreadData with TempStorage
---------------------------

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin load-store-thread-data-temp-storage
   :end-before: example-end load-store-thread-data-temp-storage

Automatic dtype inference
-------------------------

When ``dtype`` is omitted, rewrite infers it from how the same
``ThreadData`` object is used:

1. Rewrite inspects primitive calls that reference the object.
2. It collects candidate dtypes from typed peers (for example device arrays
   paired with the same ``ThreadData`` in load/store/exchange/scan calls).
3. All candidates must agree.

Outcomes:

* No candidates found: compilation raises
  ``RuntimeError: Could not infer dtype for ThreadData; ...``.
* Conflicting candidates found: compilation raises
  ``RuntimeError: Could not infer a consistent dtype for ThreadData; ...``.

Conflicting dtype example:

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin thread-data-dtype-mismatch-kernel
   :end-before: example-end thread-data-dtype-mismatch-kernel

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin thread-data-dtype-mismatch-usage
   :end-before: example-end thread-data-dtype-mismatch-usage

To avoid inference ambiguity, pass ``dtype`` explicitly:
``coop.ThreadData(items_per_thread, dtype=d_in.dtype)``.
