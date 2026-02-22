.. _cccl-python-coop-thread-data:

ThreadData
==========

``coop.ThreadData`` is a cooperative helper type for per-thread arrays. It
lets single-phase and two-phase primitives infer ``items_per_thread`` (and, in
many cases, dtype) from one object instead of repeating those values at every
call site.

Why use it
----------

* Reduces argument repetition for load/store/exchange/scan/reduce pipelines.
* Helps rewrite infer a consistent tile shape across primitive calls.
* Works with explicit :class:`coop.TempStorage` (including getitem sugar).

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

ThreadData with TempStorage
---------------------------

.. literalinclude:: ../../python/cuda_cccl/tests/coop/test_block_load_store_api.py
   :language: python
   :dedent:
   :start-after: example-begin load-store-thread-data-temp-storage
   :end-before: example-end load-store-thread-data-temp-storage

Inference rules
---------------

* ``items_per_thread`` comes from the ``ThreadData`` object when omitted from
  primitive calls.
* Dtype is inferred from primitive usage when possible; pass
  ``coop.ThreadData(..., dtype=...)`` when inference is ambiguous.
* All uses of the same ``ThreadData`` object in one kernel must agree on dtype.
