.. _cub-determinism:

CUB Determinism
===============

Several ``cub`` device algorithms let you request a reproducibility guarantee for a call. The concepts
behind the three guarantees — ``not_guaranteed``, ``run_to_run``, and ``gpu_to_gpu`` — and the meaning
of *reproducibility* are described in the :ref:`CCCL determinism overview <cccl-determinism>`. This
page documents how to request a guarantee for a CUB algorithm and which algorithms support which
guarantees.

Requesting a guarantee
----------------------

A determinism guarantee is passed to a device algorithm through its execution environment using
``cuda::execution::require``. The example below requests run-to-run reproducibility for
``cub::DeviceReduce::Sum``:

.. literalinclude:: ../../cub/test/catch2_test_device_reduce_env_api.cu
   :language: c++
   :dedent:
   :start-after: example-begin sum-env-determinism
   :end-before: example-end sum-env-determinism

The general rules for requesting a guarantee are described in the
:ref:`CCCL determinism overview <cccl-determinism>`.

Each CUB algorithm has its own default guarantee, applied when none is requested, and its own type and
operator constraints for each guarantee, summarized below.

Support matrix
--------------

.. list-table::
   :header-rows: 1
   :widths: 30 18 18 18 16

   * - Algorithm
     - ``not_guaranteed``
     - ``run_to_run``
     - ``gpu_to_gpu``
     - Default
   * - ``cub::DeviceReduce`` (``Reduce``, ``Sum``, ``Min``, ``Max``, ``TransformReduce``, ...)
     - Yes
     - Yes
     - Yes (partial)
     - ``run_to_run``
   * - ``cub::DeviceScan`` (``ExclusiveSum``, ``ExclusiveScan``, ``InclusiveSum``, ``InclusiveScan``, ...)
     - Yes
     - Yes (partial)
     - Yes (partial)
     - ``not_guaranteed``
   * - ``cub::DeviceSegmentedReduce``
     - Yes
     - Yes
     - No
     - ``run_to_run``

.. note::

   The set of algorithms that accept determinism requirements, and the type/operator constraints for
   each guarantee, are expanding over time. The matrix above reflects the current implementation.

Algorithm-specific determinism models
--------------------------------------

The three guarantees describe the *scope* of reproducibility and fit most algorithms, where a
reproducible result means a *bitwise-identical* output. A few algorithms still use the same three
levels but extend the model with additional, algorithm-specific controls, documented on their own
pages:

- :ref:`cub::DeviceTopK <cub-topk-requirements>` — determinism applies to *set membership* (which
  *K* items are selected) rather than a bitwise-identical buffer, and it adds tie-breaking
  (``cuda::execution::tie_break``) and output-ordering (``cuda::execution::output_ordering``)
  controls.
