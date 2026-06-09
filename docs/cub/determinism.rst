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
``cuda::execution::require``. For example, to request run-to-run reproducibility from
``cub::DeviceReduce::Sum``:

.. code-block:: c++

    #include <cub/device/device_reduce.cuh>
    #include <cuda/execution>

    auto input  = thrust::device_vector<int>{0, 1, 2, 3};
    auto output = thrust::device_vector<int>(1);

    auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

    cub::DeviceReduce::Sum(input.begin(), output.begin(), input.size(), env);
    // output[0] == 6

.. note::

   A determinism property must be wrapped in ``cuda::execution::require`` to take effect. Passing a
   bare ``cuda::execution::determinism::*`` value through the environment is a compile-time error: the
   algorithm only honors determinism that arrives as a *requirement*.

When a guarantee is not requested, each algorithm applies its own default (see the matrix below). When
a requested guarantee cannot be satisfied for the given value type and operator, the call fails to
compile with a message describing the constraint. When a stronger guarantee can be met by a cheaper
but still-sufficient implementation — for example an exactly-associative operator such as integer
addition — the algorithm transparently selects it.

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
     - Yes (constrained)
     - ``run_to_run``
   * - ``cub::DeviceScan`` (``ExclusiveSum``, ``ExclusiveScan``, ``InclusiveSum``, ``InclusiveScan``, ...)
     - Yes
     - Yes (constrained)
     - Yes (constrained)
     - ``not_guaranteed``
   * - ``cub::DeviceSegmentedReduce``
     - Yes
     - Yes
     - No
     - ``run_to_run``

Per-algorithm notes (TODO : should we add in algorithm-specific docs instead of here?)
---------------------------------------------------------------------------------------

``cub::DeviceReduce``
   Supports all three guarantees. The default is ``run_to_run``.

   - ``gpu_to_gpu`` has a dedicated, hardware-independent implementation for ``float`` and ``double``
     with ``cuda::std::plus``, based on a *Reproducible Floating-point Accumulator* (RFA): input values
     are grouped into a fixed number of exponent-range bins and accumulated in that fixed, hardware-
     independent order, so the same inputs yield the same bits on any GPU architecture. For exactly-
     associative cases — integral types with a known CUDA binary operator, and ``float``/``double``
     with ``min``/``max`` — the result is already identical across GPUs, so the request is satisfied by
     the (faster) ``run_to_run`` path. Other type/operator combinations under ``gpu_to_gpu`` are
     rejected at compile time.
   - ``not_guaranteed`` uses an atomic accumulation kernel when the conditions for it are met: a
     contiguous output iterator, ``cuda::std::plus``, an accumulator of at least 4 bytes, and an output
     type equal to the accumulator type. When those conditions are not met, the call falls back to
     ``run_to_run`` rather than failing.

``cub::DeviceScan``
   The default is ``not_guaranteed``. Note this differs from ``DeviceReduce``: by default a scan over a
   pseudo-associative operator (e.g. floating-point addition) may vary from run to run.

   - ``run_to_run`` is supported for integral types with a known CUDA binary operator, and for
     floating-point types with ``cuda::std::plus``. The floating-point ``plus`` case engages a stable,
     fixed reduction order so results are reproducible across runs on the same GPU.
   - ``gpu_to_gpu`` is supported for integral types with a known CUDA binary operator. (These are
     exactly associative, so the result is already identical across GPUs.)
   - Other combinations under ``run_to_run``/``gpu_to_gpu`` are rejected at compile time.

``cub::DeviceSegmentedReduce``
   Supports ``not_guaranteed`` and ``run_to_run`` (default ``run_to_run``). ``gpu_to_gpu`` is not
   supported and is rejected at compile time.

.. note::

   The set of algorithms that accept determinism requirements, and the type/operator constraints for
   each guarantee, are expanding over time. The constraints above reflect the current implementation;
   the authoritative source is the per-function documentation and the compile-time diagnostics emitted
   when a requirement cannot be met.
