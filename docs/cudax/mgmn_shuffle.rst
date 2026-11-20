.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cudax-mgmn-shuffle:

Distributed permutations
========================

``<cuda/experimental/mgmn/shuffle>`` provides ``shuffle`` and ``shuffle_copy``
in ``cuda::experimental::mgmn``. This is an experimental permutation
implementation. It preserves every input occurrence and produces the original
number of elements on each rank. It does **not** currently guarantee uniform
sampling of all permutations or repeatable ordering for a fixed seed.
Construction uses atomic reservations whose ordering can vary between executions.
Its output is not specified to match ``thrust::shuffle``.

Calling convention
------------------

The single-communicator overloads follow MGMN's distributed sort convention:

.. code-block:: cpp

   #include <cuda/experimental/mgmn/shuffle>
   #include <random>

   namespace mgmn = cuda::experimental::mgmn;

   // All ranks participate concurrently with generators producing the same seed.
   std::mt19937 generator{1234};
   mgmn::shuffle(cuda::experimental::distributed, communicator, environment,
                 input, local_size, generator);
   mgmn::shuffle_copy(cuda::experimental::distributed, communicator, environment,
                      input, local_size, output, generator);

When one host thread owns multiple local ranks, pass equal-length ranges of
communicators, environments, iterators and sizes instead. Issue single-communicator
calls concurrently when owning several ranks; serial calls can deadlock.

``shuffle_copy`` preserves its input. Its output must be disjoint from all inputs
and other output ranges. ``shuffle`` stages inputs before writing results, so
in-place operation is safe. Local input sizes may differ or be zero. Values must
be trivially copyable; payload values need not be sortable or hashable. Random
keys are derived from global occurrence indices rather than payload values, so
equal payloads do not share a routing key merely because they compare equal.

The environment provides the stream and optional memory resource. The operation
performs intermediate host metadata synchronizations and leaves final writes
ordered on the supplied streams. Synchronize those streams before host access.

The communicator must provide grouped all-gather, sum all-reduce, fixed all-to-all
and variable all-to-all. MGMN's NCCL fixed all-to-all requires NCCL 2.28 or newer.
No communicator initialization or destruction happens inside the algorithm.

Permutation generation
----------------------

For an output permutation of global indices, pass a
``cuda::counting_iterator<uint64_t>{global_first}`` as the input iterator to
``shuffle_copy``. ``global_first`` is the sum of the preceding ranks' input
sizes. This avoids allocating a materialized input index array. The output still
has the original local size on each rank.

Seed consumption
----------------

A call containing local ranks consumes one ``uniform_int_distribution<uint64_t>``
sample from its generator. Every participant must obtain the same seed, including
when all inputs are empty. A call with no local communicators consumes nothing.
Concurrent single-communicator calls must use separate generator objects.

Implementation
--------------

The implementation adapts the HashGraph algorithm for dense static hash table
construction to generate permutations.
The adapted kernels retain the original HashGraph Apache-2.0 notice.
The eight-byte XXH32 specialization retains its original NVIDIA and xxHash BSD
notices. New MGMN coordination and API files use CCCL's
Apache-2.0 WITH LLVM-exception license.
