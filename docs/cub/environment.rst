.. _cub-environment:

Execution Environments
======================

Most CUB device-wide algorithms accept an optional *execution environment* as their last
argument. The environment is a single object that bundles everything CUB needs to know
about *how* to run the algorithm: which stream to use, where to get temporary memory, what
reproducibility guarantees to apply, which tuning policy to select etc.

This page explains what the CUB environment APIs are, and how to use them. The main
properties an environment carries, each optional and freely composable with the others,
are:

1. :ref:`Streams <cub-env-building>` — select the stream the algorithm runs on.
2. :ref:`Determinism requirements <cub-env-determinism>` — request a reproducibility
   guarantee for the algorithm's results.
3. :ref:`Custom tuning policy <cub-env-tuning>` — override CUB's built-in kernel tuning
   for the current device.
4. :ref:`Memory resources <cub-env-memory-resource>` — control where temporary storage
   is allocated from.

This list is not exhaustive: a few algorithms accept additional, algorithm-specific
requirements — e.g. tie-breaking and output-ordering controls for
:ref:`cub::DeviceTopK <cub-topk-requirements>` — and the set of properties an environment
can carry keeps growing.

.. contents::
   :local:
   :depth: 2


Why environments?
-----------------

The classic two-phase CUB API requires three steps: query temporary-storage size, allocate,
then execute. That is fine for one-off calls, but it becomes repetitive once you need to
attach a custom stream or a memory pool.

The environment-based single-phase API collapses all of that into one call. The algorithm
queries the environment for a stream, allocates temporary storage from the memory resource
found there, enforces the requested determinism guarantee, and then executes:

.. code-block:: c++

   cub::DeviceReduce::Sum(d_input, d_output, num_items, env);

.. note::

   A byproduct of the environment overloads is that the environment argument is entirely
   optional: since it is defaulted, an algorithm can be invoked with no temporary-storage
   arguments and no environment at all, and every property falls back to its default
   (see :ref:`Fallback behavior <cub-environment-fallback>`):

   .. code-block:: c++

      cub::DeviceReduce::Sum(d_input, d_output, num_items);

The rest of this page shows how to build the ``env`` object.


.. _cub-env-building:

Building an environment
-----------------------

Some types are already valid environments on their own and can be passed directly to an
algorithm. The most common case is a stream: ``cuda::stream_ref`` (or a raw
``cudaStream_t``) passed as the last argument is treated as an environment containing
just that stream.

.. literalinclude:: ../../cub/test/catch2_test_device_reduce_env_api.cu
   :language: c++
   :dedent:
   :start-after: example-begin reduce-env-stream
   :end-before: example-end reduce-env-stream

When you need more than one property, combine them with ``cuda::std::execution::env``.
Properties are listed in any order; CUB queries each one independently. The example below
composes a stream, a memory pool for temporary storage, and a determinism requirement
(the required declarations are provided by ``<cuda/execution>``, ``<cuda/stream>``,
``<cuda/memory_pool>``, and ``<cuda/devices>``):

.. literalinclude:: ../../cub/examples/device/example_device_reduce_env.cu
   :language: c++
   :dedent:
   :start-after: example-begin env-overload-setup
   :end-before: example-end env-overload-setup

.. literalinclude:: ../../cub/examples/device/example_device_reduce_env.cu
   :language: c++
   :dedent:
   :start-after: example-begin env-overload-run
   :end-before: example-end env-overload-run

The same ``env`` object can be passed to multiple algorithm calls without rebuilding it each
time, see :ref:`cub-environment-reuse`.


Environment features
--------------------

Each of the controls outlined at the top of this page is detailed below.

.. _cub-env-determinism:

Determinism requirements
~~~~~~~~~~~~~~~~~~~~~~~~

Use ``cuda::execution::require`` to request a reproducibility guarantee:

.. literalinclude:: ../../cub/test/catch2_test_device_reduce_env_api.cu
   :language: c++
   :dedent:
   :start-after: example-begin reduce-env-determinism
   :end-before: example-end reduce-env-determinism

Three levels are available, in increasing strictness: ``not_guaranteed``, ``run_to_run``,
and ``gpu_to_gpu``. The meaning of each guarantee is described in the
:ref:`CCCL determinism overview <cccl-determinism>`; which algorithms support which levels,
and each algorithm's default, are listed in the :ref:`CUB determinism support matrix
<cub-determinism>`. Requesting a level an algorithm does not support is rejected at
compile time.


.. _cub-env-tuning:

Custom tuning policy
~~~~~~~~~~~~~~~~~~~~

Pass a custom policy selector through the environment to override CUB's built-in tuning.
A policy selector is a stateless callable mapping a ``cuda::compute_capability`` to the
algorithm's public policy struct:

.. literalinclude:: ../../cub/test/catch2_test_device_merge_env_api.cu
   :language: c++
   :dedent:
   :start-after: example-begin merge-keys-policy-selector
   :end-before: example-end merge-keys-policy-selector

The selector is wrapped with ``cuda::execution::tune`` and passed as (part of) the
environment:

.. literalinclude:: ../../cub/test/catch2_test_device_merge_env_api.cu
   :language: c++
   :dedent:
   :start-after: example-begin merge-keys-tuning
   :end-before: example-end merge-keys-tuning

The policy selector must be stateless (``std::is_empty_v<T> == true``). CUB passes only its
*type* to device kernels, so any captured state would be silently lost.

.. seealso:: :ref:`cub-policy-selectors` - full guide on defining and composing policy
   selectors.

.. _cub-env-memory-resource:

Memory resources
~~~~~~~~~~~~~~~~

The memory resource controls where an algorithm's temporary storage is allocated from.
Memory resource types are valid environments on their own, so they can be passed directly
or composed with other properties:

.. code-block:: c++

   auto pool = cuda::device_default_memory_pool(cuda::devices[0]);

   // Pass directly...
   cub::DeviceReduce::Sum(d_input, d_output, num_items, pool);

   // ...or compose with other properties
   auto env = cuda::std::execution::env{cuda::stream_ref{stream}, pool};
   cub::DeviceReduce::Sum(d_input, d_output, num_items, env);

Temporary storage is allocated from the memory resource on the algorithm's stream before
execution and released on the same stream afterwards. When no memory resource is present
in the environment, CUB falls back to a stream-ordered ``cudaMallocAsync`` allocator
(see :ref:`Fallback behavior <cub-environment-fallback>`).


.. _cub-environment-reuse:

Reusing an environment across multiple calls
---------------------------------------------

An ``env`` object is copyable and movable. Build it once and pass it to as many algorithm
calls as you like:

.. code-block:: c++

   auto stream = cuda::stream{cuda::devices[0]};
   auto pool   = cuda::device_default_memory_pool(cuda::devices[0]);
   auto env    = cuda::std::execution::env{cuda::stream_ref{stream}, pool};

   cub::DeviceScan::ExclusiveSum(d_a, d_out_a, n, env);
   cub::DeviceReduce::Sum(d_b, d_out_b, n, env);
   cub::DeviceSelect::If(d_c, d_out_c, d_num_selected, n, my_predicate, env);

   stream.sync();

All three calls share the same stream and memory pool. Temporary storage is allocated and
released from the pool independently for each call.

.. note::

   When a tuning policy is embedded in the environment, the *same* policy applies to every
   call that uses that environment. If two algorithms in the same pipeline need different
   tunings, build separate environments or use separate ``cuda::execution::tune`` properties.


.. _cub-environment-fallback:

Fallback behavior
-----------------

The environment argument is optional on every algorithm that supports it. CUB applies the
following defaults when a property is absent from the environment (or when no environment
is passed at all):

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Property
     - Default when absent
   * - Stream
     - The CUDA null stream (``cudaStreamDefault``), which synchronizes with all other streams
       on the current device.
   * - Memory resource
     - A stream-ordered allocator based on ``cudaMallocAsync``. Temporary storage is
       allocated on the stream the algorithm runs on.
   * - Determinism
     - Algorithm-specific, typically ``run_to_run``. See :ref:`cub-determinism`.
   * - Tuning policy
     - CUB's built-in architecture-tuned default for the current device.

Missing properties never cause an error. The algorithm simply falls back to its default for
that property.


Environment building blocks
---------------------------

These are the utilities a user needs to construct environments:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Utility
     - Purpose
   * - ``cuda::std::execution::env{props...}``
     - Compose multiple properties into one environment object.
   * - ``cuda::execution::require(property)``
     - Wrap a requirement (e.g. a determinism level) so that an algorithm enforces rather
       than hints it.
   * - ``cuda::execution::tune(selector)``
     - Embed a custom policy selector into an environment.

..
   TODO(gonidelis): link to the developer-facing environments page (queries, CPOs, prop,
   building custom environment types) once it lands via
   https://github.com/NVIDIA/cccl/pull/10013


See also
--------

- :ref:`cub-determinism` - per-algorithm determinism support matrix
- :ref:`cub-policy-selectors` - defining and registering custom tuning policies
- :ref:`device-module` - overview of the device-wide API and the two-phase alternative
- :ref:`cccl-determinism` - CCCL-level determinism concepts
