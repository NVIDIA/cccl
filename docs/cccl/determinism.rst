.. _cccl-determinism:

Determinism
===========

Determinism describes whether an algorithm produces the *same result* every time it is run with the
same input. For many parallel algorithms this is not automatic. For reductions and scans, for example,
the order in which partial results are combined depends on how work is scheduled across thousands of
threads, and that schedule can change between launches or between GPUs. When the combining operator is
not perfectly associative — most notably floating-point addition, where ``(a + b) + c`` need not equal
``a + (b + c)`` — a different combining order yields a (slightly) different result, so the output is no
longer identical from one run to the next.

What counts as the "same result" is defined *per algorithm*. For reductions and scans it means a
*bitwise-identical* output. For other algorithms it can be weaker: a deterministic top-k, for example,
guarantees the same *set* of selected items, while the order of those items within the output is a
separate guarantee that an algorithm may expose on its own.

CCCL lets users state the determinism guarantee they need as an explicit *requirement* on an
algorithm, rather than relying on implementation-defined behavior. The library then either
selects an implementation that satisfies the requirement or rejects the call at compile time if the
requirement cannot be met for the given types and operator.

Determinism guarantees
----------------------

By *reproducible* we mean: given the same inputs, an algorithm returns the same output, in the sense
defined for that algorithm (see above). What the guarantees below differ in is the *scope* of that
reproducibility — across repeated runs, across hardware, or not at all. CCCL models three levels,
defined in ``cuda::execution::determinism``:

``not_guaranteed``
   No reproducibility guarantee. The result is a valid answer, but it may differ from one invocation to
   the next — even on the same GPU with the same input. This is usually the fastest option.

``run_to_run``
   The result is reproducible across repeated runs *on the same GPU*, with the same input, build,
   tuning, and launch configuration. It may still differ on a *different* GPU architecture.

``gpu_to_gpu``
   The strongest guarantee: the result is reproducible across repeated runs *and across different GPU
   architectures* — the same inputs yield the same bits whether the algorithm runs on, say, an Ampere or
   a Hopper GPU. This is the most constrained option, is not available for every type/operator
   combination, and is typically the slowest.

The guarantees are ordered from weakest to strongest:
``not_guaranteed`` ⊆ ``run_to_run`` ⊆ ``gpu_to_gpu``. A ``gpu_to_gpu`` result is also reproducible
run-to-run, and a ``run_to_run`` result is a valid (but stronger-than-required) answer wherever
``not_guaranteed`` would be accepted.

For types and operators that are exactly associative (see
:ref:`cuda::is_associative_v <libcudacxx-extended-api-functional-operator-properties>`; for example, integral
addition with well-known operators), every invocation is already reproducible across runs and GPUs, so the
stronger guarantees come for free and the library simply selects the fastest valid implementation.

.. warning::

   ``gpu_to_gpu``/``run_to_run`` reproducibility is guaranteed for a *fixed* CCCL and CUDA Toolkit version, not
   across versions. If a policy selector is specified to change the used tuning, then reproducibility is only
   guaranteed for identical tunings. The bitwise result may also change between CCCL or CUDA Toolkit releases as
   algorithms, reduction structures, or tuning evolve.

Requesting a determinism guarantee
-----------------------------------

Determinism is expressed as a *requirement* and passed to an algorithm through its execution
environment using ``cuda::execution::require``:

.. code-block:: c++

    #include <cuda/execution>

    // Request run-to-run reproducibility for this call.
    auto env = cuda::execution::require(cuda::execution::determinism::run_to_run);

The requirement may be combined with other environment properties — such as a stream or a memory
resource — into a single environment:

.. code-block:: c++

    auto determinism = cuda::execution::require(cuda::execution::determinism::run_to_run);
    auto env         = cuda::std::execution::env{cuda::stream_ref{stream}, memory_resource, determinism};

Passing a determinism property *without* wrapping it in ``require`` is a compile-time error
(*"Determinism should be used inside requires to have an effect."*). ``require`` turns the property
into a *requirement*, which is what the algorithm honors — this prevents a stray determinism property
from being silently ignored.

If an algorithm cannot satisfy the requested guarantee for the given value type and operator, the call
fails to compile with a diagnostic explaining the constraint. If the guarantee can be satisfied by a
weaker-but-sufficient implementation (for example, an exactly-associative operator under
``gpu_to_gpu``), the library transparently selects it.

Where it is used
----------------

Determinism requirements are consumed today by several ``cub`` device algorithms. See the
:ref:`CUB determinism guide <cub-determinism>` for the per-algorithm support matrix, the exact
type/operator constraints, and some examples.

Further reading
---------------

- `Controlling Floating-Point Determinism in NVIDIA CCCL
  <https://developer.nvidia.com/blog/controlling-floating-point-determinism-in-nvidia-cccl/>`_ — a
  deeper walkthrough of the three guarantees and the implementation strategies behind them.
