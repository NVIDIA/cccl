.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Warp batched reduction
======================

``reduce_batched`` reduces each payload slot independently across a Warp group.
If every lane supplies ``B`` items, there are ``B`` reductions: batch ``i``
combines slot ``i`` from every member. The operation takes the group first,
preserves the borrowed input, and returns fresh ``ThreadData`` with the same
element type. Defaults select ``Sum`` and striped output ownership.

Physical Warps and logical widths 1, 2, 4, 8, 16 and 32 are supported within
blocks containing complete physical warps. Each group reduces independently;
all its members participate. Block groups, non-power-of-two Warp prefixes and
scalar inputs are unsupported. Use :doc:`reduce` to combine every slot into
one root-only result.

A complete program
------------------

The example uses eight-lane groups in three 64-thread blocks. It checks one,
four and ten input items per thread, including a short input tail and padding
between groups' output regions. The element type is inferred from the input.

.. literalinclude:: ../../rust/cuda-coop/examples/reduce_batched/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The separate generic helper fills the missing input tail with zero before
summing. Every member calls both the reduction and Store, including lanes
whose result contains no initialized slots.

.. literalinclude:: ../../rust/cuda-coop/examples/reduce_batched/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the manifests and ``1.99.1``.
With that toolchain installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/reduce_batched
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. The compiler discovers the concrete CUB
operations and plans their temporary storage. The consumer needs no C++ shim,
provider build script or manual scratch-byte calculation.

Derived extent and validity
---------------------------

The result capacity per lane is ``K = batched_extent(B, W) = ceil(B / W)``,
where ``W`` is the group width. ``B`` is positive. The return type is
``ThreadData<T, K>`` with default alignment; input alignment does not change
the result type. The caller supplies no separate output extent. Capacity and
validity are distinct: ``get(i)`` returns ``None`` when a slot has no batch.
Such slots remain unwritten, including when an entire lane has no result.

For lane rank ``r`` and output slot ``i``, ownership is:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Output layout
     - Batch index
     - Validity
   * - ``Striped`` (default)
     - ``r + i * W``
     - Batch index is below ``B``
   * - ``Blocked``
     - ``r * K + i``
     - Batch index is below ``B``

For ``B = 10`` and ``W = 8``, each lane has capacity two. Striped lanes zero
and one receive batches ``(0, 8)`` and ``(1, 9)``; lanes two through seven
have only their first slot initialized. With blocked output, lanes zero
through four each receive two consecutive batches, and lanes five through
seven have no initialized result.

Every input slot must be initialized. An unwritten input traps before its
numeric value is read. There is no valid-count or seed option. For a partial
input tile, fill missing items explicitly with a value suitable for the
operator, as the sum example does with zero. Invalid outputs are not filled
with zero or broadcast to other lanes.

Calculated types on the pinned compiler
---------------------------------------

The calculated return signature currently requires
``#![feature(generic_const_exprs)]`` in every crate that instantiates it,
including a separate helper crate and its consumer. The example also uses
``#![allow(incomplete_features)]`` because this Rust feature is incomplete.
Generic helpers repeat the extent expression in the well-formedness bound
``[(); coop::batched_extent(ITEMS_PER_THREAD, 8)]:``, as shown in the executed
source above. Helpers generic over the group use
``<G as coop::ReduceBatchedGroup>::WIDTH`` in that expression.

This is a limitation of the pinned compiler and calculated-result signature.
Ordinary payload, Load/Store, Scan and Reduce consumers do not need this
feature gate. The provider also checks CUB's signed-index limit for the padded
``W * K`` extent.

Operators and safe Store
------------------------

``ReduceBatchedOptions::default()`` and ``Default::default()`` infer ``T``.
The consuming ``.binary_op(op)`` builder selects a builtin operator;
``.output_layout(coop::Blocked)`` selects blocked result ownership.
The operator value can remain generic under ``Op: ReduceOperator<T>``.
All ten numeric types support ``Sum``, ``Multiplies``, ``Min`` and ``Max``;
integers also support ``BitAnd``, ``BitOr`` and ``BitXor``. Results retain the
input dtype. CUB may regroup floating-point arithmetic, so results need not
match a sequential host fold exactly. Custom operators are not yet supported.
There is no explicit algorithm or temporary-storage option.

Store a striped result with the ``Striped`` Store algorithm; store a blocked
result with ``Direct``. In either case, ``valid_items(B)`` selects exactly the
initialized batch results. Keep Store collective: do not guard it with a
lane-local ``get(...).is_some()`` test.

The output carrier is ``DisjointSlice<T, CollectiveTiles<K>>``. Each group's
region has ``W * K`` slots, of which the first ``B`` are written by the matching
Store. Padding remains unchanged. The example passes the block's output
offset, ``block_rank * block_threads * K``; Warp Store adds the subgroup's
origin itself. Do not add that subgroup origin a second time or substitute a
compact ``B``-slot stride for this ownership region.

Scratch is implicit, queried from CUB, and planned per group with completion
before reuse. Result validity does not grant global-memory ownership or
complete unrelated Load/Store accesses. Retain the matching group's ``sync()``
when a later global access depends on an earlier one. See :doc:`load_store`
for the output carrier and completion rules.
