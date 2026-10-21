.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Block and Warp Reduce
=====================

``reduce`` combines a group's inputs and returns ``Option<T>``: ``Some(total)``
at group rank zero and ``None`` at every other participating thread. ``sum``
fixes the operator to ``Sum``. Both functions take the group first, preserve
their inputs, and keep the input element type. Defaults select sum, all inputs,
and the Block ``WarpReductions`` algorithm.

Block and Warp accept either a scalar or a shared or mutable borrow of
``ThreadData``. Payload items are ordered by x-fastest thread rank, then by
each thread's item index. Block supports complete one-, two- and
three-dimensional shapes. Warp supports physical warps and logical groups;
each group's total is independent. These operations do not reduce a whole
device-wide array into one value.

A complete program
------------------

The example uses a separate generic helper, inferred payload elements, and
``ITEMS_PER_THREAD`` propagated from the host through the kernel. It checks
one and four items per thread over three blocks, including a short input tail
and full, short, and empty output buffers.

.. literalinclude:: ../../rust/cuda-coop/examples/reduce/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The helper fills the missing input tail with an explicit zero before summing.
Every thread calls the reduction. Only afterwards does the kernel inspect the
result and obtain a block-output witness.

.. literalinclude:: ../../rust/cuda-coop/examples/reduce/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Use the Oxide revision selected by the Cargo manifests and
``nightly-2026-10-11``. From a complete CCCL checkout with that toolchain installed:

.. code-block:: shell

   cd rust/cuda-coop/examples/reduce
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. The compiler discovers the concrete CUB
specializations and scratch layout; the consumer needs no C++ shim, build script,
manual linker inputs, or scratch-byte calculation.

Results and safe output
-----------------------

``Some`` establishes that this thread has a valid numeric result. It supplies
neither global-memory ownership nor collective completion. For one output per
block, ``thread::block_index()`` grants a unique witness only to thread
``(0, 0, 0)``. ``DisjointSlice<T, thread::BlockIndex>::get_mut`` checks the
output bounds before returning a mutable reference. A short or empty output
therefore leaves out-of-range blocks unwritten. This witness also supports
x-fastest indexing of two- and three-dimensional grids.

Keep the reduction outside the terminal result branch so that all members
participate. A Warp result belongs to its own group's first lane; a block-output
witness does not provide a separate slot for every logical group. Results are
not broadcast, and nonroot numeric values are never read or replaced by zero.

Operators, algorithms, and counts
---------------------------------

``ReduceOptions::default()`` and ``Default::default()`` infer the element type.
Consuming builders select static operators and Block algorithms:

.. code-block:: rust

   let maximum = coop::reduce(block, &values,
       coop::ReduceOptions::default().binary_op(coop::Max));
   let total = coop::sum(block, &values,
       coop::SumOptions::default().algorithm(coop::Raking));
   let prefix_total = coop::sum(block, scalar,
       coop::SumOptions::default().valid_items(count));

All ten numeric types (``i8``, ``u8``, ``i16``, ``u16``, ``i32``, ``u32``,
``i64``, ``u64``, ``f32``, ``f64``) support ``Sum``, ``Multiplies``, ``Min`` and
``Max``. Integer types additionally support ``BitAnd``, ``BitOr`` and
``BitXor``. Operators retain the input dtype; narrow integers are not widened.
CUB may regroup associative arithmetic, so floating-point totals need not
match a sequential host reduction exactly. Custom operators are not yet
supported. A generic helper can carry ``Op: ReduceOperator<T>`` and pass the
actual operator value through ``.binary_op(op)``.

Block algorithms are ``WarpReductions`` (default), ``Raking`` and
``RakingCommutativeOnly``. The last requires a commutative operator; all
currently supported builtins qualify. Warp uses its native algorithm and
rejects an explicit Block algorithm or temporary-storage descriptor.

``.valid_items(count)`` is available only for scalar input, even when a payload
happens to contain one item. It selects the first ``count`` scalar contributors
and requires ``1 <= count <= group.count()``. Every group member still calls
the operation with the same count. Zero and oversized counts trap before
reduction. Payload input requires every slot initialized; an unwritten slot
traps before its numeric value is read. For a partial payload, fill the missing
items explicitly with a value appropriate to the operator. An empty Load at
offset zero with an explicit zero default produces a valid all-zero payload
sum; this differs from passing a zero scalar count.

Warp membership and temporary storage
-------------------------------------

The enclosing block must contain complete physical warps. Logical widths
1, 2, 4, 8, 16 and 32 divide each physical warp into independent groups.
Non-exhaustive widths 17 through 31 select the low-width prefix of each
physical warp. Use the genuine membership guard before the operation:

.. code-block:: rust

   let group = coop::this_warp().group_by_non_exhaustive::<23>();
   if group.is_member() == 1 {
       let total = coop::sum(group, &values, Default::default());
       // Consume this group's root result here.
   }

Block operations accept ``.temp_storage(&mut scratch)`` with the same
``TempStorage`` descriptor used by :doc:`load_store` and :doc:`scan`.
Omitted storage requests managed synchronization before dirty scratch reuse.
Explicit manual storage requires completion before conflicting reuse;
``auto_sync()`` adds a trailing Block barrier for scratch-using operations.
No reduction implicitly completes unrelated global Load/Store accesses.
Rust borrowing preserves local ownership; the matching compiler checks
group participation, uniform counts, geometry and scratch use.
