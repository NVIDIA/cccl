.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Programming model
=================

A Tile value describes elements, their dtype, and a compile-time shape.
An exclusive sum consumes that value and returns its prefix sums. Applications express the operation on the logical elements.
The compiler selects a SIMT implementation and supplies its temporary
shared memory.

Logical blocks and physical threads
-----------------------------------

.. rubric:: Python

The grid in ``ct.launch(stream, grid, kernel, args)`` counts logical
blocks. ``ct.bid(0)`` selects the current block's partition.
``@ct.kernel(num_worker_warps=8)`` requests eight physical worker
warps for each logical block.


.. rubric:: C++

``kernel<<<logical_blocks, 1>>>(...)`` uses ``1`` as the Tile launch
sentinel. ``ct::bid().x`` identifies the logical block.
The kernel attribute
``[[cutile::hint(0, num_worker_warps_per_cta=8)]]`` requests eight
physical worker warps for the whole kernel. Ordinary primitive
calls keep their logical shapes and operations unchanged.


The compiler determines the physical CUDA threads that execute each block.

A 256-element tile can therefore use several values per physical thread.
Its extent does not specify a launch block size. The linkage provider
advertises candidate implementations with concrete thread counts, layouts,
resource requirements, and preference scores. Discovery does not know the
final worker count of the enclosing kernel. The compiler selects from these
candidates, and the provider validates the resolved worker count before
compiling the selected implementation.

With TileIR 13.5, cuTile Python accepts 1, 2, 4, 8, 16, or 32 worker warps.
All foreign calls in the same kernel must admit a compatible
physical execution. See the :doc:`programming guide </tile/python/programming>` for
explicit worker settings.
Ordinary calls do not expose ``BlockThreads`` or ``ItemsPerThread``.
A kernel worker hint controls the CTA's workers. A collective may use a
subset; the provider derives each call's items per thread from its logical
extent and participating workers.
Exchange retains an explicit width because that width changes its result.

Shapes and array partitions
---------------------------

Python reductions select dimensions with ``axis``; C++ reductions use an
``Axis`` template argument. Both accept multidimensional tiles. Other
cooperative operations accept one-dimensional tiles; extract
or reshape the intended row before passing it to those primitives. Keys and
values in a pair operation must have compatible extents. Their dtypes may
differ where the operation supports both types.

Load/store coordinates belong to the Tile language:

.. rubric:: Python

``ct.load(array, index=(block,), shape=(256,))`` selects the block's
256-element partition. Block 1 starts at element 256. Multiplying the
block index by 256 again would skip partitions. The
:doc:`programming guide </tile/python/programming>` checks two distinct partitions.


.. rubric:: C++

A ``ct::partition_view`` over a tensor span with shape ``ct::shape<256>``
selects 256-element partitions. Calling ``view.load(ct::bid().x)`` loads
the current logical block's partition; the view's store uses the same
partition coordinate.


Shape is a compile-time property. A runtime TopK ``k`` or a partial-sort
``valid_items`` changes which results are meaningful, but does not resize
the returned Tile value.

Inputs and results
------------------

.. list-table:: Result shapes
   :header-rows: 1
   :widths: 35 65

   * - Primitive
     - Result
   * - Scan, Exchange, Shuffle, Adjacent Difference, keys-only sorting
     - A tile with the input extent and element dtype.
   * - Sum and Reduce
     - Python removes reduced axes by default; ``keepdims=True`` retains
       them with extent one. C++ ``sum<Axis>`` and ``reduce<Axis>`` retain the
       reduced dimension with extent one. Python ``broadcast=True`` and C++
       calls without an axis repeat each aggregate across the input shape.
       Python addition promotes boolean inputs to ``int32``; C++ boolean
       inputs are unsupported. Other supported input dtypes are preserved.
   * - Pair sorting and pair TopK
     - Two tiles preserving key/value association.
   * - Histogram
     - An integer counter tile with one element per requested bin.
   * - Discontinuity
     - Integer flag tiles; the combined operation returns both head and
       tail flags.
   * - TopK
     - Tiles with the input capacity. Only the first ``min(k, valid_items)``
       positions contain selected results; selection does not promise sorted output.
   * - Run Length Decode
     - A tile with the requested decoded extent, optionally paired with
       relative offsets. Run-input extent and decoded extent are independent.

Each logical block processes its own input tile. A two-block launch performs
two independent scans or reductions. Similarly, sorting each block does not
sort the entire array. Compose a separate cross-block algorithm when the
result must cover the whole launch.

Read the :doc:`API reference </tile/python/api>` for dtype restrictions and boundary
results. In particular, Shuffle endpoints, partial-tile controls, and decoded
positions beyond the expanded runs each have operation-specific contracts.

Compile-time choices and runtime controls
-----------------------------------------

Dtypes, shapes, algorithm names, and operation names determine compiled
specializations. Supported numerical controls can be runtime values, such
as a scan prefix, a sort bit range, or TopK's ``k`` and ``valid_items``.

Use named keyword arguments in Python and the corresponding typed policies
or overloads in C++. Supported scalar-Tile forms depend on the control:
Python TopK accepts scalar or rank-zero Tile controls and rejects a
shape-one Tile. C++ overloads accept the documented shape-one controls.
Consult the specific signature before replacing a scalar with a Tile value.

Scan and Reduce accept a finite set of built-in operations: addition,
multiplication, minimum, maximum, and integer bitwise operations. Python
selects them with an operation name; C++ uses the supported operation
objects. Arbitrary callbacks and functors are outside these interfaces.

Partial tiles
-------------

Choose padding and valid-prefix controls for the operation being performed.
Zero padding preserves a sum, but a padded zero still contributes to a
Histogram bin and can change a minimum or a TopK result. A load mask alone
does not describe which values a later cooperative call should consume.

The programming guides use complete tiles. Before extending them to a tail, check
the primitive's valid-prefix interface and initialize every input element it
will inspect. A caller must also keep writes within its output allocation.

Exchange topology defines the permutation
-----------------------------------------

Exchange requires a compile-time ``block_threads`` in Python and an explicit
``exchange_policy<BlockThreads>`` in C++. For extent ``N``, thread count
``B`` defines ``I = N / B`` values per thread. The two operations compute:

.. code-block:: text

   striped_to_blocked: y[t * I + i] = x[i * B + t]
   blocked_to_striped: y[i * B + t] = x[t * I + i]

For ``N = 256`` and ``B = 128``, the first operation gives ``y[1] = x[128]``.
With ``B = 256``, each thread has one value and the permutation is the
identity. Thread count is therefore part of Exchange's meaning. Use the
same count for an Exchange and its inverse.

The count must be a multiple of 32 in ``[32, 1024]`` and divide the extent.
It also constrains the physical execution of other foreign calls in the
kernel. The compiler cannot substitute a different width while preserving
the requested permutation.

Temporary storage and composition
---------------------------------

The provider reports scratch requirements for the concrete CUB
specialization. The compiler allocates the scratch and connects the call
to the surrounding Tile computation. Application code passes no scratch
descriptor and inserts no SIMT reuse barrier around the primitive.

Multiple calls can remain in one kernel, including calls with different
logical extents. Compilation still requires a compatible implementation
for each call and enough resources for the composition. A kernel that
works with each call separately needs its own validation after the calls
are combined. See :doc:`/tile/interop` for compositions involving other libraries.

.. _tile-radix-capacity:

.. rubric:: Radix scratch capacity

Radix Sort and Rank include scratch storage that grows exponentially with
``radix_bits`` (``RadixBits`` in C++). Usable configurations depend on the
physical worker count and target GPU resources. A value within the documented
policy range can still exceed those resources and fail compilation. Reducing
the radix capacity reduces this part of the storage requirement.
