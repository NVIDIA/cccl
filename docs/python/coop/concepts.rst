.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-programming-concepts:

Programming concepts
====================

Cooperative operations let a group of threads work on data together. This
page explains the rules shared by the Numba-CUDA-MLIR and CUTLASS APIs:
how threads divide a tile, which threads must participate, and how to use
results and temporary storage. For complete kernels and launch examples,
see the :doc:`Numba-CUDA-MLIR <programming_guide>` or
:doc:`CUTLASS <../coop_cutlass>` programming guide.

.. _coop-api-namespaces:
.. _kernel-api:

.. _portable-and-qualified-apis:

Common and qualified APIs
-------------------------

``from cuda import coop`` selects the common namespace. It defines the shared
contracts for groups, payloads, operations, and temporary storage across
supported DSLs. Calls inside a kernel are compiler markers; they are not
host-side implementations of those operations.

The qualified namespaces, ``cuda.coop.numba_mlir`` and ``cuda.coop.cutlass``,
each include all common kernel operations and their backend's extensions. A
program using one compiler can use its qualified namespace alone. CUTLASS-only
code can use ``import cuda.coop.cutlass as coop``. Use ``numba_coop`` and
``cutlass_coop`` when a module contains both DSLs. Common and qualified calls
can appear in the same kernel when they belong to its compiler. The
comparisons in the :ref:`Numba-CUDA-MLIR guide <coop-programming-api-choice>`
and :ref:`CUTLASS guide <coop-cutlass-api-choice>` list the differences.

A common call preserves its operation's contract across backends. Moving
a kernel still requires adapting launch syntax, array arguments, control
flow, and other DSL code. Compiler-owned payloads cannot be passed between
Numba and CuTe traces.

.. _block-prefix-callbacks:

Numba-CUDA-MLIR supports qualified device operators and
:ref:`Scan prefix callbacks <coop-prefix-callbacks>`. CUTLASS supports
built-in operators; custom operators and stateful Scan callbacks are
currently unsupported.

.. _coop-common-calling-conventions:

Calling conventions
-------------------

A primitive's group and input/output operands precede ``/`` in its
signature and are passed positionally. Controls after ``*`` are keyword-only:

.. code-block:: python

   coop.load(group, source, items, valid_items=count, offset=offset)
   result = coop.reduce(group, items, binary_op="max")

These conventions apply to both backends. A qualified signature can add
operands or controls; check the :doc:`API reference <../coop_api>` rather than
passing a backend extension to the common namespace. Load fills its output
in place and returns ``None``; an operation that returns a new value leaves
its input payload unchanged unless its contract says otherwise.

.. _coop-backend-registration:

Registering a backend
---------------------

When you cannot ensure import order, call :func:`cuda.coop.register` on the
host before compiling kernels to activate its compiler integration
explicitly. For a Numba-CUDA-MLIR kernel:

.. code-block:: python

   from cuda import coop

   coop.register("numba-cuda-mlir")

   from numba_cuda_mlir import cuda

For a CuTe kernel:

.. code-block:: python

   from cuda import coop

   coop.register("cutlass")

   from cutlass import cute

Registration loads the selected backend and installs its compiler hooks.
Both integrations support either import order and repeated registration;
the call returns ``None``. Numba-CUDA-MLIR also accepts the spelling
``"numba_cuda_mlir"``. Registration requires the backend's dependencies to
be installed; it does not install packages.

For convenience, importing ``cuda.coop`` after a supported compiler runtime
also attempts to register its backend automatically. A standalone
``cuda.coop`` import does not discover or load optional compilers. Explicit
registration is useful in libraries and notebooks where another import may
already have loaded ``cuda.coop``.

Importing the backend namespace also registers it:

.. code-block:: python

   import cuda.coop.numba_mlir as numba_coop

Or, for CuTe kernels:

.. code-block:: python

   import cuda.coop.cutlass as cutlass_coop

The documentation reserves ``coop`` for common calls and uses ``numba_coop``
or ``cutlass_coop`` for qualified calls. See the
:ref:`namespace FAQ <coop-faq-qualified-only>` and the
:ref:`Numba <coop-programming-api-choice>` and
:ref:`CUTLASS <coop-cutlass-api-choice>` API comparisons. Both backends can be
registered in one process; the compiler tracing a kernel selects the
implementation.

Shared execution model
----------------------

.. _coop-common-groups:
.. _groups-and-thread-data:

Groups and tiles
^^^^^^^^^^^^^^^^

A group identifies the threads working together. ``this_block()`` selects
the current block; ``this_warp()`` selects a physical 32-thread warp.
``this_warp().group_by(width)`` partitions it into consecutive logical
warps of 1, 2, 4, 8, 16, or 32 threads. A group processing ``K`` items per
thread has a tile of ``group_size * K`` items. Each block or warp group
processes its own tile.

The hierarchy also describes individual threads, clusters, the grid, and mapped
groups of physical warps. These descriptors support queries such as rank,
count, and membership. Their availability does not make every primitive valid
for that group: Reduce, Load/Store, and Scan use block or warp groups, and grid
primitives are unavailable. Consult the backend guide for the supported query
levels and synchronization operations.

A multidimensional block uses x-major linear thread rank:
``x + block_x * (y + block_y * z)``. Warp primitives require an enclosing
block made of complete physical warps. For warp Load/Store, the compiler
adds the group's within-block tile origin to the memory address. A caller's
``offset`` supplies the remaining element offset, such as the block's tile
origin in a larger array; do not add the within-block group origin again.

.. _coop-common-participation:
.. _load-and-store-semantics:

Participation and valid prefixes
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every required participant must reach the same primitive invocation.
A short tile does not reduce the number of threads that must participate.
For example, ``valid_items=45`` limits a 64-item Load to its first 45 tile
positions; all threads in the group still call Load. Complete sibling
logical warps may take different paths where the operation's contract
permits it.

For Load and Store, ``valid_items`` counts tile elements, not items per
thread. It must be uniform within the group and lie between zero and the tile
capacity. Clamp a tail count before passing it. Invalid static counts are
rejected during compilation; invalid runtime counts trap on the device.
``offset`` is a nonnegative element offset, also uniform within the group.
It is independent of the valid count. Other families define their own
valid-prefix controls; for example, reduction counts contributing threads.

Load leaves invalid payload slots unspecified unless ``oob_default`` is
provided, even if those slots were initialized before Load. Store leaves
destination elements outside the valid prefix untouched. A valid prefix
controls data access; it does not make it safe to skip a required participant
or a reuse barrier.

.. _coop-common-payloads:

Per-thread payloads
^^^^^^^^^^^^^^^^^^^

``coop.ThreadData(items_per_thread)`` describes a fixed-size payload of
``items_per_thread`` values owned by each thread. Pass that count as a kernel
argument: Numba-CUDA-MLIR specializes it automatically; CuTe kernels and their
launchers declare it as ``items_per_thread: cutlass.Constexpr``. Leave the
element type unspecified for normal use: Load infers it from its source, fills
the payload in place, and returns ``None``. Other operations either consume
that payload or return a new scalar or payload according to their contract. As
in CUB, transpose Store algorithms may rearrange the input payload in place.
Copy values before Store if they are needed later.

Supported payload dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit
integers and 32- and 64-bit floating-point values. An explicit alignment is
a positive power of two in bytes and sets a minimum when payload storage
is materialized. It does not assert alignment of an input or output array.

The payload extent is a compile-time value available as
``items.items_per_thread``. Initialize every item an operation will read; Load
without ``oob_default`` leaves invalid slots unspecified. Assign those slots
after Load before reading them. Backend value rules still apply:
Numba-qualified calls can accept supported local arrays, while
CUTLASS-qualified ``ThreadData`` supports CuTe register-tensor conversions.
Supported typed assignments and cooperative producers can establish the
element type. The :ref:`element-type inference FAQ
<coop-faq-thread-data-dtype>` covers cases that need explicit information. For
backend details, see the :ref:`Numba type rules <coop-thread-data>` and
:ref:`CUTLASS payload conversions <coop-cutlass-register-payloads>`.

.. _coop-common-layouts:
.. _exchange-semantics:
.. _shuffle-semantics:
.. _scan-semantics:

Layouts and operation order
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In a :term:`blocked` layout, each thread owns consecutive tile positions.
In a :term:`striped` layout, consecutive threads own consecutive positions
at each item slot. The :ref:`layout glossary <coop-glossary-layouts>` gives
the formulas and an ownership table.

Load/Store's ``direct`` and ``vectorize`` algorithms expose blocked
payloads. ``striped`` exposes striped payloads. Transpose algorithms use
striped memory accesses and present blocked payloads to the caller.
Exchange converts between supported layouts. ``ThreadData`` does not carry
a layout tag that automatically fixes a mismatched Load/Store pair.
Shuffle shifts values within a block's tile. The
:doc:`Exchange <visualizations/exchange>` and
:doc:`Shuffle <visualizations/shuffle>` visualizations show the
common rearrangements and qualified modes.

Scan traverses the group's values in blocked tile order. A striped Load
must therefore be converted before a Scan intended to follow the array's
original order. Choosing a memory-access algorithm and choosing the order
of values seen by an operation are related but distinct decisions.

.. _coop-common-results:

Result ownership
^^^^^^^^^^^^^^^^

A primitive's return type does not say which threads may use its result.
Reduce/Sum define their result only at group rank zero. Scan returns each
thread's part of the prefix sequence. Check the operation's group and algorithm
contract before consuming a result.

Sorting and selection operate on one group's tile. Sorting each block does not
sort a whole array. TopK defines an unordered selected prefix; the remaining
payload positions are not output. See the :doc:`API reference <../coop_api>`
for sorting and selection operations.

.. _coop-common-storage:
.. _temporary-storage:

Scratch allocation and reuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The compiler allocates temporary shared storage for operations that need
it. Direct, striped, and vectorized Load/Store are storage-free: they add
no scratch pointer, shared allocation, or reuse barrier. Other algorithms
use the concrete CUB specialization's required size and alignment.

For operations that accept an explicit descriptor, construct ``TempStorage``
inside the kernel. A descriptor lets several calls reuse one region or request
capacity and minimum alignment. An undersized request is an error.
``sharing="shared"`` overlaps the uses of one descriptor;
``sharing="exclusive"`` assigns distinct call sites separate slices. With
automatic synchronization disabled, separate slices avoid barriers needed
solely for cross-call scratch reuse, at the cost of more shared memory.
Independent descriptors do not alias. See :ref:`the exclusive-storage FAQ
<coop-faq-exclusive-storage>`.

Explicit descriptors default to ``auto_sync=False`` for both sharing modes.
The kernel must provide the required barrier before reuse, including across
iterations: a call site inside a loop reuses its slice even with exclusive
storage. Set ``auto_sync=True`` to insert a trailing reuse barrier after each
storage-using call. Compiler-managed scratch, used when no descriptor is
supplied, synchronizes automatically. A scratch reuse barrier does not replace
synchronization for the kernel's own shared data.

Warp operations that need scratch keep independent storage per physical or
logical group and use the appropriate warp mask. Each primitive documents
whether it accepts explicit storage. Rules for combining cooperative scratch
with the kernel's own shared memory depend on the compiler. See
:ref:`Numba storage <coop-temp-storage>`, the
:ref:`CUTLASS storage <coop-cutlass-storage>`, and the
:ref:`storage FAQ <coop-faq-temp-storage>` for examples and limits.
