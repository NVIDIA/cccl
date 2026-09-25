.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _cccl-python-coop:

``cuda.coop``: Cooperative Group Primitives
===========================================

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Shared concepts

   Overview <self>
   coop/visualizations/index
   coop/glossary
   coop/faqs

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Numba-CUDA-MLIR

   coop/programming_guide
   coop/neighbor-operations
   coop/developer_overview

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: CUTLASS

   coop_cutlass
   coop/cutlass_developer_guide

``cuda.coop`` provides cooperative CUDA primitives inside Python kernels.
Threads work together to :doc:`load <coop/visualizations/load>` and
:doc:`store <coop/visualizations/store>` tiles, rearrange values with
:doc:`Exchange <coop/visualizations/exchange>` and
:doc:`Shuffle <coop/visualizations/shuffle>`, and compute
:doc:`reductions <coop/visualizations/reduce>` and
:doc:`scans <coop/visualizations/scan>` inside a kernel. They can also
:doc:`sort keys and associated values <coop/visualizations/merge-sort>` within a group or
compute :doc:`radix sorts and digit ranks <coop/visualizations/radix>` within a block.
:doc:`TopK <coop/visualizations/topk>` selects a block's smallest or largest keys without
sorting the full tile. Blocks can compare neighboring items with
:doc:`Adjacent Difference <coop/visualizations/adjacent-difference>` and
:doc:`Discontinuity <coop/visualizations/discontinuity>`, count samples with
:doc:`Histogram <coop/visualizations/histogram>`, or expand compressed runs
with :doc:`Run Length Decode <coop/visualizations/run-length-decode>`.
:doc:`Batched Warp Reduction <coop/visualizations/reduce-batched>` computes
an independent reduction for each per-thread payload slot.
Both compiler integrations implement these common operations using CUB and
CUDAX; see :ref:`backend coverage <coop-backends>`.

This overview introduces the shared API and execution model. Choose a
programming guide to write kernels, or a developer guide to work on the
compiler integration:

.. list-table:: Backend guides
   :header-rows: 1
   :widths: 20 40 40

   * - Backend
     - Writing kernels
     - Working on the integration
   * - Numba-CUDA-MLIR
     - :doc:`Programming Guide <coop/programming_guide>`
     - :doc:`Developer Guide <coop/developer_overview>`
   * - CUTLASS / CuTe DSL
     - :doc:`Programming Guide <coop_cutlass>`
     - :doc:`Developer Guide <coop/cutlass_developer_guide>`

The :doc:`visualizations <coop/visualizations/index>` explain how values
move. The :doc:`glossary <coop/glossary>` and :doc:`FAQs <coop/faqs>` cover
terminology and common questions. Signatures and return contracts live in
the :doc:`API reference <coop_api>`.

.. _coop-backends:

Backend coverage
----------------

The common API is the contract shared by Numba-CUDA-MLIR and CUTLASS. Both
implement every common primitive below, with the documented groups, dtypes,
and result rules. Their qualified APIs add compiler-specific payloads and
controls, described in the programming guides.

.. list-table:: Current primitive families
   :header-rows: 1
   :widths: 48 26 26

   * - Family
     - Numba-CUDA-MLIR
     - CUTLASS
   * - Group queries and supported synchronization
     - Available
     - Available
   * - Block and warp Load/Store
     - Available
     - Available
   * - Built-in Reduce/Sum and Scan
     - Available
     - Available
   * - Block and warp Exchange; block Shuffle
     - Available
     - Available
   * - Merge Sort, keys and pairs
     - Available
     - Available
   * - Radix Sort, keys and pairs; Radix Rank
     - Available
     - Available
   * - TopK, minimum and maximum keys or pairs
     - Available
     - Available
   * - Adjacent Difference and Discontinuity
     - Available
     - Available
   * - Histogram
     - Available
     - Available
   * - Run Length Decode, windowed and bulk
     - Available
     - Available
   * - Batched Warp Reduction
     - Available
     - Available

.. _block-prefix-callbacks:

Numba-CUDA-MLIR additionally supports qualified device operators and
:ref:`Scan prefix callbacks <coop-prefix-callbacks>`. CUTLASS currently
supports built-in operators; custom
operators and stateful Scan callbacks are outside its current scope.
CUTLASS qualification covers a compatible Linux/CUDA 13 environment. An
official public runtime artifact has not yet passed consumer qualification;
see its programming guide before selecting a runtime.

.. raw:: html

   <span id="coop-common-api"></span>

.. _coop-api-namespaces:
.. _kernel-api:

.. _portable-and-qualified-apis:

Common and qualified APIs
-------------------------

``from cuda import coop`` selects the common namespace. It defines the
shared contracts for groups, payloads, operations, and temporary storage
across supported DSLs. Calls inside a kernel are compiler markers; they are not
host-side implementations of those operations.

The qualified namespaces, ``cuda.coop.numba_mlir`` and
``cuda.coop.cutlass``, each include all common kernel operations and their
backend's extensions. A program using one compiler can use its qualified
namespace alone. CUTLASS-only code can use ``import cuda.coop.cutlass as coop``. Use
``numba_coop`` and ``cutlass_coop`` when a module contains both DSLs.
Common and qualified calls can appear in the same kernel when they belong
to its compiler. The comparisons in the
:ref:`Numba-CUDA-MLIR guide <coop-programming-api-choice>` and
:ref:`CUTLASS guide <coop-cutlass-api-choice>` list the differences.

A common call preserves its operation's contract across backends. Moving
a kernel still requires adapting launch syntax, array arguments, control
flow, and other DSL code. Compiler-owned payloads cannot be passed between
Numba and CuTe traces.

.. _coop-common-calling-conventions:

Calling conventions
-------------------

A primitive's group and input/output operands precede ``/`` in its
signature and are passed positionally. Controls after ``*`` are keyword-only:

.. code-block:: python

   coop.load(group, source, items, valid_items=count, offset=offset)
   result = coop.reduce(group, items, binary_op="max")

These conventions apply to both backends. A qualified signature can add
operands or controls; check the :doc:`API reference <coop_api>` rather than
passing a backend extension to the common namespace. Load fills its output
in place and returns ``None``; an operation that returns a new value leaves
its input payload unchanged unless its contract says otherwise.

Installation
------------

Install ``cuda-coop`` without adding Python package dependencies:

.. code-block:: console

   python -m pip install cuda-coop

The wheel includes the common API, every shipped DSL integration (including
``cuda.coop.numba_mlir`` and ``cuda.coop.cutlass``), type declarations, and
a matching bundle of CUB,
Thrust, libcu++, and CUDAX headers. The base install declares no Python
package dependencies. You can import ``cuda.coop`` without a compiler or GPU;
using an integration requires its backend dependencies to be installed.

For Numba-CUDA-MLIR, install the extra matching your CUDA major version:

.. code-block:: console

   python -m pip install "cuda-coop[numba-cuda-mlir-cu13]"
   # Use numba-cuda-mlir-cu12 with CUDA 12.

Both commands install the same ``cuda-coop`` wheel with the same DSL
integrations. The extra only adds the dependency requirements declared in
``pyproject.toml`` so pip installs the supported Numba-CUDA-MLIR stack for
the selected CUDA major version. The current integration requires
``numba-cuda-mlir>=0.5.0,<0.6``.
For CUTLASS / CuTe DSL, install the base wheel alongside a runtime meeting
the :ref:`CUTLASS requirements <coop-cutlass-requirements>`. A public CUTLASS
extra and minimum version await qualification of an official artifact.

Installing an extra does not register a backend in a running Python process;
see :ref:`installation versus registration <coop-faq-installed-extra>`.

Installed-wheel compilation uses the bundled CCCL headers. Development from a
CCCL source checkout uses its matching headers. ``CUDA_COOP_CCCL_ROOT`` can
select another source checkout or ``cuda-coop`` header bundle.

.. _coop-numba-validation:

Numba-CUDA-MLIR validation scope
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The package supports Python 3.10 through 3.14. The CI matrix configures these
parts of that range:

.. list-table::
   :header-rows: 1
   :widths: 36 64

   * - Environment
     - Automated checks
   * - Linux x86-64, Python 3.14, CUDA 13
     - Installed-wheel compilation with GPUs hidden and L4 runtime tests
       in pull requests; H100 runtime tests with serial synchronization
       race checking in the nightly matrix
   * - Linux x86-64, Python 3.14, CUDA 12
     - Installed-wheel compilation and GPU runtime tests in the nightly matrix
   * - Linux x86-64, Python 3.10 and 3.14
     - Common API host contracts and wheel packaging
   * - Windows x86-64, Python 3.10 and 3.14
     - Universal-wheel build, base import, and bundled-header checks

The Windows checks do not compile or launch Numba-CUDA-MLIR kernels. Other
Python versions and platform combinations need separate backend runtime
qualification. Dependency bounds allow releases in the supported series;
they do not mean that every patch release in that series has been tested.
Thread-block clusters require a CC 9.0+ GPU, and synchronization race
checking requires Compute Sanitizer. A runtime job that skips those tests
does not qualify those features.

.. _coop-numba-context-lifetime:

CUDA devices and context lifetime
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

With Numba-CUDA-MLIR 0.5.0 through 0.5.3, select the CUDA device before a
kernel's first compilation and keep its dispatcher and configured launch
callables in that CUDA context. Reusing them on another device or after
destroying and recreating the context is not qualified: the compiler can
reuse architecture, compiled overload, or launch state from the original
context. This also applies to kernels that use ``cuda.coop``.

The upstream `context-isolation fix
<https://github.com/NVIDIA/numba-cuda-mlir/pull/314>`_ must be released and
qualified with this integration before relying on that reuse. Switching
devices does not require re-registering ``cuda.coop``; registration installs
compiler hooks and does not repair dispatcher context state.

.. _coop-backend-registration:

Registering a backend
---------------------

Call :func:`cuda.coop.register` on the host before compiling kernels to
activate its compiler integration explicitly. For a Numba-CUDA-MLIR kernel:

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

The hierarchy also describes individual threads, clusters, the grid, and
mapped groups of physical warps. These descriptors support queries such as
rank, count, and membership. Their availability does not make every
primitive valid for that group: built-in Reduce supports more group forms
than Load/Store or Scan, and grid primitives are unavailable. Consult the
backend guide for the supported query levels and synchronization operations.

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

Load leaves invalid payload slots unchanged unless ``oob_default`` is
provided. Store leaves destination elements outside the valid prefix
untouched. A valid prefix controls data access; it does not make it safe
to skip a required participant or a reuse barrier.

.. _coop-common-payloads:

Per-thread payloads
^^^^^^^^^^^^^^^^^^^

``ThreadData(K, dtype=None, *, alignment=None)`` describes a fixed-size
payload of ``K`` values owned by each thread. Load fills the supplied
payload in place and returns ``None``. Its source can establish the dtype
of an untyped output. Other operations either consume that payload or
return a new scalar or payload according to their contract. Store preserves
its input.

Supported payload dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit
integers and 32- and 64-bit floating-point values. An explicit alignment is
a positive power of two in bytes and sets a minimum when payload storage
is materialized. It does not assert alignment of an input or output array.

The payload extent is a compile-time value available as
``items.items_per_thread``. Initialize every item an operation will read;
Load without ``oob_default`` does not initialize invalid slots. Backend
value rules still apply: Numba-qualified calls can accept supported local
arrays, while CUTLASS-qualified ``ThreadData`` supports CuTe register-tensor
conversions. A dtype selector describes the element representation; the
active compiler still owns the scalar values. For example, selecting a
NumPy dtype in a CuTe kernel produces CuTe values. See the
:ref:`Numba type rules <coop-thread-data>` and
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
:doc:`Exchange <coop/visualizations/exchange>` and
:doc:`Shuffle <coop/visualizations/shuffle>` visualizations show the
common rearrangements and qualified modes.

Scan traverses the group's values in blocked tile order. A striped Load
must therefore be converted before a Scan intended to follow the array's
original order. Choosing a memory-access algorithm and choosing the order
of values seen by an operation are related but distinct decisions.

.. _coop-common-results:

Result ownership
^^^^^^^^^^^^^^^^

A primitive's return type does not say which threads may use its result.
The default full-group Reduce/Sum profile broadcasts its result to group
members. A valid-prefix reduction or ``broadcast=False`` defines the result
at group rank zero. Scan returns each thread's part of the prefix sequence.
Check the operation's group and algorithm contract before consuming a result.

Sorting and selection operate on one group's tile. Sorting each block does
not sort a whole array. TopK defines an unordered selected prefix; the
remaining payload positions are not output. See
:ref:`backend coverage <coop-backends>` for the available sorting and
selection families.

.. _coop-common-storage:
.. _temporary-storage:

Scratch allocation and reuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The compiler allocates temporary shared storage for operations that need
it. Direct, striped, and vectorized Load/Store are storage-free: they add
no scratch pointer, shared allocation, or reuse barrier. Other algorithms
use the concrete CUB specialization's required size and alignment.

For operations that accept an explicit descriptor, construct
``TempStorage`` inside the kernel. A descriptor lets several calls reuse
one region or request capacity and minimum alignment. An undersized
request is an error. ``sharing="shared"`` overlaps the uses of one
descriptor; ``sharing="exclusive"`` assigns distinct call sites separate
slices. Independent descriptors do not alias.

Automatic trailing synchronization defaults to enabled for both sharing
modes. A call site inside a loop reuses its slice even with exclusive
storage. With ``auto_sync=False``, the kernel must provide the required
barrier before reuse, including across iterations. A scratch reuse barrier
does not replace synchronization for the kernel's own shared data.

Warp operations that need scratch keep independent storage per physical or
logical group and use the appropriate warp mask. Each primitive documents
whether it accepts explicit storage. Rules for combining cooperative scratch
with the kernel's own shared memory depend on the compiler. See
:ref:`Numba storage <coop-temp-storage>`, the
:ref:`CUTLASS storage <coop-cutlass-storage>`, and the
:ref:`storage FAQ <coop-faq-temp-storage>` for examples and limits.

Configuration
-------------

Runtime environment variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For the Boolean switches below, *truthy* means any value other than the
empty string, ``0``, ``false``, ``no``, or ``off``. For example, ``1``,
``true``, ``yes``, and ``on`` are all truthy. Values are case-insensitive,
and leading and trailing whitespace is ignored. An unset variable is false.

``CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION``
   A truthy value disables automatic backend activation during
   :mod:`cuda.coop` import. Explicit ``coop.register(...)`` and
   backend imports still work.

``CUDA_COOP_CCCL_ROOT``
   Selects a CCCL source checkout or a ``cuda-coop`` header bundle. An invalid
   configured root is an error; resolution does not fall back to another CCCL
   source.

``CUDA_COOP_ENABLE_CACHE``
   A truthy value enables the Numba-CUDA-MLIR persistent compiler cache.
   Read when the Numba backend cache module is imported.

``XDG_CACHE_HOME``
   For the Numba backend on Linux and other POSIX systems, sets the cache
   base directory; entries are stored in ``<value>/cccl``. Unset, empty, or
   relative values fall back to ``~/.cache/cccl``. Read when the backend
   cache module is imported.

``LOCALAPPDATA``
   For the Numba backend on Windows, sets the cache base directory; entries
   are stored in ``<value>\cccl``. Unset, empty, or relative values fall back to
   ``~\AppData\Local\cccl``. Read when the backend cache module is imported.

``CUDA_COOP_CUTLASS_PROVIDER_CACHE_DIR``
   Selects the CUTLASS provider artifact cache directory. The default is a
   user-specific directory under the system temporary directory. See the
   :doc:`CUTLASS Developer Guide <coop/cutlass_developer_guide>` for artifact
   lifetime and cache validation.

``CUDA_COOP_SOURCE_DUMP_DIR``
   Writes generated CUDA source to this directory for compiler diagnostics.
   Files use ``cuda_coop_<backend>_<hash>.cu`` names so different backends can
   share a directory. Set it before compiling; both backends also write
   the source when their provider compilation cache is hit. Unset or empty
   disables dumping.

``CUDA_PATH``
   Supplies ``<value>/include`` as a CUDA header candidate if
   ``cuda-pathfinder`` does not resolve one.

``CUDA_HOME``
   Supplies ``<value>/include`` after ``CUDA_PATH`` under the same fallback
   rule.

``CUDA_ROOT``
   Supplies ``<value>/include`` after ``CUDA_HOME`` under the same fallback
   rule.

On Linux and other POSIX systems, ``/usr/local/cuda/include`` is tried last.
Windows uses ``cuda-pathfinder`` or the configured toolkit roots above; it
does not try the Unix fallback. If no valid CUDA include directory is found,
compilation reports a header-resolution error.

Build-time CMake variables
^^^^^^^^^^^^^^^^^^^^^^^^^^

``CUDA_COOP_INSTALL_HEADER_BUNDLE``
   Defaults to ``ON``. Installs the private CCCL header and CMake-package
   bundle into the wheel.

``CUDA_COOP_ALLOW_DIRTY_HEADER_BUNDLE``
   Defaults to ``OFF``. Allows a Git-worktree bundle when selected inputs are
   changed or ``git status`` cannot verify them, and records its source
   revision as ``unknown``.

``CUDA_COOP_CCCL_SOURCE_REVISION``
   Defaults to empty. Supplies the revision token recorded instead of deriving
   it from Git. A dirty or unverifiable Git worktree still records ``unknown``.

Compilation and headers
-----------------------

``cuda-coop`` compiles providers against its configured CCCL root, the active
source checkout during in-tree development, or its installed header bundle,
in that order. It never substitutes the CUDA Toolkit's copy of CUB. CUDA
headers and compiler/linker libraries must resolve to a compatible toolkit.
Shared planner decisions describe the primitive; each backend adapts its
compiler's values and lifecycle to those decisions.

Follow a kernel through the implementation in the
:doc:`Numba-CUDA-MLIR Developer Guide <coop/developer_overview>` or
:doc:`CUTLASS Developer Guide <coop/cutlass_developer_guide>`.
