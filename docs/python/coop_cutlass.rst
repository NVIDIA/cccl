.. _coop-cutlass:

``cuda.coop.cutlass``: CuTe DSL integration
===========================================

The CUTLASS backend provides Block Load and Store with ``algorithm="direct"``
inside CuTe DSL kernels. It uses the same group-first calls and in-place
payload contract as :mod:`cuda.coop`: ``load`` fills an existing
``ThreadData`` and returns ``None``; ``store`` leaves its input payload
unchanged. Other primitive families and Warp operations are not yet available
through this backend.

Runtime requirements
--------------------

The base ``cuda-coop`` wheel includes this optional backend. Its initial
development target is Linux with CUDA 13. A supported public CUTLASS package
version has not yet been qualified, so ``cuda-coop`` does not provide a
CUTLASS installation extra.

The dependency-free base wheel does not install compiler prerequisites. A
CUTLASS environment also needs NumPy, ``cuda-pathfinder>=1.2.3``, and
``typing_extensions>=4.12.0`` for runtime discovery and type declarations,
alongside a compatible CuTe compiler and its dependencies.

A compatible CuTe compiler must provide scoped trace finalization, active
compiler-environment ownership, exact launch dimensions and flags, and
external NVIDIA LTO-IR linking. Successful import checks the Python
capabilities; compiling and running a kernel also requires a working NVRTC
and terminal linker. Importing :mod:`cuda.coop` alone does not load CUTLASS
or initialize CUDA bindings.

Activation and example
----------------------

Register CUTLASS on the host before compiling:

.. code-block:: python

   from cuda import coop

   coop.register("cutlass")

   import cutlass.cute as cute

Registration works in either import order, is safe to repeat, and remains
available when automatic registration is disabled. Importing
``cuda.coop.cutlass`` also registers the backend. For convenience, importing
``cuda.coop`` after ``cutlass`` activates it automatically.

The active CuTe compiler selects the backend while tracing a kernel. A
CUTLASS installation alone does not make portable operations callable on the
host. A failed optional activation reports the missing capability and leaves
the portable namespace available.

This example loads two adjacent items per thread and stores a partial tile in
the same blocked layout. ``module`` selects the portable or qualified API. The
full example defines the tile dimensions and checks the output against a CPU
reference. :download:`Download the example
<../../python/cuda_coop/examples/cutlass/block_load_store.py>` to run it with a
compatible compiler:

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
   :language: python
   :start-after: docs: start cutlass-block-load-store
   :end-before: docs: end cutlass-block-load-store

Block Load and Store use the exact launch dimensions supplied by CuTe,
including multidimensional blocks. ``offset`` selects the beginning of the
block's tile and ``valid_items`` specifies the number of valid items in that
tile. Load
may fill its out-of-bounds items with ``oob_default``. Without that default,
initialize any items that the valid prefix will not overwrite before reading
them. All threads in the block must call the operation with uniform controls.

DIRECT Load and Store require no temporary shared storage. They emit no
storage pointer or reuse barrier. ``ThreadData(alignment=...)`` requests a
minimum payload alignment; it does not change the logical item layout.

Qualified register payloads
---------------------------

Import ``cuda.coop.cutlass`` as ``coop`` when a kernel needs CuTe register
conversions. The qualified ``ThreadData`` provides ``from_register_tensor``
and ``to_register_tensor`` for register-memory tensors, and ``from_vector``
and ``to_tensor_ssa`` for immutable register values. These conversions use the
same fixed per-thread item count as the load/store payload.

An initialized ``ThreadData`` can cross CuTe runtime loops and branches while
retaining its fixed item count, dtype, and requested alignment. Initialize
every item in every participating thread before carrying the payload across
a runtime control-flow boundary.

.. code-block:: python

   import cuda.coop.cutlass as coop

   # Inside a CuTe kernel, with a register-memory fragment:
   values = coop.ThreadData.from_register_tensor(fragment)
   coop.store(coop.this_block(), destination, values)

Keep compiler-owned payloads within their originating DSL. Separate kernels
may use CUTLASS and Numba-CUDA-MLIR in the same process; their register values
and type systems are not interchangeable.
