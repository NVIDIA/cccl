.. _cccl-python-coop-cutlass:

Getting Started with ``cuda.coop.cutlass``
===========================================

The CUTLASS backend lets CUTLASS DSL kernels use the portable
:mod:`cuda.coop` load and store operations. The portable root selects CUTLASS
only while its exact compiler environment manager is current. The qualified
:mod:`cuda.coop.cutlass` API names the same implementation explicitly, but its
lowering still requires a compatible CUTLASS compiler context. The
initial release supports only the DIRECT block load/store algorithm; broader
algorithms and warp groups are intentional follow-up work. It supports CUDA 13
and is tested with CUDA Toolkit 13.3.

Installation
------------

Install the ``cuda-coop`` distribution with the CUTLASS dependencies:

.. code-block:: bash

   pip install 'cuda-coop[cutlass,examples]'

The ``examples`` extra installs PyTorch for the executable example below. Code
that only uses ``cuda.coop`` from CUTLASS kernels can install
``cuda-coop[cutlass]`` instead.

An operation must be traced from a compatible compiler context. If no provider
can serve that context, ``cuda.coop`` reports the structured error below
instead of silently choosing another backend:

.. code-block:: text

   cuda.coop.<feature> requires an active compiler backend; install or import a compatible backend before compiling a kernel

CUDA thread block load and store
--------------------------------

This executable example uses one complete CUDA thread block to copy a partial
tile from contiguous, pointer-backed input. ``valid_items`` defines the valid
prefix; ``oob_default`` supplies a sentinel for each tail item. The operation's
``offset`` is measured in elements. Every thread participates even when its
items fall in the tail. Static offsets and valid counts are checked against the
operand extent. When either control is a traced runtime value, the caller must
keep the accessed range within that operand.

Statically compact row-major, column-major, and hierarchical layouts are
accepted. Load and Store traverse their raw pointers in linear storage order;
they do not apply logical multidimensional indexing or layout order.

.. literalinclude:: ../../python/cuda_coop/examples/cutlass/block_load_store.py
   :language: python
   :caption: One CUDA thread block copies a partial tile.

The load reads the valid input prefix and fills the seven payload tail items
with ``-1``. The store writes four of those fill values, then leaves the final
three destination tail items at ``-999``. Separate input and output offsets
also show that ``offset`` counts elements rather than bytes.

Use :mod:`cuda.coop.cutlass` directly when a kernel should be explicitly tied
to CUTLASS. Its ``ThreadData``, ``ThreadGroup``, ``this_block``, ``load``, and
``store`` exports have the same signatures and behavior as the automatically
activated portable root API.

See :doc:`coop_api` for the complete initial API.
