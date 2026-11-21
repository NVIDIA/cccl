.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Composing Tile operations
=========================

.. _tile-python-mma-topk:

Select the highest scores in each matrix row
--------------------------------------------

The :download:`MMA and TopK example
<../../../python/cuda_coop_tile/examples/mma_topk.py>` finds the eight best
columns per row of a 512-column matrix product. Each logical block computes
16 rows by 128 columns with ``ct.mma``. Its accumulator stays in the kernel
while the block selects candidates from each row.

.. literalinclude:: ../../../python/cuda_coop_tile/examples/mma_topk.py
   :language: python
   :start-at:     global_columns =
   :end-before:         global_row =
   :dedent: 4

``ct.extract`` and ``ct.reshape`` turn one accumulator row into the
one-dimensional tile that :py:func:`cuda.coop.tile.topk_max_pairs` takes.
The second input gives each score its original column index. Only the
selected prefix contains candidates, so the kernel extracts the first
eight score/index pairs for storage. Their order is unspecified.

Each row spans four column tiles. A second kernel reads the resulting
32 candidates and selects the final eight. A score excluded from a tile's
local top eight can be omitted from the global result as well. Equal scores
at the cutoff can yield different valid column indices.

.. literalinclude:: ../../../python/cuda_coop_tile/examples/mma_topk.py
   :language: python
   :start-at: candidate_scores = ct.reshape(
   :end-at:     final_columns = ct.extract
   :dedent: 4

The merge kernel loads the 32 candidates into a 128-element tile with zero
padding. ``valid_items=partial_scores.shape[1]`` excludes the 96 padded slots,
including when every candidate score is negative. Only eight pairs per row
reach the final output. The :doc:`TopK reference <primitives/topk>` specifies
the valid prefix and tie behavior.

Run the downloaded ``mma_topk.py`` in the same :doc:`environment <installation>`
as the scan example:

.. code-block:: console

   $ python mma_topk.py
   Checked 8 scores and column indices for each of 16 rows.

The inputs give each row 512 distinct negative scores, with winners in all
four column tiles. The program checks both scores and column indices against
PyTorch's TopK of the full matrix product. It compares the selected pairs
independently of their order.

Choose compatible worker counts
-------------------------------

Leave the worker count at its default when the compiler can select a
configuration for the whole kernel. Use ``@ct.kernel(num_worker_warps=4)``
to request four warps. Both kernels in the matrix example use this setting.
With TileIR 13.5, cuTile Python accepts 1, 2, 4, 8, 16, or 32 worker warps.
The input extent of a collective still counts logical elements. A primitive
may use a subset of the physical workers.

Every foreign call in one kernel needs an implementation compatible with
the same worker count. Exchange is especially sensitive to width because
its ``block_threads`` argument also determines the permutation. See
:doc:`primitives/exchange_shuffle` for the index formulas and a checked round
trip. On CUDA 13.5, automatic worker selection cannot combine 64-item and
512-item scans in one kernel. Set ``num_worker_warps=2`` for that combination.

For a compiler diagnostic, check the reported dtype, tile extent, and
compile-time controls against the :doc:`api`. Pair operations need matching
input shapes. For provider loading or compilation failures, see the
:doc:`interoperation guide </tile/interop>`.
