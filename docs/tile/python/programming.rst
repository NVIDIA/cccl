.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Programming guide
=================

This guide uses ``cuda.coop.tile`` inside cuTile Python kernels. Set up the
:doc:`environment </tile/python/installation>` first. The examples use
``coop`` for the cooperative interface and ``ct`` for the Tile language:

.. code-block:: python

   import cuda.coop.tile as coop
   import cuda.tile as ct

Cooperative calls belong inside ``@ct.kernel`` functions. Accessing a
primitive installs its cuTile lowering; an ordinary host call does not
execute the operation on an array.

.. rubric:: A first kernel: prefix sums within tiles

An exclusive sum places the sum of preceding elements at each position.
For input ``[2, 3, 5, 7]``, the result starts ``[0, 2, 5, 10]``.
The kernel below computes a separate exclusive sum for each 256-element
partition of a device array:

.. literalinclude:: ../../../python/cuda_coop_tile/examples/block_scan.py
   :language: python
   :start-at: import cuda.tile as ct

``ct.bid(0)`` is a logical block index. The shaped load uses that index as
a tile coordinate, so block 0 reads elements 0 through 255 and block 1 reads
elements 256 through 511. ``coop.exclusive_sum`` preserves the tile's shape
and dtype, and the store writes the corresponding output partition.

The example processes complete tiles. Pass contiguous one-dimensional
device arrays with matching dtypes and a length divisible by ``TILE_ITEMS``.
Launch one logical block per partition. Skip the launch for an empty input.
See :doc:`/tile/programming_model` before adding a partial final tile.

.. rubric:: Launch and check the result

The following host code is the example's runtime test. From
``python/cuda_coop_tile``, import PyTorch as ``torch`` and run:

.. literalinclude:: ../../../python/cuda_coop_tile/tests/integration/examples/test_tile_examples.py
   :language: python
   :start-after: # docs-example-begin
   :end-before: # docs-example-end
   :dedent: 4

The two input rows contain different values. Comparing each output row
with its own prefix sum checks both the collective and the tile-coordinate
indexing. ``torch.cuda.current_stream()`` supplies the launch stream;
synchronization makes device errors and results visible to the host check.

Download the :download:`kernel and launch helper
<../../../python/cuda_coop_tile/examples/block_scan.py>` or run the packaged
example test as described in :doc:`/tile/development`.

.. rubric:: Scan controls

An exclusive sum starts at zero by default. Supply ``prefix_value`` to add
a block prefix. For a different built-in operation, use ``exclusive_scan``
or ``inclusive_scan`` and name the operation:

.. code-block:: python

   exclusive = coop.exclusive_sum(values, prefix_value=block_prefix)
   maximum = coop.inclusive_scan(values, scan_op="max")
   product = coop.exclusive_scan(values, scan_op="*", initial_value=1)

These expressions assume ``values`` is a supported one-dimensional Tile.
``block_prefix`` may be a supported runtime control. The operation name
and an optional ``algorithm`` are compile-time choices.

Exclusive operations other than addition require an initial or prefix
value. Do not supply both. Inclusive scans reject ``initial_value``; use
``prefix_value`` when the first element should combine with a block prefix.
The :doc:`reference </tile/python/api>` lists supported controls and algorithm names.

.. rubric:: Reduction returns a tile

``coop.sum(values)`` and ``coop.reduce(values, binary_op="min")`` reduce
all elements to a scalar Tile with shape ``()``. Use ``axis`` to reduce
selected dimensions and ``keepdims=True`` to retain them with extent one.
For example, ``coop.sum(matrix, axis=1, keepdims=True)`` maps shape
``(8, 128)`` to ``(8, 1)`` in one provider call.

The following complete test body reduces two separate input partitions.
It reuses the imports above and requires ``import torch``:

.. literalinclude:: ../../../python/cuda_coop_tile/tests/backends/tile_python/runtime/test_reduce.py
   :language: python
   :start-at:     num_items = 128
   :end-before: @pytest.mark.parametrize
   :dedent: 4

This example uses ``broadcast=True`` to repeat the aggregate across
the input shape. The output arrays deliberately retain every repeated
aggregate. Omit that option when the consumer needs only the aggregate;
reduced results also broadcast in later Tile expressions. Each logical
block reduces its own input partition.

.. rubric:: Choosing worker warps

Use ``@ct.kernel(num_worker_warps=8)`` to request eight worker warps
for a kernel. With TileIR 13.5, cuTile Python accepts powers of two
from 1 through 32. This setting controls the physical workers for the
whole kernel. A collective may use a subset of those workers.
Every cooperative call must have an implementation compatible
with that choice. An explicit ``block_threads=256`` requires eight
worker warps.

CUDA 13.5 automatic worker selection cannot combine 64-item and 512-item
Scan calls in the same kernel. Use ``@ct.kernel(num_worker_warps=2)`` for
that combination.

.. rubric:: Moving values

Exchange applies a permutation parameterized by an explicit physical
thread count:

.. code-block:: python

   blocked = coop.exchange_striped_to_blocked(values, block_threads=128)
   original = coop.exchange_blocked_to_striped(blocked, block_threads=128)

For a compatible extent, these calls are inverses. Changing the second
count changes its permutation. The :doc:`programming model
</tile/programming_model>` gives the index formulas and shows why the count
belongs to the operation's meaning.

Use Shuffle to move neighboring values, Adjacent Difference to calculate
neighbor differences, or Discontinuity to mark run boundaries. Each family
defines its boundary positions separately; read those contracts before
using the first or last element of a result.

.. rubric:: Composing with matrix multiplication

The :download:`MMA and TopK example
<../../../python/cuda_coop_tile/examples/mma_topk.py>` computes a matrix tile,
extracts each row, and keeps candidate scores together with their column
indices:

.. literalinclude:: ../../../python/cuda_coop_tile/examples/mma_topk.py
   :language: python
   :start-at:     global_columns =
   :end-before:         local_scores =
   :dedent: 4

The row extraction produces the one-dimensional input required by TopK.
The selected keys and column indices keep their association. Only the
selected prefix is meaningful, so the example extracts that prefix before
storing the candidates. Selection does not sort it.

The complete example uses a second kernel to select the final candidates
from several matrix tiles. Both kernels request four worker warps. The
merge kernel loads its 32 candidates into a 128-element tile with zero
padding. It passes the actual candidate count as ``valid_items``, so padded
positions cannot enter the selection. This keeps the merge extent
compatible with the worker count without changing the candidate set.
Check :doc:`/tile/development` before running the composition.

.. rubric:: Diagnosing a compilation failure

Start with the dtype, extent, and compile-time controls named in the
diagnostic. For pair operations, check both input shapes. For composed
kernels, check that explicit Exchange or worker-width constraints are
compatible across calls.

If the failure occurs while loading or compiling the provider, verify the
installed wheel set and TileLib identity. :doc:`/tile/interop` explains provider
traces and the information needed for a reproducible compiler failure.
