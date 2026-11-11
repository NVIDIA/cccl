.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Exchange and shuffle
====================

.. py:currentmodule:: cuda.coop.tile

Exchange permutes logical Tile positions between striped and blocked order.
Shuffle reads a neighboring position, with either a preserved boundary or a
wraparound. Each call returns a new Tile and preserves the input Tile's value.

Interface and examples
----------------------

.. py:function:: exchange_striped_to_blocked(value, /, *, block_threads, methods=None)
   :no-contents-entry:

   Gather each logical thread's striped items into a contiguous segment.

.. py:function:: exchange_blocked_to_striped(value, /, *, block_threads, methods=None)
   :no-contents-entry:

   Interleave the contiguous per-thread segments into striped order.

.. py:function:: shuffle_up(value, /, *, methods=None)
   :no-contents-entry:

   Read the preceding element; preserve the first element.

.. py:function:: shuffle_down(value, /, *, methods=None)
   :no-contents-entry:

   Read the following element; preserve the last element.

.. py:function:: shuffle_offset(value, /, *, distance=1, methods=None)
   :no-contents-entry:

   Read position ``i + distance`` when it is in range; otherwise preserve position ``i``.

.. py:function:: shuffle_rotate(value, /, *, distance=1, methods=None)
   :no-contents-entry:

   Read position ``(i + distance) % N`` with wraparound.

.. rubric:: Parameters and results

``value``
   A numeric Tile of shape ``(N,)``. The result retains its shape and dtype.
``block_threads``
   Required for exchange. A compile-time integer ``B`` that is a multiple of
   32 in ``[32, 1024]`` and divides ``N``. ``I = N / B`` is the number of items
   assigned to each logical thread for this permutation. The launch must
   support the chosen width.
``distance``
   Used by ``shuffle_offset`` and ``shuffle_rotate``, with a default of 1.
   It may be a compile-time integer or runtime scalar/single-element Tile.
   Offset requires ``-N < distance < N``. Rotate requires
   ``0 <= distance < N``. A positive distance reads from a higher index, so
   data moves toward lower output positions.
``methods``
   Must be omitted or empty for all six functions.

.. rubric:: Example: inspect exchange and its inverse

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def exchange_kernel(source, permuted, restored):
       values = ct.load(source, (ct.bid(0),), shape=(256,))
       blocked = coop.exchange_striped_to_blocked(values, block_threads=128)
       original = coop.exchange_blocked_to_striped(blocked, block_threads=128)
       ct.store(permuted, (ct.bid(0),), blocked)
       ct.store(restored, (ct.bid(0),), original)

   source = torch.arange(256, device="cuda", dtype=torch.int32)
   permuted = torch.empty_like(source)
   restored = torch.empty_like(source)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), exchange_kernel,
             (source, permuted, restored))
   expected = source.reshape(2, 128).transpose(0, 1).contiguous().flatten()
   assert torch.equal(permuted, expected)
   assert torch.equal(restored, source)

For a 128-element Tile, ``coop.shuffle_rotate(values, distance=3)`` gives
``output[i] = values[(i + 3) % 128]``. Use four worker warps for that example.
Unlike exchange, shuffle does not need a ``block_threads`` keyword.


Behavior
--------


Exchange is precisely the following permutation, for ``0 <= t < B`` and
``0 <= j < I``:

.. code-block:: text

   striped_to_blocked: output[t * I + j] = input[j * B + t]
   blocked_to_striped: output[j * B + t] = input[t * I + j]

The two operations are inverses when they use the same ``B``. If ``I = 1``,
both leave the logical sequence unchanged. Choosing a different width changes
the permutation, even if the total number of items is unchanged.

``shuffle_up`` and ``shuffle_down`` shift by exactly one item. They support
multiple items per thread. The current ``shuffle_offset`` and ``shuffle_rotate``
implementations require one worker for every logical item: the Python input
extent must be 32, 64, 128, 256, 512, or 1024, and the kernel must admit that worker count.
This is an implementation restriction; callers do not pass an items-per-thread
option. For example, with ``[0, 1, 2, 3]`` and distance 1, offset gives ``[1, 2, 3, 3]`` and
rotate gives ``[1, 2, 3, 0]``. Up gives ``[0, 0, 1, 2]``.

.. figure:: /tile/visualizations/exchange-shuffle.svg
   :alt: Striped-to-blocked permutation and neighbor shifts with preserved or wrapped boundaries.

   Exchange shows 256 logical items with 128 block threads. Shuffle uses
   128 items and 128 threads. Ellipses abbreviate each full Tile.
