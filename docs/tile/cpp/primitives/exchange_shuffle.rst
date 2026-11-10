.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Exchange and shuffle
====================

Exchange permutes logical Tile positions between striped and blocked order.
Shuffle reads a neighboring position, with either a preserved boundary or a
wraparound. Each call returns a new Tile and preserves the input Tile's value.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

All six operations return a numeric Tile with the input's positive extent
``N`` and element type. Up and down preserve the first and last boundary,
respectively; offset preserves positions whose source is out of range, while
rotate wraps. Require ``-N < distance < N`` for offset and
``0 <= distance < N`` for rotate. Both default to distance 1. Exchange's
explicit width must divide ``N``; changing it changes the permutation.

Exchange requires ``exchange_policy<B>`` as the second argument.
Shuffle accepts an optional empty ``shuffle_policy<>``. Its physical
participation is selected from the logical extent. Offset and rotate
require an implementation with ``N`` workers and accept either an integer
scalar or a shape-one ``int`` Tile as distance.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto blocked = coop::exchange_striped_to_blocked(input, coop::exchange_policy<128>{});
   auto original = coop::exchange_blocked_to_striped(blocked, coop::exchange_policy<128>{});
   auto previous = coop::shuffle_up(input);
   // A separate 128-element input is required for this rotation.
   auto rotated = coop::shuffle_rotate(input128, 3);


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
implementations require one worker for every logical item. The input extent
must be 32, 64, 128, 256, 512, or 1024, and the kernel must admit that worker count.
This is an implementation restriction; callers do not pass an items-per-thread
option. For example, with ``[0, 1, 2, 3]`` and distance 1, offset gives ``[1, 2, 3, 3]`` and
rotate gives ``[1, 2, 3, 0]``. Up gives ``[0, 0, 1, 2]``.

.. figure:: /tile/visualizations/exchange-shuffle.svg
   :alt: Striped-to-blocked permutation and neighbor shifts with preserved or wrapped boundaries.

   Exchange shows 256 logical items with 128 block threads. Shuffle uses
   128 items and 128 threads. Ellipses abbreviate each full Tile.
