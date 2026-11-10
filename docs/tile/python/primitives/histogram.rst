.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Histogram
=========

.. py:currentmodule:: cuda.coop.tile

Histogram counts each input byte in a bin with the same integer index. A call
starts its own zeroed histogram and returns counts for that Tile only.

Interface and examples
----------------------

.. py:function:: histogram(samples, /, *, bins=256, methods=None)
   :no-contents-entry:

   Count samples into a compile-time number of bins.

.. py:function:: histogram_256(samples, /, *, methods=None)
   :no-contents-entry:

   Count samples into all 256 possible byte-value bins.

.. rubric:: Parameters and results

``samples``
   A ``uint8`` Tile of shape ``(N,)``. Each sample must satisfy
   ``0 <= sample < bins``. Out-of-range samples are not clamped or ignored.
``bins``
   A compile-time choice of 32, 64, 128, or 256 (default). The result
   extent must be a power of two. This does not by itself guarantee a
   supported topology. The
   selected block width must divide both ``N`` and ``bins``. The native
   provider uses widths that are multiples of 32, and the compiler must
   support the selected width. A 128-thread implementation requires 128
   or 256 bins and an input extent divisible by 128. A 256-thread
   implementation requires 256 bins and an input extent divisible by 256.
``methods``
   Must be omitted or empty. The current API does not expose a histogram
   algorithm selector.

.. rubric:: Example: count byte values

Each value in ``[0, 63]`` occurs four times; the remaining bins are zero.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def histogram_kernel(source, counts):
       samples = ct.load(source, (ct.bid(0),), shape=(256,))
       histogram = coop.histogram(samples, bins=256)
       ct.store(counts, (ct.bid(0),), histogram)

   source = (torch.arange(256, device="cuda") % 64).to(torch.uint8)
   counts = torch.empty(256, device="cuda", dtype=torch.int32)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), histogram_kernel, (source, counts))
   expected = torch.bincount(source.to(torch.int64), minlength=256).to(torch.int32)
   assert torch.equal(counts, expected)


Behavior
--------


The result is an ``int32`` Tile of shape ``(bins,)``. Bin ``b`` contains the
number of input elements equal to ``b``. Counts start at zero for each call;
there is no accumulation into a preexisting histogram. Their sum equals ``N``
for valid input. ``histogram_256(samples)`` is equivalent to
``histogram(samples, bins=256)``.

.. figure:: /tile/visualizations/histogram.svg
   :alt: A 256-sample byte Tile produces 256 counts with 64 occupied bins.

   The 256 input samples occupy 64 bins. All 256 output bins are present,
   including zero counts for values that do not occur.
