.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Histogram
=========

.. py:currentmodule:: cuda.coop.tile

Histogram counts each input byte in a bin with the same integer index. A call
starts its own zeroed histogram and returns counts for that Tile only.

.. figure:: /tile/visualizations/histogram.svg
   :alt: A 256-sample byte Tile produces 256 counts with 64 occupied bins.

   Four occurrences of each byte from 0 to 63 give a count of four in
   those bins. Bins 64 through 255 have a count of zero.

Interface and examples
----------------------

.. py:function:: histogram(samples, /, *, bins=256, methods=None)
   :no-contents-entry:

   Count samples into a compile-time number of bins.

.. py:function:: histogram_256(samples, /, *, methods=None)
   :no-contents-entry:

   Equivalent to ``histogram(samples, bins=256)``.

.. rubric:: Parameters and results

``samples``
   A ``uint8`` Tile of shape ``(N,)``. Each sample must satisfy
   ``0 <= sample < bins``. Out-of-range samples are not clamped or ignored.
``bins``
   The result is an ``int32`` tile of shape ``(bins,)``. Choose 32, 64, 128,
   or 256 bins (the default) at compile time. The selected worker count must
   divide both the input extent and the bin count. For example, a 128-worker
   implementation can produce 128 or 256 bins.
``methods``
   Must be ``None`` or ``()``. The current API does not expose a histogram
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

For valid inputs the bin counts sum to the number of samples. To combine
histograms from several tiles, add the count tiles explicitly. Any zero
padding in the input increments bin zero.
