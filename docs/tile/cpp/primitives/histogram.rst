.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Histogram
=========

Histogram counts each input byte in a bin with the same integer index. A call
starts its own zeroed histogram and returns counts for that Tile only.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

``Bins`` is a compile-time value of 32, 64, 128, or 256; every sample must
be below ``Bins``. The selected block width must divide both the positive input
extent ``N`` and ``Bins``. With 128 threads, use 128 or 256 bins and an input
extent divisible by 128. With 256 threads, use 256 bins and an input extent
divisible by 256. Counts start at zero on each call. The API exposes no
histogram algorithm selector.

``histogram<Bins>(samples, policy)`` returns a Tile of ``int`` with extent
``Bins``. ``histogram_256`` fixes that extent at 256. Samples must have element
type ``unsigned char``. The optional ``histogram_policy<>`` has no options;
the provider selects a width that divides the sample extent and bin count.

.. code-block:: cpp

   namespace coop = cuda::coop::tile;
   auto counts = coop::histogram<256>(samples);
   auto same_counts = coop::histogram_256(samples);


Behavior
--------


The result is an ``int`` Tile of extent ``Bins``. Bin ``b`` contains the
number of input elements equal to ``b``. Counts start at zero for each call;
there is no accumulation into a preexisting histogram. Their sum equals ``N``
for valid input. ``histogram_256(samples)`` is equivalent to
``histogram<256>(samples)``.

.. figure:: /tile/visualizations/histogram.svg
   :alt: A 256-sample byte Tile produces 256 counts with 64 occupied bins.

   The 256 input samples occupy 64 bins. All 256 output bins are present,
   including zero counts for values that do not occur.
