.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Run-length decode
=================

Run-length decoding expands each run value by its run length and returns a
fixed-size window into the expanded sequence. Python and C++ provide decoded
values alone or decoded values followed by relative offsets within each run.
The output capacity is a compile-time choice independent of the number of runs.
Both extents must be powers of two, at least 32, and admit a common worker count.

Interface and examples
----------------------

See the :doc:`C++ API reference </tile/cpp/api>` for indexed declarations
and overload forms.

.. rubric:: Calls and return types

.. code-block:: cpp

   run_length_decode<DecodedN>(run_values, run_lengths, policy = {});
   run_length_decode<DecodedN>(run_values, run_lengths, decoded_window_offset, policy = {});
   run_length_decode_with_offsets<DecodedN>(run_values, run_lengths, policy = {});
   run_length_decode_with_offsets<DecodedN>(run_values, run_lengths, decoded_window_offset, policy = {});

``run_values`` and ``run_lengths`` have one-dimensional shape ``RunN``.
``run_length_decode`` returns a value Tile of shape ``DecodedN`` with the
same element type as ``run_values``. ``run_length_decode_with_offsets`` returns
that value Tile followed by a relative-offset Tile of the same shape and
length element type. Use a structured binding for the pair.

.. rubric:: Parameters

``DecodedN``
   The positive compile-time output capacity. It is independent of ``RunN``
   and need not equal the sum of the run lengths.
``run_values``
   A Tile of supported numeric run values. The order of the runs determines
   the order of the expanded sequence; equal values in separate runs remain
   separate runs for the relative offsets.
``run_lengths``
   A Tile with the same shape as ``run_values``. Supported C++ element types
   are ``unsigned int``, ``unsigned long``, and ``unsigned long long``.
   Positive lengths may be followed by trailing zero-length runs. A zero
   length must not precede a positive length. The sum must be representable
   in the length type.
``decoded_window_offset``
   An integral scalar convertible to the length type, or a shape-one Tile of
   exactly that type. It is an offset into the expanded sequence, not an
   index into the run input. The default is zero. Use a nonnegative value
   representable in the length type; avoid narrowing a large offset.
``policy``
   Optional ``run_length_decode_policy<>`` with no options. The provider
   derives the runs and decoded items per thread from ``RunN``, ``DecodedN``,
   and a compatible worker count dividing both. Both extents must be positive
   and representable by the provider's signed 32-bit extent controls.

.. rubric:: Example: decode a window with offsets

This kernel reads 128 runs and writes 256 decoded items. The caller supplies
128 run values and lengths, two output allocations of 256 elements each, and
a window offset. The :doc:`/tile/cpp/programming` guide describes compiling and launching
this kernel with the Tile provider.

.. code-block:: cpp

   #include <cuda/coop/tile.cuh>

   namespace ct = cuda::tiles;
   namespace coop = cuda::coop::tile;

   __tile_global__ void decode_window(
     const int* values, const unsigned int* lengths,
     int* decoded, unsigned int* within_run, unsigned int offset)
   {
     auto values_span = ct::tensor_span{values, ct::shape<128>{}};
     auto lengths_span = ct::tensor_span{lengths, ct::shape<128>{}};
     auto decoded_span = ct::tensor_span{decoded, ct::shape<256>{}};
     auto offsets_span = ct::tensor_span{within_run, ct::shape<256>{}};
     auto values_view = ct::partition_view{values_span, ct::shape<128>{}};
     auto lengths_view = ct::partition_view{lengths_span, ct::shape<128>{}};
     auto decoded_view = ct::partition_view{decoded_span, ct::shape<256>{}};
     auto offsets_view = ct::partition_view{offsets_span, ct::shape<256>{}};

     auto [result, relative] = coop::run_length_decode_with_offsets<256>(
       values_view.load(0), lengths_view.load(0), offset);
     decoded_view.store(result, 0);
     offsets_view.store(relative, 0);
   }

For a concrete host input, put ``7, 4, 9`` in the first three value slots and
``3, 1, 2`` in the corresponding length slots; fill the other 125 lengths
with zero. With ``offset = 2``, the first six output positions match the
window below, and the remaining 250 positions are padded. This input uses the
kernel's actual 128-run and 256-output shapes.


Window boundaries and padding
-----------------------------

For run values ``[7, 4, 9]`` and lengths ``[3, 1, 2]``, the expanded sequence is
``[7, 7, 7, 4, 9, 9]``. A window starting at offset 2 with capacity 6 returns:

.. code-block:: text

   decoded values:    [7, 4, 9, 9, 0, 0]
   relative offsets:  [2, 0, 0, 1, MAX, MAX]

``MAX`` is the maximum value of the unsigned length type. It marks output
positions beyond the expanded sequence. The corresponding value is zero.
A window at or beyond the total decoded length contains only this padding.
A valid relative offset counts from the beginning of its run, even when the
window begins partway through the run. Test the offset sentinel if zero is a
valid run value and the consumer needs to distinguish it from padding.

.. figure:: /tile/visualizations/run-length-decode.svg
   :alt: A decode window begins within the first run and pads beyond the final run.

   The decoded output keeps its requested capacity. Relative offsets identify
   positions within each run and mark the padded tail.

Python/C++ parity
-----------------

``decoded_items`` in Python corresponds to ``DecodedN`` in C++. Both forms
preserve input order, value dtype, length dtype, window position, padding, and
relative offsets. Python returns a tuple where C++ returns a structured-binding
result. Python shapes are compile-time powers of two; both extents must admit
a common provider worker count. ``uint32`` and ``uint64`` lengths are supported.
All-zero lengths produce a fully padded window. Interior zero runs violate the
underlying CUB contract.
