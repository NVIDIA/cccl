.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Run-length decode
=================

.. py:currentmodule:: cuda.coop.tile

Run-length decoding expands each run value by its run length and returns a
fixed-size window into the expanded sequence. Python and C++ provide decoded
values alone or decoded values followed by relative offsets within each run.
The output capacity is a compile-time choice independent of the number of runs.
Both extents must be powers of two, at least 32, and admit a common worker count.

Interface and examples
----------------------

.. py:function:: run_length_decode(run_values, run_lengths, /, *, decoded_items, decoded_window_offset=0, methods=None)
   :no-contents-entry:

   Expand runs into a fixed-capacity window of decoded values.

.. py:function:: run_length_decode_with_offsets(run_values, run_lengths, /, *, decoded_items, decoded_window_offset=0, methods=None)
   :no-contents-entry:

   Return decoded values followed by offsets within each original run.

.. rubric:: Parameters and results

``run_values``
   A numeric Tile of shape ``(RunN,)``. The order of the runs determines
   the expanded sequence. Both functions leave their inputs unchanged.
``run_lengths``
   A Tile of matching shape with dtype ``uint32`` or ``uint64``. Positive
   lengths may be followed by trailing zero-length runs. A zero-length run
   must not precede a positive run. The sum must fit the length dtype.
``decoded_items``
   The required compile-time output capacity, independent of ``RunN`` and
   the sum of the run lengths. Both extents must be powers of two, at least
   32, and fit the provider's signed 32-bit extent controls. The selected
   worker count must divide both extents.
``decoded_window_offset``
   A starting position in the expanded sequence, defaulting to zero.
   An integer constant, runtime integral scalar, or single-element integer
   Tile converts to the length dtype. The offset must be nonnegative and
   representable in that dtype. A window may begin inside a run or beyond
   the entire sequence.
``methods``
   Must be omitted or empty.

``run_length_decode`` returns a Tile of shape ``(decoded_items,)`` with
the run-value dtype. ``run_length_decode_with_offsets`` returns
``(values, relative_offsets)``, both with that shape. Relative offsets
retain the length dtype and count from the start of the original run,
including when the window starts inside it. Positions past the expanded
sequence have value zero and the maximum value of the length dtype as
their offset. All-zero lengths or an entirely out-of-range window produce
only this padding.

.. rubric:: Example: decode a window with offsets

The following complete example loads 128 runs and produces 256 decoded
items. Only the first three runs are nonempty. Both interfaces return
the same values; the second also returns each position within its run.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel
   def decode_window(values, lengths, decoded, decoded_pair, within_run):
       run_values = ct.load(values, (0,), shape=(128,))
       run_lengths = ct.load(lengths, (0,), shape=(128,))
       result = coop.run_length_decode(
           run_values, run_lengths, decoded_items=256,
           decoded_window_offset=2,
       )
       paired, relative = coop.run_length_decode_with_offsets(
           run_values, run_lengths, decoded_items=256,
           decoded_window_offset=2,
       )
       ct.store(decoded, (0,), result)
       ct.store(decoded_pair, (0,), paired)
       ct.store(within_run, (0,), relative)

   values = torch.tensor([7, 4, 9] + [0] * 125,
                         device="cuda", dtype=torch.int32)
   lengths = torch.tensor([3, 1, 2] + [0] * 125,
                          device="cuda", dtype=torch.uint32)
   decoded = torch.empty(256, device="cuda", dtype=torch.int32)
   decoded_pair = torch.empty_like(decoded)
   within_run = torch.empty(256, device="cuda", dtype=torch.uint32)
   ct.launch(torch.cuda.current_stream(), (1,), decode_window,
             (values, lengths, decoded, decoded_pair, within_run))
   torch.cuda.synchronize()
   assert decoded.cpu().tolist() == [7, 4, 9, 9] + [0] * 252
   assert decoded_pair.cpu().tolist() == decoded.cpu().tolist()
   assert within_run.cpu().tolist() == [2, 0, 0, 1] + [2**32 - 1] * 252


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
