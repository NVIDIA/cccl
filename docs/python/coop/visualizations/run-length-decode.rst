.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-run-length-decode:

Run Length Decode
=================

Run Length Decode expands each run value by its length. Values ``[7, 9]``
with lengths ``[3, 2]`` describe the stream ``[7, 7, 7, 9, 9]``.
:func:`cuda.coop.run_length_decode` returns one window of that stream as a
fresh per-thread payload. :func:`cuda.coop.run_length_decode_into` writes
the whole stream to an array and returns its total size to every thread.
Both preserve the run inputs.

The explorer uses a complete block of four teaching threads. Choose the
input lengths and the number of decoded items per thread, then compare a
single window with the bulk loop. Input runs and decoded outputs have
independent per-thread extents. The tested kernels below use 128 threads.

.. coop-visualization:: run-length-decode

   .. only:: html

      .. figure:: run-length-decode.svg
         :alt: Run values 7 and 9 with lengths 3 and 2 have exclusive starts 0 and 3 and total size 5. A four-item window at offset 2 returns 7, 9, 9, 0 with relative offsets 2, 0, 1, MAX. The final slot is outside the stream and zero-filled.
         :width: 100%

         One decoded item per teaching thread, window offset 2. ``MAX``
         denotes the selected unsigned offset dtype's maximum value.

   An exclusive scan of the run lengths gives the run starts. A decoded
   index belongs to the last positive-length run whose start is no greater
   than that index. Subtract its run start to obtain the relative offset.

   .. code-block:: text

      Run values          7    9    2    5
      Run lengths         3    2    0    0     (trailing padding)
      Exclusive starts    0    3    5    5     total = 5
      Full stream         7    7    7    9    9
      Window indices      2    3    4    5     window offset = 2
      Window values       7    9    9    0     last slot zero-filled
      Relative offsets    2    0    1   MAX

Run inputs and window ownership
-------------------------------

Each block owns a fixed tile of runs in :term:`blocked` order. Thread
``t`` holds run ``t * runs_per_thread + i`` in its local slot ``i``. All
positive run lengths must precede any zero padding. Zeros let a fixed run
tile represent fewer runs; an all-zero tile represents an empty stream.
An interior zero followed by a positive length is invalid.

``decoded_items_per_thread`` sets the returned payload's extent. If it is
``D``, a block of ``B`` threads returns a window of ``B * D`` positions.
Thread ``t`` receives decoded indices beginning at
``decoded_window_offset + t * D``. The window offset counts decoded items,
so it can begin inside a run. It must have the same value in every thread.

For the example above, offset 2 starts at the last ``7``. The next two
items are ``9``. A four-item window also includes decoded index 5, which
is beyond the total of 5. The wrapper fills that value slot with zero and
its optional relative offset with ``MAX``. These tail values are defined;
an empty stream or a window starting at or beyond the total produces the
same fill values in every slot.

Preparing and reusing the table
-------------------------------

The provider checks the lengths and computes exclusive run starts, then
passes the values and starts to :cpp:class:`cub::BlockRunLengthDecode`.
CUB holds the prepared run table in shared storage. Its decode method
looks up each requested window in that table; it does not advance a
hidden cursor. The explorer shows these dependencies and storage lifetime,
not a sequence of compiled CUB instructions.

Each :func:`~cuda.coop.run_length_decode` call prepares its own table.
The bulk :func:`~cuda.coop.run_length_decode_into` call prepares the table
once and keeps it alive across an internal loop. Each iteration decodes
``block_threads * decoded_items_per_thread`` positions and stores only the
valid part of its window. The final iteration can be partial. No public
parent object or prepared-state token is needed for either operation.

In bulk mode, ``destination_offset`` is the index where the full stream
is written. It does not skip decoded input. The destination must have
room for that offset plus the entire decoded total. Select "One slot too
small" to see the guard stop before any output write. An empty stream
leaves the destination unchanged, provided its offset is within capacity.
The optional relative-offset destination uses the same offset, and both
capacities are checked before either output changes.

Only complete one-dimensional blocks are supported. All members take
part, including threads holding only zero-length padding. The API checks
negative lengths, padding order, total overflow, and runtime offset ranges.
The caller supplies uniform controls and nonoverlapping output arrays.
See the :doc:`../../coop_api` reference for dtype and size limits.

A window with relative offsets
------------------------------

The qualified API adds ``total_decoded_size`` and ``relative_offsets``
outputs to the common window operation. The total is the size of the full
stream, available in an extent-one payload in every thread. Relative
offsets restart at zero for each run. Both outputs default to uint32;
``decoded_offset_dtype`` can select uint64. Fixed local arrays are also
accepted. Shared window behavior follows the common API.

This tested kernel uses two runs per thread and four decoded items per
thread. Its two positive runs produce only three valid items after offset
2. The remaining window slots show the defined zero and ``MAX`` fills.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_run_length_examples.py
   :language: python
   :start-after: # run-length-window-example-begin
   :end-before: # run-length-window-example-end
   :dedent: 4

Decode a full stream
--------------------

This common-API example expands 1,105 items through three internal
512-item windows. The last window writes 81 items. Destination positions
before offset 3 and after the decoded interval keep their initial values.

.. literalinclude:: ../../../../python/cuda_coop/tests/backends/numba_mlir/runtime/test_run_length_examples.py
   :language: python
   :start-after: # run-length-bulk-example-begin
   :end-before: # run-length-bulk-example-end
   :dedent: 4

The qualified bulk operation can also write global relative offsets and
select uint64 totals. Both forms allocate scratch automatically unless
you supply ``temp_storage``. Scratch remains occupied through the internal
loop. When reusing a descriptor with ``auto_sync=False``, synchronize the
block after the call and before its next use; see
:ref:`coop-faq-temp-storage`.
