.. SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

.. _coop-visualization-run-length-decode:

Run Length Decode
=================

Run-length decoding repeats each run's value as many times as its length.
Values ``[7, 9]`` with lengths ``[3, 2]`` decode to the stream
``7, 7, 7, 9, 9``.

The explorer follows :func:`cuda.coop.cutlass.run_length_decode` and
:func:`cuda.coop.cutlass.run_length_decode_into` on one block of four
teaching threads. Choose the run lengths, the number of runs per thread and
the number of decoded items per thread, then compare one window with the
whole stream.

.. coop-visualization:: run-length-decode

   .. only:: html

      .. figure:: run-length-decode.svg
         :alt: Run values 7 and 9 with lengths 3 and 2 start at positions 0
               and 3 of a stream of five items. A window of four items at
               offset 2 holds 7, 9, 9 and 0. Its last position is past the
               end of the stream and holds zero.
         :width: 100%

         One run and one decoded item per thread, window offset 2.

   A run starts at the sum of the lengths of the runs before it, which the
   picture calls its exclusive start. The window starts at position 2, the
   last 7. Positions 3 and 4 hold 9. Position 5 is past the end of the
   stream, so thread 3 receives zero.

   .. code-block:: text

      Run values          7    9    2    5
      Run lengths         3    2    0    0     (padding at the end)
      Exclusive starts    0    3    5    5     total = 5
      Full stream         7    7    7    9    9
      Window positions    2    3    4    5     window offset = 2
      Window values       7    9    9    0     last position holds zero

Reading the stages
------------------

The explorer follows one call. The first stage shows the runs of the
block, thread by thread: with two runs per thread, thread ``t`` holds runs
``2t`` and ``2t + 1``. The second stage checks the lengths and adds them
up, which gives the start of each run and the length of the stream.

The next stage puts the value and the start of each run into a table in
scratch, drawn as ``value@start``. :cpp:class:`cub::BlockRunLengthDecode`
holds that table and finds each window in it from the window's offset. A
position of the stream belongs to the last run with a positive length that
starts at or before it. The explorer skips the table when there is nothing
to decode: the stream is empty, or the window's offset is the length of the
stream or larger.

With "Decode full stream", the explorer checks the room in the destination
before it makes the table, and then writes one window after another from
that one table. Choose "One slot too small" as the destination capacity to
see the call stop before it writes anything. A cell that the call does not
write keeps its ``-1``.

:ref:`Run Length Decode <coop-cutlass-run-length>` in the API reference has
the rules for the lengths, the offsets and the destination.

Using Run Length Decode in a kernel
-----------------------------------

This tested kernel writes the same stream twice on a block of 32 threads:
with a loop of ``run_length_decode`` calls, then with one
``run_length_decode_into`` call. With one run per thread the stream has 94
items, and the windows of 32 items start at 0, 32 and 64. The last window
has 30 items. Both write the stream from element 5 on, and the test
compares both outputs with ``numpy.repeat``.

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_run_length_documentation.py>`__

.. literalinclude:: ../../../../python/cuda_coop_cutlass/tests/runtime/test_run_length_documentation.py
   :language: python
   :start-after: # cutlass-run-length-example-begin
   :end-before: # cutlass-run-length-example-end
   :dedent: 4

:ref:`Run Length Decode <coop-cutlass-run-length>` in the API reference sets
the two functions side by side and covers padding, windows past the end of
the stream and scratch.
