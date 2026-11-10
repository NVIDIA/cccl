.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Scan and prefix sums
====================

.. py:currentmodule:: cuda.coop.tile

Scan computes a prefix at each position. An exclusive prefix omits the current
input; an inclusive prefix includes it. For ``[2, 1, 3, 2]``, addition produces
``[0, 2, 3, 6]`` exclusively and ``[2, 3, 6, 8]`` inclusively.

Interface and examples
----------------------

.. py:function:: exclusive_sum(value, /, *, prefix_value=None, algorithm=None, methods=None)
   :no-contents-entry:

   Add all preceding values. The default prefix is zero.

.. py:function:: inclusive_sum(value, /, *, prefix_value=None, algorithm=None, methods=None)
   :no-contents-entry:

   Add the current value and all preceding values.

.. py:function:: scan(value, /, *, mode='exclusive', scan_op=None, initial_value=None, prefix_value=None, algorithm=None, methods=None)
   :no-contents-entry:

   Select exclusive or inclusive scan with ``mode``.

.. py:function:: exclusive_scan(value, /, *, scan_op=None, initial_value=None, prefix_value=None, algorithm=None, methods=None)
   :no-contents-entry:

   Apply ``scan_op`` to preceding values, starting with the supplied seed.

.. py:function:: inclusive_scan(value, /, *, scan_op=None, initial_value=None, prefix_value=None, algorithm=None, methods=None)
   :no-contents-entry:

   Apply ``scan_op`` through the current position, optionally with a prefix.

.. rubric:: Parameters and results

``value``
   A numeric Tile of shape ``(N,)``. The result has the same shape and dtype.
   No output crosses the boundary of this input Tile.
``mode``
   For ``scan`` only, the compile-time string ``"exclusive"`` (default) or
   ``"inclusive"``.
``scan_op``
   A compile-time operation: ``"sum"`` or ``"+"`` (default), ``"*"``, ``"min"``, ``"max"``,
   ``"&"``, ``"|"``, or ``"^"``. The bitwise operations require integer inputs.
   The operation is a supported name, not a Python callable.
``initial_value``
   The seed for an exclusive scan, converted to the input dtype. Addition
   defaults to zero. Other exclusive operations require an explicit
   ``initial_value`` or ``prefix_value``. Inclusive scan requires this
   argument to be ``None``; supply its seed through ``prefix_value``.
``prefix_value``
   A seed applied before the Tile's values. This can be a constant, runtime
   scalar, or single-element Tile. For exclusive scan, it supplies the first
   result. For inclusive scan, it combines with the first input. It cannot be
   combined with ``initial_value``. It is a value, not a callback.
``algorithm``
   ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``. Omitting it uses
   ``"raking"``. The algorithm changes how the prefix is computed;
   floating-point roundoff can change with the reduction tree.
``methods``
   Sum variants accept ``algorithm`` and ``prefix_value``. Generic scan variants
   also accept ``scan_op`` and ``initial_value``. ``mode`` remains a direct
   keyword of ``scan``.

.. rubric:: Example: exclusive offsets

This complete Python example scans 256 integer counts. Each output gives the
starting offset of that count in a packed sequence.

.. code-block:: python

   import torch
   import cuda.tile as ct
   import cuda.coop.tile as coop

   @ct.kernel(num_worker_warps=4)
   def offsets_kernel(counts, offsets):
       values = ct.load(counts, (ct.bid(0),), shape=(256,))
       result = coop.exclusive_sum(values)
       ct.store(offsets, (ct.bid(0),), result)

   counts = torch.arange(256, device="cuda", dtype=torch.int32) % 5
   offsets = torch.empty_like(counts)
   ct.launch(torch.cuda.current_stream(), (1, 1, 1), offsets_kernel, (counts, offsets))
   expected = counts.cumsum(0, dtype=torch.int32) - counts
   assert torch.equal(offsets, expected)


Behavior
--------


With a prefix of 10, the exclusive example becomes ``[10, 12, 13, 16]`` and the
inclusive example becomes ``[12, 13, 16, 18]``. For multiplication, use an
exclusive seed of 1 to obtain the usual product prefix. Choose seeds in the
input dtype; the provider does not widen accumulation.

.. figure:: /tile/visualizations/scan.svg
   :alt: Exclusive and inclusive prefixes with seed values and different operations.

   Prefix boundaries, seeds, multiplication, and maximum for the same input.
   This small array illustrates the operation; it is not a supported launch shape.
