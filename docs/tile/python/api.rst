.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

``cuda.coop.tile`` API Reference
================================

Use these operations on logical tile values inside a ``cuda.tile`` kernel.
Each block computes its own result. Choose the family by the value you
want to feed into the next operation or store to memory.

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Result
     - Primitive family
   * - A running total or a prefix per element
     - :doc:`primitives/scan`
   * - An aggregate for a whole tile or for each row
     - :doc:`primitives/reduce`
   * - A new element order, a shift, or a rotation
     - :doc:`primitives/exchange_shuffle`
   * - Differences between neighbors or flags at run boundaries
     - :doc:`primitives/adjacent_discontinuity`
   * - A count for each byte value
     - :doc:`primitives/histogram`
   * - A window of expanded runs, optionally with the position within each run
     - :doc:`primitives/run_length_decode`
   * - Keys or key/value pairs ordered by comparison
     - :doc:`primitives/merge_sort`
   * - Keys ordered by radix bits, or a destination rank per input
     - :doc:`primitives/radix_sort_rank`
   * - The largest or smallest keys, optionally with their payloads
     - :doc:`primitives/topk`

For Python helpers that locate the installed C++ headers, CMake package and
TileLib, see :doc:`host_api`.

.. toctree::
   :hidden:

   primitives/scan
   primitives/reduce
   primitives/exchange_shuffle
   primitives/adjacent_discontinuity
   primitives/histogram
   primitives/run_length_decode
   primitives/merge_sort
   primitives/radix_sort_rank
   primitives/topk
   host_api

Shapes and element types
------------------------

Data inputs have compile-time shapes. Python reductions support multiple
axes; other operations accept one-dimensional shape ``(N,)``. Pair operations
require equal key and value shapes; their element types may differ. Outputs
preserve the input element type and extent unless the family page states
otherwise. Python reductions remove reduced axes by default; ``keepdims=True``
retains them with extent one, and ``broadcast=True`` restores the input shape.
TopK keeps capacity ``N`` while defining only the first ``min(k, valid_items)`` elements.

The numeric element types are ``int8``, ``uint8``, ``int16``, ``uint16``,
``int32``, ``uint32``, ``int64``, ``uint64``, ``float16``, ``bfloat16``,
``float32``, and ``float64``. Both ``sum`` and ``reduce`` with addition
accept boolean inputs and promote them to ``int32``. Other reductions and scans
use the listed numeric types.
Histogram accepts only ``uint8`` samples; radix rank accepts only the eight
integer types. Bitwise scan and reduction operations also require integers.

.. rubric:: Topology

For a valid cuTile shape, one-dimensional primitives other than reductions
require a positive multiple of 32 input elements. The :doc:`reduction
reference <primitives/reduce>` gives the extent rule for each aggregate,
including shorter rows. Each family can impose additional constraints on
extent and worker count.
An omitted hint leaves selection to the compiler, subject to each call's
requirements. See the
:doc:`/tile/python/installation` and :doc:`/tile/programming_model` for the compiler and launch contract.

.. rubric:: Parameters and options

Python positional data parameters are Tiles. Integer controls such as
``valid_items`` or ``begin_bit`` may be compile-time values or runtime scalar
Tiles where the family permits them. They become signed 32-bit controls.
Run-length decode window offsets instead use the unsigned run-length dtype.
Numeric seeds and padding keys use the data or key type. For integer data,
a constant seed or padding key must be an integer; a floating-point constant
such as ``1.0`` is rejected even when it has an integral value. Boolean
constants are also rejected. Runtime controls are cast to the required dtype.
Most scalar controls accept a Tile containing one element and normalize its
shape; TopK requires a scalar or rank-zero Tile.

Option strings, booleans, ``bins``, ``decoded_items``, and ``block_threads``
are compile-time parameters. ``methods`` is an optional compile-time flat tuple
of alternating names and values, for example ``methods=("algorithm", "raking")``.
Use ``None`` or ``()`` when no options are needed. Prefer named keyword arguments
in new code. The family page lists accepted method keys. An explicit non-``None``
constant keyword overrides the corresponding method value. A runtime keyword
cannot also appear in ``methods``. Unknown method keys are errors.

Invalid constant controls raise typing errors during compilation. Runtime
controls must satisfy each family's documented bounds. Native providers
trap on the bounds they validate. Data-dependent contracts, such as histogram
samples lying below ``bins``, remain the caller's responsibility.
