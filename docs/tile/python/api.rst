.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

``cuda.coop.tile`` API Reference
================================

``cuda.coop.tile`` provides 31 Python primitives inside ``cuda.tile`` kernels.
Operations consume logical Tile values. Calls on different Tiles do not
share prefixes, reductions, run boundaries, or sorting state.
:doc:`/tile/python/host_api` covers Python helpers for locating installed build inputs.

.. toctree::
   :maxdepth: 1

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
TopK
keeps capacity ``N`` while defining only the first ``min(k, valid_items)`` elements.

The numeric element types are ``int8``, ``uint8``, ``int16``, ``uint16``,
``int32``, ``uint32``, ``int64``, ``uint64``, ``float16``, ``bfloat16``,
``float32``, and ``float64``. Python sum also accepts boolean inputs and
promotes them to ``int32``. Other boolean and structured element types are unsupported.
Histogram accepts only ``uint8`` samples; radix rank accepts only the eight
integer types. Bitwise scan and reduction operations also require integers.

.. rubric:: Topology

Ordinary calls describe the logical extent and operation. The provider
offers compatible physical implementations; the Python frontend currently
requires that a width from 32, 64, 128, or 256 divides ``N`` for operations
other than reductions. Reductions also accept rows smaller than a warp.
Individual families add constraints, described on their pages. This shape check does
not guarantee that each width is supported by a particular compiler build.
The runnable examples state their logical extents and any worker hints.
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
are compile-time parameters. ``methods`` is an optional compile-time mapping
or a flat tuple of alternating names and values, for example
``methods=("algorithm", "raking")``. Prefer named keyword arguments in new
code. The family page lists accepted method keys. An explicit non-``None``
constant keyword overrides the corresponding method value. A runtime keyword
cannot also appear in ``methods``. Unknown method keys are errors.

Invalid constant controls raise typing errors during compilation. Runtime
controls must satisfy each family's documented bounds. Native providers
trap on the bounds they validate. Data-dependent contracts, such as histogram
samples lying below ``bins``, remain the caller's responsibility.
