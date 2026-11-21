.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FAQs and troubleshooting
========================

.. _coop-tile-native-or-cooperative:

When should I use ct.x() instead of coop.x()?
---------------------------------------------

Choose the result shape and numerical behavior your next operation needs.
For example, both Python sums return a scalar Tile when reducing all
elements. Cooperative sum also accepts ``broadcast=True`` to repeat the
aggregate across the input shape. For a row minimum, retaining the reduced
axis lets you subtract the minimum from every element in that row.

The :doc:`Python </tile/python/primitives/reduce>` and
:doc:`C++ </tile/cpp/primitives/reduce>` references compare cooperative
and native reductions, including precision and NaN handling.
When either operation meets those requirements, time the complete
kernel on the target GPU.

Why does import succeed but the first primitive fails?
------------------------------------------------------

Importing ``cuda.coop.tile`` loads the host helpers. Accessing a primitive
loads the cuTile integration. Compiling the kernel also requires the
matching frontend packages, compiler, and TileLib. Check the package paths
in :doc:`/tile/python/installation`, then run the :doc:`first Python kernel
</tile/python/programming>` in that environment.

Which compiler and frontend packages do I need?
-----------------------------------------------

Use CUDA 13.5. Python needs the matching ``cuda-tile`` and
``cuda-tile-preview`` versions in :doc:`/tile/python/installation`.
Building the native provider also requires the separate Tile/SIMT SDK
inputs listed in :doc:`/tile/cpp/installation`.

Does a 256-element tile need 256 physical threads?
--------------------------------------------------

A worker can process several elements. The extent counts logical values.
The compiler selects workers for the kernel, taking any explicit worker
hint into account. The :doc:`programming model </tile/programming_model>`
shows how the grid and partition coordinates fit together.

Does Histogram keep counts between calls?
-----------------------------------------

Each call returns counts for one input tile. Add the counter tiles to
combine batches. Padding contributes samples too. See the
:doc:`Histogram example </tile/python/primitives/histogram>`.

Why do Exchange calls require ``block_threads``?
------------------------------------------------

Changing the width changes which element lands at each position.
For 256 values and 128 threads, striped-to-blocked sends input 128 to
output 1. With 256 threads, input 1 stays at output 1.
See :doc:`the permutation and its inverse </tile/programming_model>`.

Does TopK return sorted results or resize the output?
-----------------------------------------------------

The result keeps its input capacity. Read the first
``min(k, valid_items)`` positions and treat them as an unordered selection.
If the consumer needs ordered values, sort only that prefix or use a
sort with a suitable padding key. See the
:doc:`Python </tile/python/primitives/topk>` and
:doc:`C++ </tile/cpp/primitives/topk>` references.

Why does Python TopK reject a one-element Tile control?
-------------------------------------------------------

Use a scalar or a rank-zero Tile with shape ``()``. A shape-one Tile,
``(1,)``, has an axis and is rejected for these controls. Runtime ``k``,
``valid_items``, and bit bounds remain subject to the primitive's numerical
constraints. See the :doc:`TopK reference </tile/python/primitives/topk>`.

Can Run Length Decode be called from Python?
--------------------------------------------

Use ``coop.run_length_decode(run_values, run_lengths, decoded_items=256)``
to return 256 decoded positions. For each value's offset within its run,
use ``coop.run_length_decode_with_offsets``. The references illustrate a
window that starts partway through a run in
:doc:`Python </tile/python/primitives/run_length_decode>` and
:doc:`C++ </tile/cpp/primitives/run_length_decode>`.

How do Tile calls compare with CUB block primitives?
----------------------------------------------------

A CUB block primitive operates on items owned by each thread. In Tile
you supply the logical input value. The compiler and provider supply
the physical implementation and scratch space. For example, Run Length
Decode takes the *whole* decoded output capacity in Python's
``decoded_items`` or C++'s ``DecodedN``. See the
:doc:`programming model </tile/programming_model>` for result shapes and
the :doc:`developer guide </tile/development>` for a CUB call's compiler path.

Why does the resolver reject my development TileLib?
----------------------------------------------------

Check that ``CUDA_COOP_TILE_LIBRARY_PATH`` names the intended file. If
``provider-manifest.json`` accompanies it, the distribution, version,
filename, and hash must agree. An unpackaged build needs a new
``CUDA_COOP_TILE_LIBRARY_VERSION`` token when its contents change. See
:doc:`/tile/interop` for the identity and cache rules.

Why does CMake find the package but use the wrong compiler?
-----------------------------------------------------------

The imported target supplies headers and Tile compilation options. Select
CUDA 13.5's compiler with ``CMAKE_CUDA_COMPILER`` when configuring a
fresh build directory. ``cuda_coop_tile_link(target)`` does not install or
choose ``nvcc``. The :doc:`C++ guide </tile/cpp/programming>` gives a complete consumer.

If a replaced TileLib does not rebuild the CUDA source, call the helper
after adding that target's CUDA sources. Each consuming target needs the
compilation dependency.

Why do separately working calls fail when combined?
---------------------------------------------------

The combined kernel needs compatible physical participation and sufficient
resources for every foreign call. An explicit Exchange width or a
``num_worker_warps`` setting may eliminate the implementations accepted
by another call. Calls from another provider also need a compatible ABI.
Use the provider trace in :doc:`/tile/interop` to inspect the resolved choices.

How is provider selection qualified on CUDA 13.5?
-------------------------------------------------

On CUDA 13.5 the provider scores candidates in ``provide_impls``, before
the compiler chooses the kernel's worker count. The :doc:`developer guide
</tile/development>` describes how to check those scores against the
implementation the compiler chose and the result the kernel produced.
