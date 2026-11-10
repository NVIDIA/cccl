.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

FAQs and troubleshooting
========================

.. _coop-tile-native-or-cooperative:

When should I use ct.x() instead of coop.x()?
---------------------------------------------

Choose by the operation's input, result shape, and semantics. Here ``ct``
names ``cuda.tile`` and ``coop`` names ``cuda.coop.tile``; in C++, the
corresponding namespaces are ``cuda::tiles`` and ``cuda::coop::tile``.
Similar names do not guarantee interchangeable calls.

Python ``ct.sum(values)`` and ``coop.sum(values)`` both return a scalar Tile
when reducing all elements. Both support integer and tuple axes, negative
axes, and ``keepdims=True``. Cooperative reductions also offer
``broadcast=True`` to repeat an aggregate across its input shape. C++
axis overloads retain the reduced dimension with extent one, while calls
without an axis repeat the result across a one-dimensional input. See
:doc:`/tile/python/primitives/reduce` for numerical differences, supported operators, and
examples of both result forms.

A reduced result can participate in subsequent elementwise expressions
through normal Tile broadcasting. Retain the reduced axes when needed to
align the result with its input, for example when subtracting a row minimum.

For other overlaps, compare the full contracts: scan axes, inclusive or
exclusive mode, initial and prefix values, supported dtypes, histogram
bin bounds, and treatment of padded or out-of-range inputs. Use the
cooperative operation when its documented behavior or algorithm controls
fit the computation. When both choices fit, benchmark the complete
consumer kernel on the target GPU, distinguishing default settings from
tuned configurations. Neither namespace promises to be faster for every
operation or shape.

Why does import succeed but the first primitive fails?
------------------------------------------------------

The Tile package loads its Python compiler integration lazily. Importing
``cuda.coop.tile`` or locating C++ build inputs does not prove that the
Python Tile compiler is available. Accessing a primitive needs the matching
``cuda-tile`` and ``cuda-tile-preview`` interfaces; compiling it also needs
the compatible TileLib and compiler toolchain.

Use the package/path check in :doc:`/tile/python/installation`, then run the first
kernel in the :doc:`programming guide </tile/python/programming>`. Check for an
editable installation or ``PYTHONPATH`` that selects a different checkout.

Which compiler and frontend packages do I need?
-----------------------------------------------

Use CUDA 13.5 and compatible ``cuda-tile`` and ``cuda-tile-preview`` packages
with the foreign-call interface. Building the provider also requires the
separately supplied interoperation SDK. See the :doc:`Python installation
guide </tile/python/installation>` and :doc:`native build
</tile/cpp/installation>` for the required inputs.

Does a 256-element tile need 256 physical threads?
--------------------------------------------------

The compiler chooses physical threads and the values handled by each
thread. The 256-element extent describes the logical input. Explicit
kernel worker hints constrain implementation selection; ordinary operation
policies do not expose thread counts or items per thread. Exchange retains
its width because it defines a permutation.

In C++, the trailing ``1`` in ``kernel<<<logical_blocks, 1>>>(...)`` is
the Tile launch sentinel. It does not request one physical CUDA thread.
See :doc:`/tile/programming_model` for both launch forms.

Does Histogram keep counts between calls?
-----------------------------------------

Each call computes counts for its input tile. Accumulate returned counter
tiles explicitly when combining several batches. Padded input values
still count as samples. The :doc:`Histogram reference
</tile/python/primitives/histogram>` describes the supported input dtype and bin bounds.

Why do Exchange calls require ``block_threads``?
------------------------------------------------

The thread count determines the index permutation. For 256 values,
striped-to-blocked with 128 threads maps output position 1 to input
position 128; with 256 threads it is the identity. Use the same count for
an Exchange and its inverse. This explicit count also constrains other
foreign calls in the kernel. See :doc:`/tile/programming_model` for the formulas.

Does TopK return sorted results or resize the output?
-----------------------------------------------------

TopK defines the first ``min(k, valid_items)`` positions as its selected
candidates without promising their order. Its returned tiles retain the input capacity.
Consume only the selected prefix. If the application needs ordered
candidates, sort that prefix using a supported tile extent and the sort's
valid-prefix or padding contract. Do not include the undefined output
tail as candidate data. Pair operations preserve each key's associated
value.

Why does Python TopK reject a one-element Tile control?
-------------------------------------------------------

Use a scalar or a rank-zero Tile with shape ``()``. A shape-one Tile,
``(1,)``, has an axis and is rejected for these controls. Runtime ``k``,
``valid_items``, and bit bounds remain subject to the primitive's numerical
constraints. See the :doc:`TopK reference </tile/python/primitives/topk>`.

Can Run Length Decode be called from Python?
--------------------------------------------

Use ``coop.run_length_decode(run_values, run_lengths, decoded_items=256)``
for values, or ``coop.run_length_decode_with_offsets`` for values and offsets
within each run. The compile-time ``decoded_items`` extent corresponds to
C++'s ``DecodedN`` template argument and is independent of the run count.
Both interfaces accept a runtime ``decoded_window_offset`` and use zero
values plus maximum-length-type offset sentinels for padding.
See the :doc:`Python </tile/python/primitives/run_length_decode>` and
:doc:`C++ </tile/cpp/primitives/run_length_decode>` references for examples.

How do Tile calls compare with CUB block primitives?
----------------------------------------------------

Tile calls use CUB block primitives and preserve their mathematical contracts
on supported inputs. For example, ``radix_rank_keys`` uses CUB's ordered
representation of signed integers; TopK defines the first
``min(k, valid_items)`` outputs, with unspecified tie order. Scan defaults to
``raking`` and reduction to ``warp_reductions`` in both Tile languages.

A Tile describes the complete logical input, so its operations infer input
extent and return new Tiles. CUB block APIs operate on per-thread items and
expose the block configuration and temporary storage. The Tile compiler and
provider select that configuration and supply the storage. For Run Length
Decode, Python's
``decoded_items`` and C++'s ``DecodedN`` specify the whole output capacity.
Neither Tile interface requires thread groups or user-managed scratch storage.

Python compile-time keyword controls correspond to C++ policy/template choices;
runtime scalar controls remain runtime values in both. Python tuples correspond
to C++ structured-binding results. Python can convert a one-element control
Tile to the required dtype; C++ one-element Tile overloads require that exact
dtype. Scalar conversions follow the destination dtype in both languages.

Tile exposes a subset of CUB's controls. Its scans and reductions use named
built-in operations, and its histogram accepts byte samples with 32-bit counts.
The primitive references state these limits.
Python Tile reductions return reduced shapes by default, with an explicit
``broadcast=True`` option. C++ calls without an axis repeat their result;
axis overloads retain the reduced dimension with extent one.

Why does the resolver reject my development TileLib?
----------------------------------------------------

``CUDA_COOP_TILE_LIBRARY_PATH`` must name an existing file. Python preview
calls also require its identity. With an adjacent
``provider-manifest.json``, the resolver checks the distribution, version,
filename, and hash. Without a manifest, supply a nonempty
``CUDA_COOP_TILE_LIBRARY_VERSION`` and change it when the library or its
dependencies change.

A manifest mismatch means the selected file does not match the packaged
identity. Rebuild/reinstall the matching artifacts or select a coherent
development artifact; do not reuse an old version token to make a changed
library appear unchanged. See :doc:`/tile/interop`.

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

CUDA 13.5 does not invoke ``filter_and_rank``. Filtering and preference
scoring happen in ``provide_impls``, before the enclosing kernel's final
worker count is known. Qualification checks the advertised scores and the
implementation selected for compilation, together with numerical results.
A successful kernel does not establish support for every worker width or
logical extent. See :doc:`/tile/development` for the validation steps.
