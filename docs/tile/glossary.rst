.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
..
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Glossary
========

These terms describe the logical Tile interface and its compiler/provider
implementation. The :doc:`programming model </tile/programming_model>` relates
shapes, physical execution, and result ownership.

.. glossary::
   :sorted:

   logical Tile
      A value with an element dtype and a compile-time shape, such as
      ``cuda.tile.Tile`` in Python or ``cuda::tiles::tile`` in C++. A
      cooperative call consumes and returns these values inside a kernel.
      Reductions also support multidimensional input tiles.

   logical block
      One independently indexed execution of a Tile kernel. Python
      ``ct.bid(0)`` and C++ ``ct::bid().x`` select the block's partition.
      A launch grid counts logical blocks; each block's scan or reduction
      processes its own input tile.

   tile coordinate
      An index into shaped array partitions. For a one-dimensional load
      with shape ``(256,)``, coordinate 1 selects elements 256 through 511.
      It is not an element offset. See the :doc:`Python example </tile/python/programming>`.

   tile extent
      The compile-time number of elements along an axis. The extent of a
      one-dimensional cooperative input is its capacity. Extent does not
      specify how many physical threads execute the call.

   Tile result capacity
      The number of positions in a returned Tile value. Runtime controls
      do not change this compile-time shape. TopK returns the input
      capacity even when only its first ``min(k, valid_items)`` positions are
      selected.

   rank-zero Tile
      A scalar Tile with shape ``()``. Python TopK accepts scalar and
      rank-zero Tile controls for ``k``, ``valid_items``, and bit bounds.
      A rank-zero Tile differs from a one-dimensional tile of extent one.

   shape-one Tile
      A one-dimensional Tile with shape ``(1,)``. It contains one element
      but retains an axis. Some numeric-control overloads accept this
      representation; Python TopK rejects it. Check the operation's
      :doc:`signature </tile/python/api>` before using it as a control.

   worker warp
      A physical CUDA warp assigned by the compiler to execute a Tile
      kernel. Python's ``num_worker_warps`` option constrains that choice.
      The supported values depend on the selected frontend.

   implementation topology
      The concrete physical thread count and values per thread chosen
      for a cooperative implementation. Compatible topology must exist
      for the calls combined in one kernel. Exchange additionally exposes
      a thread count because it changes the requested permutation.

   Tile policy
      A C++ type selecting compile-time operation options, such as
      ``scan_policy<scan_algorithm::raking>`` or ``exchange_policy<128>``.
      Ordinary policies select algorithms rather than physical participation.
      Exchange's width defines its permutation. Policies do not allocate scratch.

   compile-time option
      A choice used to select a compiled specialization, such as the
      operation name, an algorithm, a dtype, or a tile shape. Python
      lowerings require these options to be constants.

   runtime control
      A numerical operand that can change between executions without
      changing the Tile shape, such as a supported scan prefix or sort
      bit bound. The accepted scalar and scalar-Tile forms vary by
      operation and language interface.

   repeated reduction result
      A reduction whose aggregate is repeated across the input shape:
      Python ``broadcast=True`` or a C++ call without an axis. Python
      reductions return reduced shapes by default; C++ axis overloads
      retain each reduced dimension with extent one. Applications can
      consume repeated results in a Tile expression or extract an element.

   selected prefix
      The first ``min(k, valid_items)`` result positions containing TopK's
      chosen candidates.
      Their order is unspecified, and the rest of the result capacity is
      not a selected output. Pair TopK preserves each selected key's
      associated value.

   Exchange permutation
      The conversion between blocked and striped logical index order for
      an explicit thread count. With extent ``N``, count ``B``, and
      ``I = N / B``, striped-to-blocked computes
      ``y[t * I + i] = x[i * B + t]``. The reverse call uses the inverse
      mapping. See :doc:`/tile/programming_model`.

   compiler-owned scratch
      Temporary shared memory supplied by the compiler for a selected
      cooperative implementation. The provider reports its required size
      and alignment. Tile applications do not pass a scratch descriptor
      to the primitive.

   decoded extent
      The output capacity selected by Run Length Decode's ``decoded_items`` or C++ ``DecodedN``
      template argument. It is independent of the number of input runs.
      A decoded-window offset selects where that output starts in the
      expanded sequence.

   TileLib
      The ``cuda_coop_tile.tilelib`` artifact containing the operation
      declarations and native linkage provider. Both Python and C++
      consumers use the matching TileLib during kernel compilation.

   linkage provider
      The native compiler extension that discovers compatible CUB
      implementations, reports their resources, filters or ranks
      candidates, and compiles the selected implementations. It runs
      during compilation, not as a host implementation of the primitive.

   foreign declaration
      The compiler-facing ABI of a provided operation: its constant
      parameters, Tile types, operand access modes, and results. The
      canonical MLIR interface generates the declarations and schema
      tables consumed by both language paths and the provider.

   foreign call
      A Tile operation referring to a provider-owned declaration and its
      arguments. Python emits a preview foreign call; C++ calls a
      generated declaration. The compiler connects the selected SIMT
      implementation to that call.

   TileLib artifact identity
      The distribution/version and file hash associated with a packaged
      TileLib, or the explicit version token for a development artifact.
      Python compilation caching needs a stable identity for the exact
      library being used. See :doc:`/tile/interop`.

   logical request
      The provider's validated description of the requested operation
      before a concrete implementation is selected. ``LogicalRequest``
      retains logical extents, dtype identity, options, target, and
      Exchange's semantic width. Ordinary physical topology belongs to
      the implementation specification.

   implementation specification
      A validated concrete candidate represented by ``ImplementationSpec``.
      Source generation, scratch probes, and provider caches consume this
      representation after topology and family constraints have been
      checked. See :doc:`/tile/development`.

   LTO-IR
      Link-time-optimization intermediate representation. The provider
      uses NVRTC to compile selected CUB wrappers into LTO-IR for the
      compiler to link into the surrounding Tile kernel.
