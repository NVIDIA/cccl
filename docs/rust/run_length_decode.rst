.. Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
.. SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Decoding run lengths
========================

This example requires Oxide's generic native-registration support. The revision
currently pinned by the Cargo manifests does not yet include that support.

``run_length_decode`` returns a fresh window from a run-length encoded stream.
``run_length_decode_into`` writes the complete stream into a checked destination
and returns its total length to every Block member:

.. code-block:: rust

   let window = coop::run_length_decode(
       block, &values, &lengths, coop::RunLengthDecodeOptions::<3>::default(),
   );

The required decoded-items count, here three, is independent of the number of
input runs per thread. Values support all ten ``Numeric`` types; lengths support
the eight fixed-width signed and unsigned integer types. Both inputs share their
per-thread extent, retain their independent alignments and remain unchanged.
Every input slot must be initialized, including values for zero-length padding.

A complete program
------------------

The helper keeps the input item count generic. It retains a fresh window while
Into decodes the complete stream through several internal windows:

.. literalinclude:: ../../rust/cuda-coop/examples/run_length_decode/helper/src/lib.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

The host instantiates one and four input runs per thread on two 32-thread Blocks.
It checks decoded values, all-lane totals, input preservation, explicit zero
window tails and untouched destination margins:

.. literalinclude:: ../../rust/cuda-coop/examples/run_length_decode/src/main.rs
   :language: rust
   :start-after: // SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

With an Oxide revision containing that support and ``nightly-2026-10-11``
installed, run from a complete CCCL checkout:

.. code-block:: shell

   cd rust/cuda-coop/examples/run_length_decode
   CUDA_HOME=/usr/local/cuda-13.3 CUDA_PATH=/usr/local/cuda-13.3 \
     cargo oxide run --arch sm_120 --materialize-cubin

Choose the architecture for your GPU. ``auto_shared`` uses the compiler's
complete temporary-storage plan.

Windows and complete output
---------------------------

``RunLengthDecodeOptions::<D>::default()`` selects a window starting at decoded
stream offset zero. ``decoded_window_offset(offset)`` accepts a uniform integer
control, and ``decoded_window_offset_const::<OFFSET>()`` selects a static u64
offset. Each member receives D consecutive decoded values in blocked order.
The fresh result preserves the value payload's alignment. All D slots are valid:
positions after the stream contain explicit zeros. An empty stream, or a window
starting at or beyond its end, returns only zeros. Wide unsigned offsets retain
their full range through validation.

``RunLengthDecodeIntoOptions::<D>`` selects D items per internal window; it does
not limit the complete output size. ``destination_offset`` and
``destination_offset_const`` select the destination start, defaulting to zero.
The writable ``DisjointSlice<T,CollectiveTiles<CAPACITY>>`` supplies each Block's
ownership region independently of both the input extent and D. The complete
interval must fit that region and the actual allocation before any write occurs.
Only that interval changes. Empty input writes nothing, but its destination
offset must still lie within the allocation capacity.

Into returns the same numerical u32 total to every member. The returned total
may be stored or reported, but is not currently supported as a later collective
control, including after explicit Block completion.

Input and participation rules
-----------------------------

Positive lengths must precede all trailing zero padding across the Block.
All-zero lengths represent an empty stream. Negative lengths, a positive length
after a zero, missing input, or a total exceeding u32::MAX trap before decoding.
Negative runtime offsets and invalid destination intervals also trap. Counts
and offsets are checked before narrowing; validity checks precede numeric reads.

Both operations support complete one-dimensional Blocks of 1 through 1024
threads, including non-power-of-two sizes. Every member participates with the
same controls and storage choice. Input and decoded per-thread extents must be
positive, and both Block-sized tiles must fit a signed i32 capacity.

Scratch and completion
----------------------

Implicit scratch uses compiler-managed reuse. ``temp_storage(&mut scratch)``
borrows existing ``TempStorage``; the offset and storage builders preserve each
other in either order. Manual scratch requires an explicit Block barrier before
conflicting reuse, including empty operations. Auto-sync storage supplies its
trailing barrier. Into prepares one run table and retains it throughout all its
internal windows. Fresh Window results do not alias that table or either input.

Into is a global write. Complete it before a conflicting collective reads or
writes the destination; native internal barriers do not complete unrelated
resource accesses. See :doc:`payload`, :doc:`groups` and :doc:`load_store` for
validity, participation and storage rules.
