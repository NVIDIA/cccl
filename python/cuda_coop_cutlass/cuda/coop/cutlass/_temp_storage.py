# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Some primitives need scratch space in shared memory while they run.

A block scan does, and so do the transpose forms of Load and Store. Leave
``temp_storage`` out of a call and the compiler provides the scratch that
call needs and adds a block barrier after it.

Pass a :class:`~cuda.coop.cutlass.TempStorage` to choose which calls share
bytes and where the barriers go. Construct it inside the kernel and give the
same object to every call that should use it. Here three calls share one
region. Load and Store use no scratch with their default algorithm, so this
kernel asks for ``algorithm="transpose"``.

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
    :language: python
    :start-after: # cutlass-concept-temp-storage-begin
    :end-before: # cutlass-concept-temp-storage-end

``sharing`` sets how the calls are laid out inside the region. The sizes are
those of the kernel above on a block of 128 threads with four ``Int32`` items
each, where Load, Scan and Store need 2,048, 672 and 2,064 bytes with
CUDA 13.3. Other block shapes, element types and CUB versions give other
sizes.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * -
     - ``sharing="shared"`` (default)
     - ``sharing="exclusive"``
   * - Where a call's scratch goes
     - Offset 0, over the previous call's
     - Its own slice, one per call site
   * - Size of the region
     - The largest call: 2,064 bytes
     - All calls together: 4,784 bytes
   * - When bytes are reused
     - At the next call
     - When the same call site runs again, as in a loop

Each ``TempStorage`` has its own region. Calls that must keep their scratch
apart take different objects.

Before scratch is reused, every thread in the block must be finished with
what was there. ``auto_sync=True`` adds a block barrier after each call that
used the region. The default is ``auto_sync=False``, as in CUB: you call
:meth:`~cuda.coop.cutlass.TempStorage.sync` before the bytes are reused. This
is the body of the same kernel with its barriers written by hand. These
barriers are for scratch. Shared data of your own needs its own
synchronization.

`Full example and test on GitHub
<https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py>`__

.. literalinclude::
    ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
    :language: python
    :start-after: # cutlass-concept-temp-storage-manual-sync-begin
    :end-before: # cutlass-concept-temp-storage-manual-sync-end
    :dedent: 4

Block primitives that use scratch accept a ``TempStorage``: transpose Load
and Store, Reduce, Scan, Merge Sort, Radix Sort, TopK, Adjacent Difference,
Discontinuity, Histogram and Run Length Decode. Warp primitives manage their
own scratch and reject one.

The region lasts for one launch of one block. To keep your own data in it,
:meth:`~cuda.coop.cutlass.TempStorage.reserve` returns a typed shared tensor
from the same region, and :meth:`~cuda.coop.cutlass.TempStorage.release`
hands the whole region back as ordinary shared memory once the primitives
are finished with it.
"""

# Implementation notes. The descriptor holds compile-time policy and identity
# and no device pointer. Finalization allocates storage after C++ layout
# probes establish its size. The MLIR hooks carry an analysis token through
# branches and loops, preserving aliases until finalization checks identity
# and terminal release.

from enum import Enum

from ._validation import _normalize_alignment

# Provider capacities and released extents use signed i32 operands.
_MAX_STORAGE_SIZE_IN_BYTES = (1 << 31) - 1
_MAX_STORAGE_ALIGNMENT = 1024


def _validate_storage_size(size_in_bytes):
    if size_in_bytes > _MAX_STORAGE_SIZE_IN_BYTES:
        raise ValueError(
            "TempStorage size_in_bytes must fit a signed 32-bit capacity "
            f"(at most {_MAX_STORAGE_SIZE_IN_BYTES} bytes)."
        )


def _normalize_storage_alignment(alignment):
    alignment = _normalize_alignment(alignment)
    if alignment is not None and alignment > _MAX_STORAGE_ALIGNMENT:
        raise ValueError("TempStorage alignment must not exceed 1024 bytes.")
    return alignment


class TempStorage:
    """A region of scratch shared memory that primitives in one kernel share.

    Construct one inside the kernel and pass it as ``temp_storage`` to each
    primitive that should use it.

    Attributes
    ----------
    size_in_bytes : int or None
        Capacity of the region in bytes. ``None``, the default, lets the
        compiler size it for every use, including reserved arrays. An
        explicit capacity must be at least that large and fit a signed
        32-bit integer. The device's shared-memory limit still applies.
    alignment : int or None
        Minimum alignment of the region in bytes, a power of two up to 1024.
        The compiler raises it when a primitive needs more.
    auto_sync : bool
        ``True`` adds a block barrier after each call that uses the region.
        The default, ``False``, leaves the barriers to the kernel: call
        :meth:`~cuda.coop.cutlass.TempStorage.sync` before scratch is
        reused. ``None`` means ``False``.
    sharing : {"shared", "exclusive"}
        ``"shared"``, the default, places every call and reservation at the
        start of the region, which is sized for the largest of them.
        ``"exclusive"`` gives each call site and reservation its own slice,
        in trace order. A call site that runs again, as in a loop, reuses
        its slice.

    Notes
    -----
    * Two objects are two regions, even when their arguments are equal.
      Pass the same object to the calls that should share.
    * An object created on the host works for straight-line calls. It
      cannot be carried through a runtime ``if`` or loop and cannot be
      released.
    * Inside a runtime ``if`` or loop, keep using the same object. Choosing
      between two objects at run time, or rebinding the name to another
      one, is a compile error. An object created and used entirely inside
      one branch or one loop iteration is fine.
    * Put a conditional call in an ``if`` statement whose condition is the
      same for every thread in the group. A conditional expression is
      rejected.

    The Developer Guide describes :ref:`how the compiler sizes and places
    the region <coop-cutlass-scratch-allocation>`.
    """

    def __init__(
        self,
        size_in_bytes=None,
        *,
        alignment=None,
        auto_sync=False,
        sharing="shared",
    ):
        if size_in_bytes is not None:
            if not isinstance(size_in_bytes, int) or isinstance(
                size_in_bytes, bool
            ):
                raise TypeError(
                    "TempStorage size_in_bytes must be an integer or None."
                )
            if size_in_bytes <= 0:
                raise ValueError(
                    "TempStorage size_in_bytes must be a positive integer."
                )
            _validate_storage_size(size_in_bytes)

        alignment = _normalize_storage_alignment(alignment)

        if not isinstance(sharing, str) or isinstance(sharing, Enum):
            raise TypeError(
                "TempStorage sharing must be a string: 'shared' or 'exclusive'."
            )
        sharing_value = sharing.strip().lower()
        if sharing_value not in {"shared", "exclusive"}:
            raise ValueError(
                "TempStorage sharing must be 'shared' or 'exclusive'."
            )

        if auto_sync is not None and not isinstance(auto_sync, bool):
            raise TypeError("TempStorage auto_sync must be None/True/False.")

        self.size_in_bytes = size_in_bytes
        self.alignment = alignment
        self.sharing = sharing_value
        # Sharing selects the slice layout; synchronization is independent.
        # The caller synchronizes before reuse unless auto_sync=True.
        self.auto_sync = False if auto_sync is None else auto_sync
        from ._compiler._lifetime import register_creation

        self._identity_owner = self
        self._identity_token = register_creation(self)

    def __extract_mlir_values__(self):
        if self._identity_token is None:
            from cutlass.base_dsl.common import DSLUserCodeError

            from ._compiler import _state

            if _state._active_trace_module_op() is not None:
                raise DSLUserCodeError(
                    "Host-created TempStorage cannot be carried through "
                    "runtime control flow; construct the descriptor inside "
                    "the kernel."
                )
            return []
        return [self._identity_token]

    def __new_from_mlir_values__(self, values):
        if self._identity_token is None:
            if values:
                raise ValueError("Host TempStorage has no runtime MLIR values")
            return self
        if len(values) != 1:
            raise ValueError(
                "TempStorage requires one descriptor identity token"
            )
        result = object.__new__(type(self))
        result.__dict__.update(self.__dict__)
        result._identity_token = (
            values[0].ir_value()
            if hasattr(values[0], "ir_value")
            else values[0]
        )
        return result

    @property
    def capacity_size_in_bytes(self):
        """Return explicit capacity without resolving an inferred size."""
        return self.size_in_bytes

    @property
    def is_deferred(self):
        """Indicate that layout resolution waits for trace finalization."""
        return True

    def sync(self):
        """Wait at a block barrier before scratch is reused.

        Use it with ``auto_sync=False``, between a call that used the region
        and the next use of the same bytes. Every thread in the block must
        reach the call.
        """
        from cutlass.cute.arch import sync_threads

        from ._compiler._lifetime import register_use

        register_use(self, "sync()")
        sync_threads()

    def reserve(self, num_elems, dtype, *, alignment=None):
        """Return a one-dimensional shared tensor in this region for your data.

        ``num_elems``, ``dtype`` and the optional byte ``alignment`` are
        compile-time constants. Supported element types are the 8-, 16-,
        32- and 64-bit signed and unsigned integers, ``Float16``,
        ``BFloat16``, ``Float32`` and ``Float64``, or the matching NumPy
        types.

        A reservation follows the object's ``sharing``. With ``"shared"``
        it overlaps primitive scratch and other reservations. With
        ``"exclusive"`` it has its own slice. Each call site has one view,
        also when it runs again in a loop. Other ``TempStorage`` objects
        and tensors from CuTe's ``SmemAllocator`` occupy separate bytes.

        Reservations need ``auto_sync=False``. Before a primitive or a later
        phase reuses bytes that a reservation overlaps, finish with them,
        for example by copying the values into registers, and call
        :meth:`~cuda.coop.cutlass.TempStorage.sync`. ``sync`` is a block
        barrier. It does not wait for an asynchronous library operation to
        complete.

        This kernel hands a reserved tensor to native CuTe code, reuses its
        bytes for a block sum, and checks that a tensor from
        ``SmemAllocator`` keeps its values.

        `Full example on GitHub
        <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/examples/shared_workspace.py>`__

        .. literalinclude::
            ../../../python/cuda_coop_cutlass/examples/shared_workspace.py
            :language: python
            :start-after: docs: start cutlass-shared-workspace
            :end-before: docs: end cutlass-shared-workspace
            :dedent: 4
        """

        from ._compiler._storage import reserve_temp_storage_tensor

        return reserve_temp_storage_tensor(
            self, num_elems, dtype, alignment=alignment
        )

    def release(self, dtype=None):
        """Return the whole region as an ordinary shared tensor.

        ``dtype`` is a supported compile-time scalar type. Without it the
        tensor holds bytes (``Uint8``). The tensor covers the full capacity:
        every exclusive slice, any padding, and an explicit
        ``size_in_bytes``. That size must be a multiple of the element size
        and the region's alignment must suit the type. An inferred region
        that nothing uses is empty and is rejected. The object must have
        been constructed inside the kernel.

        ``release`` retires the object and every alias of it. A primitive,
        ``reserve``, ``sync`` or second ``release`` that uses it afterwards
        is a compile error, on the same path and on a later loop iteration.
        An object constructed inside a loop body is a new one each
        iteration.

        ``release`` adds no barrier and waits for nothing. Finish and
        synchronize the earlier work before writing through the returned
        tensor.

        This kernel sums a block, then takes the region back and uses it to
        hand each thread the value of the thread opposite it.

        `Full example and test on GitHub
        <https://github.com/NVIDIA/cccl/blob/main/python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py>`__

        .. literalinclude::
            ../../../python/cuda_coop_cutlass/tests/runtime/test_concepts_documentation.py
            :language: python
            :start-after: # cutlass-concept-temp-storage-release-begin
            :end-before: # cutlass-concept-temp-storage-release-end
        """
        from ._compiler._storage import release_temp_storage_tensor

        return release_temp_storage_tensor(self, dtype)


__all__ = ["TempStorage"]
