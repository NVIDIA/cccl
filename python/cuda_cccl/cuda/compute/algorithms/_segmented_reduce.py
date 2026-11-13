# Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_build_results, cache_with_registered_key_functions
from .._cccl_interop import (
    get_value_type,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._serialization import BUILD_RESULTS, ITER, OP, VALUE, Serializable
from .._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike, GpuStruct, IteratorT, Operator


class _SegmentedReduce(Serializable):
    __slots__ = [
        "_bound_build_result",
        "build_results",
        "loaded_build_result",
        "d_in_cccl",
        "d_out_cccl",
        "start_offsets_in_cccl",
        "end_offsets_in_cccl",
        "h_init_cccl",
        "op_cccl",
        # Perf: see _Reduce's identical cache for rationale. Stateless op
        # state is a constant (b"" by definition) once seen for a given op
        # object, so __call__ can skip re-deriving it on repeat calls.
        # Holds a strong reference (not a bare id) so identity can never
        # false-positive against a GC'd-and-reused object.
        "_last_stateless_op",
    ]

    __serialization_schema__ = (
        ("d_in_cccl", ITER),
        ("d_out_cccl", ITER),
        ("start_offsets_in_cccl", ITER),
        ("end_offsets_in_cccl", ITER),
        ("h_init_cccl", VALUE),
        ("op_cccl", OP),
        ("build_results", BUILD_RESULTS(_bindings.DeviceSegmentedReduceBuildResult)),
    )

    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        start_offsets_in: DeviceArrayLike | IteratorT,
        end_offsets_in: DeviceArrayLike | IteratorT,
        op: OpAdapter,
        h_init: np.ndarray | GpuStruct,
        compute_capability=None,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)
        self.start_offsets_in_cccl = cccl.to_cccl_input_iter(start_offsets_in)
        self.end_offsets_in_cccl = cccl.to_cccl_input_iter(end_offsets_in)
        self.h_init_cccl = cccl.to_cccl_value(h_init)

        # Compile the op with value types
        value_type = get_value_type(h_init)

        self.op_cccl = op.compile((value_type, value_type), value_type)

        self._last_stateless_op: Callable | OpAdapter | None = None
        self.build_results, self._bound_build_result = cache_build_results(
            _bindings.DeviceSegmentedReduceBuildResult,
            d_in,
            d_out,
            start_offsets_in,
            end_offsets_in,
            op,
            h_init,
            compute_capability=compute_capability,
            builder=lambda: cccl.build_for_ccs(
                _bindings.DeviceSegmentedReduceBuildResult,
                self.d_in_cccl,
                self.d_out_cccl,
                self.start_offsets_in_cccl,
                self.end_offsets_in_cccl,
                self.op_cccl,
                self.h_init_cccl,
                compute_capability=compute_capability,
            ),
        )

    def _after_deserialize(self) -> None:
        # deserialize() bypasses __init__ (see Serializable.deserialize),
        # where this cache is otherwise initialized -- without this, a
        # deserialized _SegmentedReduce would raise AttributeError the
        # first time __call__ reads it.
        self._last_stateless_op = None

    def set_op(self, op: Callable | OpAdapter) -> None:
        """Explicitly rebind this reducer's operator.

        ``execute()`` (unlike ``__call__``) never re-derives op state on its
        own -- if the operator or its captured state has changed since
        construction or the last ``set_op()``, you must call this first, or
        ``execute()`` will silently launch with stale op state. Call this
        only when it has actually changed; it is not needed after every
        ``execute()``.
        """
        op_adapter = make_op_adapter(op)
        self.op_cccl.state = op_adapter.get_state()
        # Perf: invalidate __call__'s stateless-op cache -- a RawOp with
        # state still reports is_stateful=False, so without this, a later
        # __call__ with the same (now stale) op object would see the
        # identity match and skip rebinding, launching with this op's state
        # instead of that op's.
        self._last_stateless_op = None

    def set_h_init(self, h_init) -> None:
        """Explicitly rebind this reducer's initial value.

        The underlying ``Value`` object points directly at ``h_init``'s host
        buffer, so mutating that same array object in place is picked up
        automatically. What is NOT picked up automatically is passing a
        *different* array object as the initial value on a later
        ``execute()`` call: ``execute()`` has no ``h_init`` parameter and
        never re-reads it, so the pointer keeps referencing whichever buffer
        was bound by ``__init__`` or the last ``set_h_init()``. Call this
        whenever you swap in a new ``h_init`` object, or ``execute()`` will
        silently launch against the old one.
        """
        self.h_init_cccl.state = to_cccl_value_state(h_init)

    def execute(
        self,
        *,
        temp_storage,
        d_in,
        d_out,
        num_segments: int,
        start_offsets_in,
        end_offsets_in,
        max_segment_size: int | None = None,
        stream=None,
    ):
        """Minimal per-call path: updates only the ``d_in``/``d_out``/offset
        pointer state and issues the launch. Unlike ``__call__``, this never
        re-derives op state or ``h_init`` state -- construct with the right
        initial values (or call ``set_op``/``set_h_init`` beforehand) and
        reuse this for repeat calls against the same op/h_init. This is the
        deliberately-unsafe-by-default counterpart to ``__call__``: it trades
        the implicit per-call re-derivation (and its correctness margin for
        mutated/varying op or h_init) for the lower fixed cost of a call
        where op and h_init are call-invariant, which is the common case in
        a hot loop. Use ``__call__`` (via ``segmented_reduce``/the object
        returned by ``make_segmented_reduce``) if you can't make that
        guarantee.
        """
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )

        if num_segments > np.iinfo(np.int32).max:
            raise RuntimeError(
                "Segmented sort does not currently support more than 2^31-1 segments."
            )

        if max_segment_size is None:
            max_segment_size = 0  # CCCL.c treats 0 as "not specified"

        if max_segment_size > 0:
            try:
                from .._build_info import USING_V2  # type: ignore[import-not-found]
            except ImportError:
                USING_V2 = False
            if USING_V2:
                import warnings

                warnings.warn(
                    "max_segment_size is not used by the v2 backend and will be ignored",
                    stacklevel=2,
                )

        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        set_cccl_iterator_state(self.start_offsets_in_cccl, start_offsets_in)
        set_cccl_iterator_state(self.end_offsets_in_cccl, end_offsets_in)

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.loaded_build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_segments,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
            self.op_cccl,
            self.h_init_cccl,
            max_segment_size,
            stream_handle,
        )
        return temp_storage_bytes

    def __call__(
        self,
        *,
        temp_storage,
        d_in,
        d_out,
        num_segments: int,
        start_offsets_in,
        end_offsets_in,
        op: Callable | OpAdapter,
        h_init,
        max_segment_size: int | None = None,
        stream=None,
    ):
        # Select (and lazily load) the build result for the current device.
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )

        if num_segments > np.iinfo(np.int32).max:
            raise RuntimeError(
                "Segmented sort does not currently support more than 2^31-1 segments."
            )

        if max_segment_size is None:
            max_segment_size = 0  # CCCL.c treats 0 as "not specified"

        if max_segment_size > 0:
            try:
                from .._build_info import USING_V2  # type: ignore[import-not-found]
            except ImportError:
                USING_V2 = False
            if USING_V2:
                import warnings

                warnings.warn(
                    "max_segment_size is not used by the v2 backend and will be ignored",
                    stacklevel=4,
                )

        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)
        set_cccl_iterator_state(self.start_offsets_in_cccl, start_offsets_in)
        set_cccl_iterator_state(self.end_offsets_in_cccl, end_offsets_in)

        # Update op state for stateful ops; see _Reduce.__call__ for the
        # stateless-op caching rationale.
        op_adapter = make_op_adapter(op)
        if op_adapter.is_stateful or op is not self._last_stateless_op:
            self.op_cccl.state = op_adapter.get_state()
            self._last_stateless_op = None if op_adapter.is_stateful else op

        self.h_init_cccl.state = to_cccl_value_state(h_init)

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.loaded_build_result.compute(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_segments,
            self.start_offsets_in_cccl,
            self.end_offsets_in_cccl,
            self.op_cccl,
            self.h_init_cccl,
            max_segment_size,
            stream_handle,
        )
        return temp_storage_bytes


@cache_with_registered_key_functions
def make_segmented_reduce(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    start_offsets_in: DeviceArrayLike | IteratorT,
    end_offsets_in: DeviceArrayLike | IteratorT,
    op: Operator,
    h_init: np.ndarray | GpuStruct,
    compute_capability=None,
):
    """Computes a device-wide segmented reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_segmented_reduce`` is used to create a segmented reduction object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/segmented/segmented_reduce_object.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the reduction
        start_offsets_in: Device array or iterator containing offsets to start of segments
        end_offsets_in: Device array or iterator containing offsets to end of segments
        op: Binary operator to apply.
            The signature is ``(T, T) -> T``, where ``T`` is
            the data type of the initial value ``h_init``.
        init: Numpy array storing initial value of the reduction
        compute_capability: Compute capability, or list of capabilities, to
            build for ahead of time. Accepts a packed int (e.g. ``90``), a
            ``(major, minor)`` pair, a string (e.g. ``"9.0"``), or a list
            thereof. When ``None`` (the default), the current device's
            architecture is used.

    Returns:
        A callable object that can be used to perform the reduction
    """
    op_adapter = make_op_adapter(op)
    return _SegmentedReduce(
        d_in,
        d_out,
        start_offsets_in,
        end_offsets_in,
        op_adapter,
        h_init,
        compute_capability=compute_capability,
    )


def segmented_reduce(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    num_segments: int,
    start_offsets_in: DeviceArrayLike | IteratorT,
    end_offsets_in: DeviceArrayLike | IteratorT,
    op: Operator,
    h_init: np.ndarray | GpuStruct,
    max_segment_size: int | None = None,
    stream=None,
):
    """
    Performs device-wide segmented reduction.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``segmented_reduce`` is used to compute the minimum value of segments in a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/segmented/segmented_reduce_basic.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array to store the result of the reduction for each segment
        num_segments: Number of segments to reduce
        start_offsets_in: Device array or iterator containing the sequence of beginning offsets
        end_offsets_in: Device array or iterator containing the sequence of ending offsets
        op: Binary operator to apply.
            The signature is ``(T, T) -> T``, where ``T`` is
            the data type of the initial value ``h_init``.
        h_init: Initial value for the reduction
        max_segment_size: The number of elements in the largest segment (optional)
            If provided, this information is used to dispatch to the
            optimal kernel for best performance.
        stream: CUDA stream for the operation (optional)
    """
    reducer = make_segmented_reduce(
        d_in=d_in,
        d_out=d_out,
        start_offsets_in=start_offsets_in,
        end_offsets_in=end_offsets_in,
        op=op,
        h_init=h_init,
    )
    tmp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start_offsets_in,
        end_offsets_in=end_offsets_in,
        op=op,
        h_init=h_init,
        max_segment_size=max_segment_size,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(
        temp_storage=tmp_storage,
        d_in=d_in,
        d_out=d_out,
        num_segments=num_segments,
        start_offsets_in=start_offsets_in,
        end_offsets_in=end_offsets_in,
        op=op,
        h_init=h_init,
        max_segment_size=max_segment_size,
        stream=stream,
    )
