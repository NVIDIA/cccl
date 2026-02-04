# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Callable, cast

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_with_registered_key_functions
from .._cccl_interop import (
    call_build,
    get_value_type,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._utils.protocols import (
    get_data_pointer,
    is_device_array,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..iterators._iterators import IteratorBase
from ..op import OpAdapter, OpKind, make_op_adapter
from ..typing import DeviceArrayLike, GpuStruct


def get_init_kind(
    init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
) -> _bindings.InitKind:
    match init_value:
        case None:
            return _bindings.InitKind.NO_INIT
        case _ if is_device_array(init_value):
            return _bindings.InitKind.FUTURE_VALUE_INIT
        case _:
            return _bindings.InitKind.VALUE_INIT


class _Scan:
    __slots__ = [
        "build_result",
        "d_in_cccl",
        "d_out_cccl",
        "init_value_cccl",
        "op_cccl",
        "device_scan_fn",
        "init_kind",
    ]

    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorBase,
        d_out: DeviceArrayLike | IteratorBase,
        op: OpAdapter,
        init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
        force_inclusive: bool,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        self.init_kind = get_init_kind(init_value)

        self.init_value_cccl: _bindings.Iterator | _bindings.Value | None

        match self.init_kind:
            case _bindings.InitKind.NO_INIT:
                self.init_value_cccl = None
                value_type = get_value_type(d_in)
                init_value_type_info = self.d_in_cccl.value_type

            case _bindings.InitKind.FUTURE_VALUE_INIT:
                self.init_value_cccl = cccl.to_cccl_input_iter(
                    cast(DeviceArrayLike, init_value)
                )
                value_type = get_value_type(cast(DeviceArrayLike, init_value))
                init_value_type_info = self.init_value_cccl.value_type

            case _bindings.InitKind.VALUE_INIT:
                init_value_typed = cast(np.ndarray | GpuStruct, init_value)
                self.init_value_cccl = cccl.to_cccl_value(init_value_typed)
                value_type = get_value_type(init_value_typed)
                init_value_type_info = self.init_value_cccl.type

        # Compile the op with value types
        self.op_cccl = op.compile((value_type, value_type), value_type)

        self.build_result = call_build(
            _bindings.DeviceScanBuildResult,
            self.d_in_cccl,
            self.d_out_cccl,
            self.op_cccl,
            init_value_type_info,
            force_inclusive,
            self.init_kind,
        )

        match (force_inclusive, self.init_kind):
            case (True, _bindings.InitKind.FUTURE_VALUE_INIT):
                self.device_scan_fn = self.build_result.compute_inclusive_future_value
            case (True, _bindings.InitKind.VALUE_INIT):
                self.device_scan_fn = self.build_result.compute_inclusive
            case (True, _bindings.InitKind.NO_INIT):
                self.device_scan_fn = self.build_result.compute_inclusive_no_init

            case (False, _bindings.InitKind.FUTURE_VALUE_INIT):
                self.device_scan_fn = self.build_result.compute_exclusive_future_value
            case (False, _bindings.InitKind.VALUE_INIT):
                self.device_scan_fn = self.build_result.compute_exclusive
            case (False, _bindings.InitKind.NO_INIT):
                raise ValueError("Exclusive scan with No init value is not supported")

    def __call__(
        self,
        temp_storage,
        d_in,
        d_out,
        op: Callable | OpAdapter,
        num_items: int,
        init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
        stream=None,
    ):
        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        # Update op state for stateful ops
        op_adapter = make_op_adapter(op)
        op_adapter.update_op_state(self.op_cccl)

        match self.init_kind:
            case _bindings.InitKind.FUTURE_VALUE_INIT:
                # We know that the init_value_cccl is an Iterator, so this cast
                # tells MyPy what the actual type is. cast() is a no-op at runtime,
                # which makes it better than isinstance() since this is a hot path
                # and we have to minimize the work we do prior to calling the
                # kernel.
                self.init_value_cccl = cast(_bindings.Iterator, self.init_value_cccl)
                set_cccl_iterator_state(self.init_value_cccl, init_value)

            case _bindings.InitKind.VALUE_INIT:
                self.init_value_cccl = cast(_bindings.Value, self.init_value_cccl)
                self.init_value_cccl.state = to_cccl_value_state(
                    cast(np.ndarray | GpuStruct, init_value)
                )

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.device_scan_fn(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            self.init_value_cccl,
            stream_handle,
        )
        return temp_storage_bytes


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_registered_key_functions
def make_exclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
):
    """Computes a device-wide scan using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_exclusive_scan`` is used to create an exclusive scan object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/scan/exclusive_scan_object.py
          :language: python
          :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the scan
        op: Callable or OpKind representing the binary operator to apply
        init_value: Numpy array, device array, or GPU struct storing initial value of the scan, or None for no initial value

    Returns:
        A callable object that can be used to perform the scan
    """
    op_adapter = make_op_adapter(op)
    return _Scan(d_in, d_out, op_adapter, init_value, False)


def exclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide exclusive scan.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``exclusive_scan`` is used to compute an exclusive scan with max operation.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/scan/exclusive_scan_max.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array or iterator to store the result of the scan
        op: Binary scan operator
        init_value: Initial value for the scan
        num_items: Number of items to scan
        stream: CUDA stream for the operation (optional)
    """
    scanner = make_exclusive_scan(d_in, d_out, op, init_value)
    tmp_storage_bytes = scanner(None, d_in, d_out, op, num_items, init_value, stream)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    scanner(tmp_storage, d_in, d_out, op, num_items, init_value, stream)


# TODO Figure out `sum` without operator and initial value
# TODO Accept stream
@cache_with_registered_key_functions
def make_inclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
):
    """Computes a device-wide scan using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_inclusive_scan`` is used to create an inclusive scan object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/scan/inclusive_scan_object.py
          :language: python
          :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array that will store the result of the scan
        op: Callable or OpKind representing the binary operator to apply
        init_value: Numpy array, device array, or GPU struct storing initial value of the scan, or None for no initial value

    Returns:
        A callable object that can be used to perform the scan
    """
    op_adapter = make_op_adapter(op)
    return _Scan(d_in, d_out, op_adapter, init_value, True)


def inclusive_scan(
    d_in: DeviceArrayLike | IteratorBase,
    d_out: DeviceArrayLike | IteratorBase,
    op: Callable | OpKind,
    init_value: np.ndarray | DeviceArrayLike | GpuStruct | None,
    num_items: int,
    stream=None,
):
    """
    Performs device-wide inclusive scan.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``inclusive_scan`` is used to compute an inclusive scan (prefix sum).

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/scan/inclusive_scan_custom.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array or iterator to store the result of the scan
        op: Binary scan operator
        init_value: Initial value for the scan
        num_items: Number of items to scan
        stream: CUDA stream for the operation (optional)
    """
    scanner = make_inclusive_scan(d_in, d_out, op, init_value)
    tmp_storage_bytes = scanner(None, d_in, d_out, op, num_items, init_value, stream)
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    scanner(tmp_storage, d_in, d_out, op, num_items, init_value, stream)
