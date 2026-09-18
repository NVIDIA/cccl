# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable, cast

import numpy as np

from .. import _bindings, types
from .. import _cccl_interop as cccl
from .._caching import cache_build_results, cache_with_registered_key_functions
from .._cccl_interop import (
    get_value_type,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._serialization import (
    BOOL,
    BUILD_RESULTS,
    CONDITIONAL,
    ENUM,
    ITER,
    OP,
    VALUE,
    Serializable,
)
from .._utils import get_init_kind
from .._utils.protocols import (
    get_data_pointer,
    validate_and_get_stream,
)
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..op import OpAdapter, make_op_adapter
from ..typing import DeviceArrayLike, GpuStruct, IteratorT, Operator


class _ScanByKey(Serializable):
    """Device-wide scan over runs of equal adjacent keys.

    Two operators travel with the object and must stay distinct: ``op`` combines values inside a run,
    while ``equality_op`` decides where a run ends. The initial value only exists for the exclusive
    form -- CUB's ``InclusiveScanByKey`` has no init parameter, which is why ``force_inclusive`` and
    ``init_kind == NO_INIT`` always travel together here.
    """

    __slots__ = [
        "_bound_build_result",
        "build_results",
        "loaded_build_result",
        "d_keys_in_cccl",
        "d_values_in_cccl",
        "d_values_out_cccl",
        "init_value_cccl",
        "op_cccl",
        "equality_op_cccl",
        "init_kind",
        "force_inclusive",
        "device_scan_by_key_fn",
    ]

    __serialization_schema__ = (
        ("init_kind", ENUM(_bindings.InitKind)),
        ("force_inclusive", BOOL),
        ("d_keys_in_cccl", ITER),
        ("d_values_in_cccl", ITER),
        ("d_values_out_cccl", ITER),
        ("op_cccl", OP),
        ("equality_op_cccl", OP),
        (
            "init_value_cccl",
            CONDITIONAL(
                "init_kind",
                {
                    _bindings.InitKind.NO_INIT: None,
                    _bindings.InitKind.VALUE_INIT: VALUE,
                },
            ),
        ),
        ("build_results", BUILD_RESULTS(_bindings.DeviceScanByKeyBuildResult)),
    )

    def __init__(
        self,
        d_keys_in: DeviceArrayLike | IteratorT,
        d_values_in: DeviceArrayLike | IteratorT,
        d_values_out: DeviceArrayLike | IteratorT,
        op: OpAdapter,
        equality_op: OpAdapter,
        init_value: np.ndarray | GpuStruct | None,
        force_inclusive: bool,
        compute_capability=None,
    ):
        self.d_keys_in_cccl = cccl.to_cccl_input_iter(d_keys_in)
        self.d_values_in_cccl = cccl.to_cccl_input_iter(d_values_in)
        self.d_values_out_cccl = cccl.to_cccl_output_iter(d_values_out)

        self.init_kind = get_init_kind(init_value)
        if self.init_kind == _bindings.InitKind.FUTURE_VALUE_INIT:
            # The C API carries a single host-side init value. A device-resident (future) init would
            # need a different CUB overload and a different run entry point; refusing is better than
            # silently reading whatever happens to sit at that address.
            raise NotImplementedError(
                "scan-by-key does not support a device-array (future value) init; "
                "pass a host value (e.g. a numpy scalar) instead"
            )

        key_value_type = get_value_type(d_keys_in)

        self.init_value_cccl: _bindings.Value | None
        if self.init_kind == _bindings.InitKind.NO_INIT:
            self.init_value_cccl = None
            init_value_type_info = self.d_values_in_cccl.value_type
            value_type = get_value_type(d_values_in)
        else:
            init_value_typed = cast(np.ndarray | GpuStruct, init_value)
            self.init_value_cccl = cccl.to_cccl_value(init_value_typed)
            init_value_type_info = self.init_value_cccl.type
            value_type = get_value_type(init_value_typed)

        self.force_inclusive = force_inclusive

        # The scan operator works on values; the equality operator works on keys and yields a bool.
        self.op_cccl = op.compile((value_type, value_type), value_type)
        self.equality_op_cccl = equality_op.compile(
            (key_value_type, key_value_type), types.uint8
        )

        self.build_results, self._bound_build_result = cache_build_results(
            _bindings.DeviceScanByKeyBuildResult,
            d_keys_in,
            d_values_in,
            d_values_out,
            op,
            equality_op,
            init_value,
            force_inclusive,
            self.init_kind,
            compute_capability=compute_capability,
            builder=lambda: cccl.build_for_ccs(
                _bindings.DeviceScanByKeyBuildResult,
                self.d_keys_in_cccl,
                self.d_values_in_cccl,
                self.d_values_out_cccl,
                self.op_cccl,
                self.equality_op_cccl,
                init_value_type_info,
                force_inclusive,
                self.init_kind,
                compute_capability=compute_capability,
            ),
        )

    def _bind_device_scan_by_key_fn(self) -> None:
        # Derived from force_inclusive + init_kind on the loaded build result, not serialized.
        match (self.force_inclusive, self.init_kind):
            case (True, _bindings.InitKind.NO_INIT):
                self.device_scan_by_key_fn = (
                    self.loaded_build_result.compute_inclusive
                )
            case (False, _bindings.InitKind.VALUE_INIT):
                self.device_scan_by_key_fn = (
                    self.loaded_build_result.compute_exclusive
                )
            case (True, _):
                raise ValueError(
                    "Inclusive scan-by-key has no init value; build it with init_value=None"
                )
            case (False, _bindings.InitKind.NO_INIT):
                raise ValueError(
                    "Exclusive scan-by-key requires an init value"
                )

    def __call__(
        self,
        *,
        temp_storage,
        d_keys_in,
        d_values_in,
        d_values_out,
        op: Callable | OpAdapter,
        equality_op: Callable | OpAdapter,
        init_value: np.ndarray | GpuStruct | None,
        num_items: int,
        stream=None,
    ):
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )
        self._bind_device_scan_by_key_fn()

        set_cccl_iterator_state(self.d_keys_in_cccl, d_keys_in)
        set_cccl_iterator_state(self.d_values_in_cccl, d_values_in)
        set_cccl_iterator_state(self.d_values_out_cccl, d_values_out)

        # Update op state for stateful ops; the two operators carry independent state.
        self.op_cccl.state = make_op_adapter(op).get_state()
        self.equality_op_cccl.state = make_op_adapter(equality_op).get_state()

        if self.init_kind == _bindings.InitKind.VALUE_INIT:
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

        # The inclusive and exclusive native entry points have different arities: only the exclusive
        # form carries an init value (CUB's InclusiveScanByKey has no init parameter).
        if self.init_kind == _bindings.InitKind.VALUE_INIT:
            temp_storage_bytes = self.device_scan_by_key_fn(
                d_temp_storage,
                temp_storage_bytes,
                self.d_keys_in_cccl,
                self.d_values_in_cccl,
                self.d_values_out_cccl,
                num_items,
                self.op_cccl,
                self.equality_op_cccl,
                self.init_value_cccl,
                stream_handle,
            )
        else:
            temp_storage_bytes = self.device_scan_by_key_fn(
                d_temp_storage,
                temp_storage_bytes,
                self.d_keys_in_cccl,
                self.d_values_in_cccl,
                self.d_values_out_cccl,
                num_items,
                self.op_cccl,
                self.equality_op_cccl,
                stream_handle,
            )
        return temp_storage_bytes


@cache_with_registered_key_functions
def make_exclusive_scan_by_key(
    *,
    d_keys_in: DeviceArrayLike | IteratorT,
    d_values_in: DeviceArrayLike | IteratorT,
    d_values_out: DeviceArrayLike | IteratorT,
    op: Operator,
    equality_op: Operator,
    init_value: np.ndarray | GpuStruct,
    compute_capability=None,
):
    """Builds a reusable exclusive scan-by-key object.

    The scan runs over runs of equal *adjacent* keys; the sequence is not sorted or grouped for you.
    ``init_value`` restarts at the head of every run.

    Args:
        d_keys_in: Device array of keys identifying the runs
        d_values_in: Device array of values to scan
        d_values_out: Device array that will store the scanned values
        op: Binary scan operator, ``(T, T) -> T``
        equality_op: Key equality operator, ``(K, K) -> bool``
        init_value: Host value used to seed each run
        compute_capability: Compute capability to build for ahead of time

    Returns:
        A callable object that performs the scan
    """
    return _ScanByKey(
        d_keys_in,
        d_values_in,
        d_values_out,
        make_op_adapter(op),
        make_op_adapter(equality_op),
        init_value,
        False,
        compute_capability=compute_capability,
    )


def exclusive_scan_by_key(
    *,
    d_keys_in: DeviceArrayLike | IteratorT,
    d_values_in: DeviceArrayLike | IteratorT,
    d_values_out: DeviceArrayLike | IteratorT,
    op: Operator,
    equality_op: Operator,
    init_value: np.ndarray | GpuStruct,
    num_items: int,
    stream=None,
):
    """Performs a device-wide exclusive scan-by-key, allocating temporary storage internally.

    ``init_value`` is applied at the head of every run of equal adjacent keys.
    """
    scanner = make_exclusive_scan_by_key(
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_values_out=d_values_out,
        op=op,
        equality_op=equality_op,
        init_value=init_value,
    )
    tmp_storage_bytes = scanner(
        temp_storage=None,
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_values_out=d_values_out,
        op=op,
        equality_op=equality_op,
        init_value=init_value,
        num_items=num_items,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    scanner(
        temp_storage=tmp_storage,
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_values_out=d_values_out,
        op=op,
        equality_op=equality_op,
        init_value=init_value,
        num_items=num_items,
        stream=stream,
    )


@cache_with_registered_key_functions
def make_inclusive_scan_by_key(
    *,
    d_keys_in: DeviceArrayLike | IteratorT,
    d_values_in: DeviceArrayLike | IteratorT,
    d_values_out: DeviceArrayLike | IteratorT,
    op: Operator,
    equality_op: Operator,
    compute_capability=None,
):
    """Builds a reusable inclusive scan-by-key object.

    CUB's ``InclusiveScanByKey`` takes no initial value, so neither does this entry point.

    Args:
        d_keys_in: Device array of keys identifying the runs
        d_values_in: Device array of values to scan
        d_values_out: Device array that will store the scanned values
        op: Binary scan operator, ``(T, T) -> T``
        equality_op: Key equality operator, ``(K, K) -> bool``
        compute_capability: Compute capability to build for ahead of time

    Returns:
        A callable object that performs the scan
    """
    return _ScanByKey(
        d_keys_in,
        d_values_in,
        d_values_out,
        make_op_adapter(op),
        make_op_adapter(equality_op),
        None,
        True,
        compute_capability=compute_capability,
    )


def inclusive_scan_by_key(
    *,
    d_keys_in: DeviceArrayLike | IteratorT,
    d_values_in: DeviceArrayLike | IteratorT,
    d_values_out: DeviceArrayLike | IteratorT,
    op: Operator,
    equality_op: Operator,
    num_items: int,
    stream=None,
):
    """Performs a device-wide inclusive scan-by-key, allocating temporary storage internally."""
    scanner = make_inclusive_scan_by_key(
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_values_out=d_values_out,
        op=op,
        equality_op=equality_op,
    )
    tmp_storage_bytes = scanner(
        temp_storage=None,
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_values_out=d_values_out,
        op=op,
        equality_op=equality_op,
        init_value=None,
        num_items=num_items,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    scanner(
        temp_storage=tmp_storage,
        d_keys_in=d_keys_in,
        d_values_in=d_values_in,
        d_values_out=d_values_out,
        op=op,
        equality_op=equality_op,
        init_value=None,
        num_items=num_items,
        stream=stream,
    )
