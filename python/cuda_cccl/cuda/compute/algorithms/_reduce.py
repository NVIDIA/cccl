# Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Callable, cast

import numpy as np

from .. import _bindings
from .. import _cccl_interop as cccl
from .._caching import cache_build_results, cache_with_registered_key_functions
from .._cccl_interop import (
    get_value_type,
    set_cccl_iterator_state,
    to_cccl_value_state,
)
from .._serialization import (
    BUILD_RESULTS,
    CONDITIONAL,
    ENUM,
    ITER,
    OP,
    VALUE,
    Serializable,
)
from .._utils import get_init_kind
from .._utils.protocols import get_data_pointer, get_dtype, validate_and_get_stream
from .._utils.temp_storage_buffer import TempStorageBuffer
from ..determinism import Determinism
from ..op import OpAdapter, make_op_adapter
from ..typing import (
    DeviceArrayLike,
    GpuStruct,
    IteratorBase,
    IteratorT,
    Operator,
    _Struct,
)


class _Reduce(Serializable):
    __slots__ = [
        "_bound_build_result",
        "d_in_cccl",
        "d_out_cccl",
        "h_init_cccl",
        "op_cccl",
        "init_kind",
        "build_results",
        "loaded_build_result",
        "device_reduce_fn",
    ]

    __serialization_schema__ = (
        ("init_kind", ENUM(_bindings.InitKind)),
        ("d_in_cccl", ITER),
        ("d_out_cccl", ITER),
        ("op_cccl", OP),
        (
            "h_init_cccl",
            CONDITIONAL(
                "init_kind",
                {
                    _bindings.InitKind.NO_INIT: None,
                    _bindings.InitKind.VALUE_INIT: VALUE,
                },
            ),
        ),
        ("build_results", BUILD_RESULTS(_bindings.DeviceReduceBuildResult)),
    )

    # TODO: constructor shouldn't require concrete `d_in`, `d_out`:
    def __init__(
        self,
        d_in: DeviceArrayLike | IteratorT,
        d_out: DeviceArrayLike | IteratorT,
        op: OpAdapter,
        h_init: np.ndarray | GpuStruct | None,
        determinism: Determinism,
        compute_capability=None,
    ):
        self.d_in_cccl = cccl.to_cccl_input_iter(d_in)
        self.d_out_cccl = cccl.to_cccl_output_iter(d_out)

        self.init_kind = get_init_kind(h_init)

        self.h_init_cccl: _bindings.Value | None

        match self.init_kind:
            case _bindings.InitKind.NO_INIT:
                # The non-deterministic (atomic) kernel folds the initial value
                # into block zero's aggregate, so it requires one to exist.
                if Determinism(determinism) is Determinism.NOT_GUARANTEED:
                    raise ValueError(
                        "h_init=None is not supported with Determinism.NOT_GUARANTEED"
                    )
                self.h_init_cccl = None
                value_type = get_value_type(d_in)
                init_value_type_info = self.d_in_cccl.value_type

            case _bindings.InitKind.FUTURE_VALUE_INIT:
                raise ValueError(
                    "Passing a device array as h_init is not supported for reduction"
                )

            case _bindings.InitKind.VALUE_INIT:
                h_init_typed = cast(np.ndarray | GpuStruct, h_init)
                self.h_init_cccl = cccl.to_cccl_value(h_init_typed)
                value_type = get_value_type(h_init_typed)
                init_value_type_info = self.h_init_cccl.type

        # Compile the op with value types
        self.op_cccl = op.compile((value_type, value_type), value_type)

        # loaded_build_result / device_reduce_fn are bound lazily on the first
        # __call__ (see _bind_device_reduce_fn).
        self.build_results, self._bound_build_result = cache_build_results(
            _bindings.DeviceReduceBuildResult,
            d_in,
            d_out,
            op,
            h_init,
            determinism,
            self.init_kind,
            compute_capability=compute_capability,
            builder=lambda: cccl.build_for_ccs(
                _bindings.DeviceReduceBuildResult,
                self.d_in_cccl,
                self.d_out_cccl,
                self.op_cccl,
                init_value_type_info,
                self.init_kind,
                determinism,
                compute_capability=compute_capability,
            ),
        )

    def _bind_device_reduce_fn(self) -> None:
        # Derived from the loaded build result (not serialized); bound at __call__
        # once resolve_build_result picks + loads the current device's build result.
        # compute() handles both init kinds: it ignores h_init for NO_INIT builds.
        if (
            Determinism(self.loaded_build_result.determinism)
            is Determinism.NOT_GUARANTEED
        ):
            self.device_reduce_fn = self.loaded_build_result.compute_nondeterministic
        else:
            self.device_reduce_fn = self.loaded_build_result.compute

    def __call__(
        self,
        *,
        temp_storage,
        d_in,
        d_out,
        num_items: int,
        op: Callable | OpAdapter,
        h_init: np.ndarray | GpuStruct | None,
        stream=None,
    ):
        # Select (and lazily load) the current device's build result, then bind the
        # derived compute fn from it.
        self.loaded_build_result = cccl.resolve_build_result(
            self.build_results, self._bound_build_result
        )
        self._bind_device_reduce_fn()

        set_cccl_iterator_state(self.d_in_cccl, d_in)
        set_cccl_iterator_state(self.d_out_cccl, d_out)

        # Update op state for stateful ops
        op_adapter = make_op_adapter(op)
        self.op_cccl.state = op_adapter.get_state()

        if self.init_kind is _bindings.InitKind.VALUE_INIT:
            # We know that h_init_cccl is a Value here, so this cast tells MyPy
            # what the actual type is. cast() is a no-op at runtime, which makes
            # it better than isinstance() since this is a hot path and we have
            # to minimize the work we do prior to calling the kernel.
            self.h_init_cccl = cast(_bindings.Value, self.h_init_cccl)
            self.h_init_cccl.state = to_cccl_value_state(
                cast(np.ndarray | GpuStruct, h_init)
            )

        stream_handle = validate_and_get_stream(stream)

        if temp_storage is None:
            temp_storage_bytes = 0
            d_temp_storage = 0
        else:
            temp_storage_bytes = temp_storage.nbytes
            d_temp_storage = get_data_pointer(temp_storage)

        temp_storage_bytes = self.device_reduce_fn(
            d_temp_storage,
            temp_storage_bytes,
            self.d_in_cccl,
            self.d_out_cccl,
            num_items,
            self.op_cccl,
            self.h_init_cccl,
            stream_handle,
        )
        return temp_storage_bytes


def _input_item_dtype(d_in) -> np.dtype | None:
    """Item dtype of a reduce input, or None when it cannot be determined.

    Best-effort, for error messages only: some inputs (e.g. a struct with a
    bfloat16 field and no ml_dtypes installed) have no numpy representation.
    """
    try:
        if isinstance(d_in, IteratorBase):
            return d_in.value_type.dtype
        return get_dtype(d_in)
    except (AttributeError, TypeError, KeyError):
        return None


def _input_item_type_info(d_in) -> _bindings.TypeInfo | None:
    """CCCL type info for one input item, or None when it cannot be determined."""
    try:
        if isinstance(d_in, IteratorBase):
            return d_in.value_type.info
        return cccl._type_info_from_dtype(get_dtype(d_in))
    except (AttributeError, TypeError, KeyError):
        return None


@cache_with_registered_key_functions
def make_reduce_into(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    op: Operator,
    h_init: np.ndarray | GpuStruct | None = None,
    **kwargs,
):
    """Computes a device-wide reduction using the specified binary ``op`` and initial value ``init``.

    Example:
        Below, ``make_reduce_into`` is used to create a reduction object that can be reused.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/reduce_object.py
            :language: python
            :start-after: # example-begin


    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array (of size 1) or iterator that will store the result of the reduction
        op: Binary operator to apply.
            The signature is ``(T, T) -> T``, where ``T`` is the data type of
            the initial value ``h_init`` (or of the input items when ``h_init``
            is None).
        h_init: Numpy array or GPU struct storing initial value of the
            reduction, or None for no initial value. When ``h_init`` is a
            struct or complex dtype, the input items must share its layout;
            reduce a different item type into it by wrapping ``d_in`` in a
            ``TransformIterator`` that returns the accumulator type.
        compute_capability: Compute capability, or list of capabilities, to
            build for ahead of time. Accepts a packed int (e.g. ``90``), a
            ``(major, minor)`` pair, a string (e.g. ``"9.0"``), or a list
            thereof. When ``None`` (the default), the current device's
            architecture is used.

    Returns:
        A callable object that can be used to perform the reduction
    """
    if h_init is not None:
        try:
            accum_dtype = get_dtype(h_init)
        except (AttributeError, TypeError) as e:
            raise TypeError(
                "Could not determine accumulator dtype from h_init; "
                "expected numpy array or object with .dtype"
            ) from e

        # Validate d_in and d_out if they are device arrays (iterators may not expose
        # dtype reliably here). Additionally, only require equality of dtypes for
        # struct objects; mixed scalar dtypes (e.g. int8 input with int64 output)
        # is acceptable
        if isinstance(h_init, _Struct):
            for arr, name in ((d_in, "input"), (d_out, "output")):
                if isinstance(arr, IteratorBase):
                    continue

                dtype = get_dtype(arr)
                if dtype != accum_dtype:
                    raise TypeError(
                        f"reduce_into dtype mismatch: {name} dtype {dtype} != "
                        f"accumulator dtype {accum_dtype}. "
                        f"Ensure {name} elements and h_init have identical dtype to "
                        "avoid truncation or misinterpretation."
                    )

        # The C layer names both the input item type and the accumulator
        # `storage_t` when each is an opaque struct, so a struct accumulator
        # constrains the input's layout. Compare CCCL's type enum, not numpy's
        # dtype, so a primitive input into a struct accumulator is caught even
        # though complex dtypes are also STORAGE here. Compare alignment with
        # `<` rather than `!=` so a structured dtype built without align=True
        # (numpy reports alignment 1 for it, even though its fields are
        # aligned) is not rejected for being under-strict about a bound it
        # already satisfies.
        in_info = _input_item_type_info(d_in)
        accum_info = cccl._type_info_from_dtype(accum_dtype)
        if (
            in_info is not None
            and accum_info.typenum == _bindings.TypeEnum.STORAGE
            and (
                in_info.typenum != _bindings.TypeEnum.STORAGE
                or in_info.size != accum_info.size
                or in_info.alignment < accum_info.alignment
            )
        ):
            in_dtype = _input_item_dtype(d_in)
            in_dtype_repr = (
                in_dtype
                if in_dtype is not None
                else f"{in_info.size}-byte, {in_info.alignment}-aligned type"
            )
            if in_info.typenum != _bindings.TypeEnum.STORAGE:
                reason = (
                    "Input items are not an opaque type, so they have no "
                    "conversion to the accumulator's generated storage type."
                )
            else:
                reason = (
                    f"Input items are {in_info.size} bytes aligned to "
                    f"{in_info.alignment}, h_init is {accum_info.size} bytes "
                    f"aligned to {accum_info.alignment}; they must match."
                )
            raise TypeError(
                f"reduce_into dtype mismatch: input dtype "
                f"{in_dtype_repr} != accumulator dtype {accum_dtype}. {reason} "
                "To reduce items of a different type into this accumulator, wrap "
                "d_in in a TransformIterator that returns the accumulator type."
            )

    op_adapter = make_op_adapter(op)
    return _Reduce(
        d_in,
        d_out,
        op_adapter,
        h_init,
        kwargs.get("determinism", Determinism.RUN_TO_RUN),
        compute_capability=kwargs.get("compute_capability"),
    )


def reduce_into(
    *,
    d_in: DeviceArrayLike | IteratorT,
    d_out: DeviceArrayLike | IteratorT,
    num_items: int,
    op: Operator,
    h_init: np.ndarray | GpuStruct | None = None,
    stream=None,
    **kwargs,
):
    """
    Performs device-wide reduction.

    This function automatically handles temporary storage allocation and execution.

    Example:
        Below, ``reduce_into`` is used to compute the sum of a sequence of integers.

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/reduction/sum_reduction.py
            :language: python
            :start-after: # example-begin

    Args:
        d_in: Device array or iterator containing the input sequence of data items
        d_out: Device array or iterator to store the result of the reduction
        num_items: Number of items to reduce
        op: Binary operator to apply.
            The signature is ``(T, T) -> T``, where ``T`` is the data type of
            the initial value ``h_init`` (or of the input items when ``h_init``
            is None).
        h_init: Initial value for the reduction, or None for no initial value.
            With None, the result is the reduction of just the input elements
            (equivalent to having the first element of ``d_in`` be the initial
            value). If ``num_items`` is also zero, ``d_out`` is unmodified.
        stream: CUDA stream for the operation (optional)
    """
    reducer = make_reduce_into(d_in=d_in, d_out=d_out, op=op, h_init=h_init, **kwargs)
    tmp_storage_bytes = reducer(
        temp_storage=None,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=op,
        h_init=h_init,
        stream=stream,
    )
    tmp_storage = TempStorageBuffer(tmp_storage_bytes, stream)
    reducer(
        temp_storage=tmp_storage,
        d_in=d_in,
        d_out=d_out,
        num_items=num_items,
        op=op,
        h_init=h_init,
        stream=stream,
    )
