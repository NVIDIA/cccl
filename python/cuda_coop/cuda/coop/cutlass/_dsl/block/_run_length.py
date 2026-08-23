# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Any

from cuda.coop._core.block import normalize_positive_int

from ..._internal._thread_data import (
    _infer_fragment_items_per_thread,
    _infer_vector_items_per_thread,
    _is_memory_backed_payload,
    _is_register_fragment,
)
from ..._run_length_controls import validate_decoded_window_offset
from .._scope import BLOCK_SCOPE as _SCOPE
from .._scope import merge_block_payload as merge_payload
from .._scope import validate_no_extra_block_args as validate_no_extra_args
from .._thread_data import (
    ThreadData,
    _is_thread_payload_candidate,
)
from ._dispatch import dispatch_primitive, register_primitive_impl

_UNSET = object()


class BlockRunLengthDecode:
    """Parent object for block-wide CuTe run-length decode.

    The object provides the scoped
    ``coop._block.run_length(...).decode(...)`` spelling while preserving
    CuTe's single-phase register API. ``decode`` returns decoded
    ``ThreadData`` by default; when a ``ThreadData`` output object is passed as
    the first positional argument, the decoded values are copied into that
    object and the same object is returned.
    """

    def __init__(
        self,
        run_values: Any,
        run_lengths: Any,
        /,
        *,
        runs_per_thread: int | None,
        decoded_items_per_thread: int,
        total_decoded_size: Any = None,
        decoded_offset_dtype: Any = None,
        temp_storage: Any = None,
        call_kwargs: dict[str, Any] | None = None,
    ):
        """Configure one deferred block-wide run-length decoder.

        ``run_values`` and ``run_lengths`` describe the encoded per-thread
        payload. ``decoded_items_per_thread`` fixes the output window owned by
        each thread, while ``runs_per_thread`` validates the encoded payload
        shape when it is known. Optional size, offset-dtype, scratch, and call
        keyword controls are retained for subsequent :meth:`decode` calls.
        """
        if runs_per_thread is not None:
            _validate_runs_per_thread(
                runs_per_thread,
                run_values=run_values,
                run_lengths=run_lengths,
            )

        self._run_values = run_values
        self._run_lengths = run_lengths
        self._decoded_items_per_thread = _validate_positive_int(
            "decoded_items_per_thread",
            decoded_items_per_thread,
        )
        self._total_decoded_size = total_decoded_size
        self._decoded_offset_dtype = decoded_offset_dtype
        self._call_kwargs = dict(call_kwargs) if call_kwargs is not None else {}
        if temp_storage is not None:
            if "temp_storage" in self._call_kwargs:
                raise TypeError(f"{_SCOPE}.run_length got duplicate temp_storage")
            self._call_kwargs["temp_storage"] = temp_storage

    def decode(
        self,
        *args: Any,
        decoded_window_offset: Any = 0,
        relative_offsets: Any = _UNSET,
        **kwargs: Any,
    ) -> Any:
        """Decode the configured block-wide run-length stream.

        With no decoded-items output argument, the decoded values are returned
        as a new ``ThreadData`` object. Passing a ``ThreadData`` as the first
        positional argument uses the compatibility output-buffer spelling and
        copies the returned values into that object. ``relative_offsets`` and
        ``total_decoded_size`` are filled by the underlying CUB-backed provider
        when their corresponding ``ThreadData`` outputs are provided.
        """
        (
            decoded_items,
            decoded_window_offset,
            positional_relative_offsets,
        ) = _parse_decode_args(
            args,
            decoded_window_offset=decoded_window_offset,
        )
        if relative_offsets is not _UNSET and positional_relative_offsets is not _UNSET:
            raise TypeError(
                f"{_SCOPE}.run_length.decode got duplicate relative_offsets"
            )
        if positional_relative_offsets is not _UNSET:
            relative_offsets = positional_relative_offsets
        elif relative_offsets is _UNSET:
            relative_offsets = None
        call_kwargs = _merge_call_kwargs(
            "run_length.decode",
            self._call_kwargs,
            kwargs,
        )
        run_lengths = _coerce_run_lengths_for_decode(
            self._run_lengths,
            dtype=_infer_length_dtype_from_outputs(
                relative_offsets=relative_offsets,
                total_decoded_size=self._total_decoded_size,
                decoded_offset_dtype=self._decoded_offset_dtype,
            ),
            primitive_name="run_length.decode",
        )
        decoded = run_length_decode(
            self._run_values,
            run_lengths,
            decoded_items_per_thread=self._decoded_items_per_thread,
            decoded_window_offset=decoded_window_offset,
            relative_offsets=relative_offsets,
            total_decoded_size=self._total_decoded_size,
            **call_kwargs,
        )
        if decoded_items is None:
            return decoded
        return _copy_decoded_items(decoded_items, decoded)


def _validate_positive_int(name: str, value: Any) -> int:
    try:
        return normalize_positive_int(name, value)
    except ValueError as exc:
        raise ValueError(f"{_SCOPE}.run_length_decode: {exc}") from exc


def _validate_nonnegative_int(name: str, value: Any) -> Any:
    if name != "decoded_window_offset":
        raise ValueError(f"unsupported run-length control {name!r}")
    return validate_decoded_window_offset(value, scope=_SCOPE)


def _validate_runs_per_thread(
    value: Any,
    *,
    run_values: Any,
    run_lengths: Any,
) -> int:
    runs_per_thread = _validate_positive_int("runs_per_thread", value)
    value_items = _payload_items_per_thread(run_values)
    length_items = _payload_items_per_thread(run_lengths)
    inferred = None
    if value_items is not None or length_items is not None:
        if value_items is None or length_items is None:
            raise TypeError(
                f"{_SCOPE}.run_length requires run_values and "
                "run_lengths to both be ThreadData, both be per-thread "
                "payloads, or both be scalar values"
            )
        if value_items != length_items:
            raise ValueError(
                f"{_SCOPE}.run_length requires matching "
                "ThreadData.items_per_thread for run_values and run_lengths"
            )
        inferred = value_items
    else:
        inferred = 1
    if runs_per_thread != inferred:
        raise ValueError(
            f"{_SCOPE}.run_length runs_per_thread must match "
            f"the input items_per_thread ({runs_per_thread} != {inferred})"
        )
    return runs_per_thread


def _payload_items_per_thread(value: Any) -> int | None:
    if isinstance(value, ThreadData):
        return value.items_per_thread
    if _is_register_fragment(value):
        inferred = _infer_fragment_items_per_thread(value)
    elif _is_memory_backed_payload(value):
        raise TypeError(
            f"{_SCOPE}.run_length requires per-thread register payloads, "
            "not memory-backed tensors or arrays"
        )
    else:
        inferred = _infer_vector_items_per_thread(value)
    if inferred is None and _is_thread_payload_candidate(value):
        raise TypeError(
            f"{_SCOPE}.run_length could not infer items_per_thread "
            "from register payload shape"
        )
    return inferred


def _coerce_run_payload_arg(primitive_name: str, arg_name: str, value: Any) -> Any:
    if isinstance(value, ThreadData) or not _is_thread_payload_candidate(value):
        return value
    try:
        return ThreadData.from_payload(value)
    except Exception as exc:
        raise TypeError(
            f"{_SCOPE}.{primitive_name} could not auto-convert "
            f"'{arg_name}' payload to ThreadData: {exc}"
        ) from exc


def _thread_data_output_dtype(value: Any) -> Any:
    if isinstance(value, ThreadData):
        return value.dtype
    return None


def _infer_length_dtype_from_outputs(
    *,
    relative_offsets: Any,
    total_decoded_size: Any,
    decoded_offset_dtype: Any = None,
) -> Any:
    dtype = _thread_data_output_dtype(relative_offsets)
    total_dtype = _thread_data_output_dtype(total_decoded_size)
    if dtype is None:
        dtype = total_dtype
    elif total_dtype is not None and total_dtype != dtype:
        raise TypeError(
            f"{_SCOPE}.run_length_decode relative_offsets and total_decoded_size "
            "must have matching dtype when both outputs are ThreadData"
        )
    if decoded_offset_dtype is None:
        return dtype
    if dtype is None or dtype == decoded_offset_dtype:
        return decoded_offset_dtype
    raise TypeError(
        f"{_SCOPE}.run_length_decode decoded_offset_dtype must match "
        "relative_offsets and total_decoded_size dtype when output dtypes are set"
    )


def _coerce_run_lengths_for_decode(
    value: Any,
    *,
    dtype: Any,
    primitive_name: str,
) -> Any:
    if dtype is None or isinstance(value, ThreadData):
        return value
    if not _is_thread_payload_candidate(value):
        return value
    try:
        return ThreadData.from_payload(value, dtype=dtype)
    except Exception as exc:
        raise TypeError(
            f"{_SCOPE}.{primitive_name} could not auto-convert "
            f"'run_lengths' payload to ThreadData with dtype: {exc}"
        ) from exc


def _parse_constructor_args(
    args: tuple[Any, ...],
    *,
    runs_per_thread: int | None,
    decoded_items_per_thread: int | None,
    total_decoded_size: Any,
    decoded_offset_dtype: Any,
) -> tuple[int | None, int, Any, Any]:
    if len(args) > 4:
        raise TypeError(
            f"{_SCOPE}.run_length accepts at most four extra "
            "positional arguments: runs_per_thread, decoded_items_per_thread, "
            "total_decoded_size, decoded_offset_dtype"
        )
    if len(args) >= 1:
        if runs_per_thread is not None:
            raise TypeError(f"{_SCOPE}.run_length got duplicate runs_per_thread")
        runs_per_thread = args[0]
    if len(args) >= 2:
        if decoded_items_per_thread is not None:
            raise TypeError(
                f"{_SCOPE}.run_length got duplicate decoded_items_per_thread"
            )
        decoded_items_per_thread = args[1]
    if len(args) >= 3:
        if total_decoded_size is not None:
            raise TypeError(f"{_SCOPE}.run_length got duplicate total_decoded_size")
        total_decoded_size = args[2]
    if len(args) >= 4:
        if decoded_offset_dtype is not None:
            raise TypeError(f"{_SCOPE}.run_length got duplicate decoded_offset_dtype")
        decoded_offset_dtype = args[3]

    if decoded_items_per_thread is None:
        raise TypeError(
            f"{_SCOPE}.run_length missing required decoded_items_per_thread"
        )
    return (
        runs_per_thread,
        _validate_positive_int(
            "decoded_items_per_thread",
            decoded_items_per_thread,
        ),
        total_decoded_size,
        decoded_offset_dtype,
    )


def _parse_decode_args(
    args: tuple[Any, ...],
    *,
    decoded_window_offset: Any,
) -> tuple[ThreadData | None, Any, Any]:
    decoded_items = None
    relative_offsets = _UNSET
    if len(args) > 3:
        raise TypeError(
            f"{_SCOPE}.run_length.decode accepts at most "
            "decoded_items, decoded_window_offset, and relative_offsets positional "
            "arguments"
        )
    if len(args) == 1:
        if isinstance(args[0], ThreadData):
            decoded_items = args[0]
        else:
            decoded_window_offset = args[0]
    elif len(args) >= 2:
        decoded_items = args[0]
        decoded_window_offset = args[1]
        if not isinstance(decoded_items, ThreadData):
            raise TypeError(
                f"{_SCOPE}.run_length.decode decoded_items must be ThreadData"
            )
        if len(args) == 3:
            relative_offsets = args[2]
    return (
        decoded_items,
        _validate_nonnegative_int(
            "decoded_window_offset",
            decoded_window_offset,
        ),
        relative_offsets,
    )


def _merge_call_kwargs(
    primitive_name: str,
    base_kwargs: dict[str, Any],
    extra_kwargs: dict[str, Any],
) -> dict[str, Any]:
    overlap = base_kwargs.keys() & extra_kwargs.keys()
    if overlap:
        names = ", ".join(sorted(overlap))
        raise TypeError(
            f"{_SCOPE}.{primitive_name} got duplicate "
            f"metadata keyword argument(s): {names}"
        )
    merged = dict(base_kwargs)
    merged.update(extra_kwargs)
    return merged


def _copy_decoded_items(decoded_items: ThreadData, decoded: Any) -> ThreadData:
    if not isinstance(decoded, ThreadData):
        raise TypeError(
            f"{_SCOPE}.run_length.decode expected provider to "
            "return ThreadData when decoded_items output is provided"
        )
    if decoded_items.items_per_thread != decoded.items_per_thread:
        raise ValueError(
            f"{_SCOPE}.run_length.decode decoded_items must "
            f"have items_per_thread={decoded.items_per_thread}"
        )
    if decoded_items.dtype is not None:
        if decoded.dtype is not None and decoded_items.dtype != decoded.dtype:
            raise ValueError(
                f"{_SCOPE}.run_length.decode decoded_items "
                "dtype must match decoded output dtype"
            )
        # If the provider result lacks dtype metadata, keep the caller's explicit
        # output dtype and only copy the values.
    elif decoded_items.dtype is None:
        decoded_items.dtype = decoded.dtype
    for idx in range(decoded.items_per_thread):
        decoded_items[idx] = decoded[idx]
    return decoded_items


def _run_length_decode_provider(
    *,
    run_values: Any,
    run_lengths: Any,
    args: tuple[Any, ...] = (),
    decoded_items_per_thread: int,
    decoded_window_offset: Any = 0,
    relative_offsets: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
    **kwargs: Any,
) -> Any:
    if args:
        validate_no_extra_args(
            "run_length_decode",
            args=args,
            kwargs={},
            expected="does not accept extra positional args",
        )

    from ... import _group_run_length_decode as _group_frontend
    from ..._thread_group import this_block

    return _group_frontend._run_length_decode(
        this_block(),
        run_values,
        run_lengths,
        decoded_items_per_thread=_validate_positive_int(
            "decoded_items_per_thread",
            decoded_items_per_thread,
        ),
        decoded_window_offset=_validate_nonnegative_int(
            "decoded_window_offset",
            decoded_window_offset,
        ),
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
        source="scoped_block",
        **kwargs,
    )


_run_length_decode_provider._supports_native_thread_data = True
_run_length_decode_provider._preserves_launch_metadata = True
_run_length_decode_provider._uses_planned_temp_storage = True


def run_length(
    run_values: Any,
    run_lengths: Any,
    /,
    *args: Any,
    runs_per_thread: int | None = None,
    decoded_items_per_thread: int | None = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
    temp_storage: Any = None,
    **kwargs: Any,
) -> BlockRunLengthDecode:
    """Create a two-phase parent object for block run-length decode.

    ``run_values`` and ``run_lengths`` should be scalar per-thread runs or
    matching ``ThreadData`` payloads. ``runs_per_thread`` is optional for CuTe
    and, when provided, must match the inferred ``ThreadData.items_per_thread``.
    Actual run lengths must be positive and may be followed by one suffix of
    zero-length padding entries; their block-wide sum must be positive and
    representable in the run-length dtype.
    Call ``decode()`` on the returned object to produce decoded register items
    through the same CUB-backed LTO-IR provider used by ``run_length_decode``.
    """
    run_values = _coerce_run_payload_arg("run_length", "run_values", run_values)
    (
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype,
    ) = _parse_constructor_args(
        args,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
    )
    return BlockRunLengthDecode(
        run_values,
        run_lengths,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        total_decoded_size=total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
        temp_storage=temp_storage,
        call_kwargs=kwargs,
    )


def run_length_decode(
    run_values: Any,
    run_lengths: Any,
    /,
    *args: Any,
    decoded_items_per_thread: int,
    decoded_window_offset: Any = 0,
    relative_offsets: Any = None,
    total_decoded_size: Any = None,
    decoded_offset_dtype: Any = None,
    **kwargs: Any,
) -> Any:
    """Decode a block-wide run-length window into per-thread output items.

    Out-of-range decoded targets return a zero decoded value and an all-ones
    relative-offset sentinel (``-1`` for signed run-length dtypes). Actual run
    lengths must be positive and may be followed by one suffix of zero-length
    padding entries; their block-wide sum must be positive and representable in
    the run-length dtype. Dynamic window offsets must be uniform, nonnegative,
    and representable in that dtype.
    """
    structural_payload = {
        "run_values": run_values,
        "run_lengths": _coerce_run_lengths_for_decode(
            run_lengths,
            dtype=_infer_length_dtype_from_outputs(
                relative_offsets=relative_offsets,
                total_decoded_size=total_decoded_size,
                decoded_offset_dtype=decoded_offset_dtype,
            ),
            primitive_name="run_length_decode",
        ),
        "args": args,
        "decoded_items_per_thread": _validate_positive_int(
            "decoded_items_per_thread",
            decoded_items_per_thread,
        ),
        "decoded_window_offset": _validate_nonnegative_int(
            "decoded_window_offset",
            decoded_window_offset,
        ),
    }
    if relative_offsets is not None:
        structural_payload["relative_offsets"] = relative_offsets
    if total_decoded_size is not None:
        structural_payload["total_decoded_size"] = total_decoded_size
    if decoded_offset_dtype is not None:
        structural_payload["decoded_offset_dtype"] = decoded_offset_dtype
    payload = merge_payload(
        "run_length_decode",
        structural_payload,
        kwargs,
    )
    return dispatch_primitive("run_length_decode", kwargs=payload)


register_primitive_impl("run_length_decode", impl=_run_length_decode_provider)
