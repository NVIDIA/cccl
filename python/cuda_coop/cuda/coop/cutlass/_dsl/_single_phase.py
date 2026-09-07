# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Shared single-phase dispatch helpers for CuTe cooperative primitives."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from ._launch import LAUNCH_METADATA_KEYS
from ._temp_storage import TempStorageBase
from ._thread_data import ThreadData, _coerce_thread_payload


@dataclass(frozen=True)
class SinglePhaseContext:
    thread_data: ThreadData | None
    temp_storage: TempStorageBase | None


_ACTIVE_SINGLE_PHASE_CONTEXT: ContextVar[SinglePhaseContext | None] = ContextVar(
    "cuda_coop_cutlass_active_single_phase_context",
    default=None,
)

THREAD_PAYLOAD_ARG_NAMES = (
    "value",
    "key",
    "keys",
    "values",
    "samples",
    "run_values",
    "run_lengths",
    "ranks",
    "valid_flags",
)

THREAD_DATA_MAPPING_ARG_NAMES = (
    "value",
    "key",
    "keys",
    "values",
)


@contextmanager
def activate_single_phase_context(context: SinglePhaseContext):
    token = _ACTIVE_SINGLE_PHASE_CONTEXT.set(context)
    try:
        yield
    finally:
        _ACTIVE_SINGLE_PHASE_CONTEXT.reset(token)


def get_active_single_phase_context() -> SinglePhaseContext | None:
    return _ACTIVE_SINGLE_PHASE_CONTEXT.get()


def extract_single_phase_context(
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    scope: str,
    temp_storage_type: type[TempStorageBase],
    reserve_context_fields: bool = False,
) -> SinglePhaseContext:
    has_thread_data = "thread_data" in kwargs
    has_temp_storage = "temp_storage" in kwargs
    thread_data_candidate = kwargs.get("thread_data", None)
    temp_storage_candidate = kwargs.get("temp_storage", None)

    if isinstance(thread_data_candidate, ThreadData):
        thread_data = kwargs.pop("thread_data")
    elif reserve_context_fields and has_thread_data and thread_data_candidate is None:
        kwargs.pop("thread_data")
        thread_data = None
    elif reserve_context_fields and thread_data_candidate is not None:
        raise TypeError(
            f"{scope}.{primitive_name} expected thread_data to be ThreadData or None"
        )
    else:
        thread_data = None

    if isinstance(temp_storage_candidate, temp_storage_type):
        temp_storage = kwargs.pop("temp_storage")
    elif reserve_context_fields and has_temp_storage and temp_storage_candidate is None:
        kwargs.pop("temp_storage")
        temp_storage = None
    elif reserve_context_fields and temp_storage_candidate is not None:
        raise TypeError(
            f"{scope}.{primitive_name} expected temp_storage to be TempStorage or None"
        )
    else:
        temp_storage = None

    return SinglePhaseContext(
        thread_data=thread_data,
        temp_storage=temp_storage,
    )


def _require_tuple_result_with_arity(
    primitive_name: str,
    result: Any,
    *,
    scope: str,
    context: str,
    arity: int,
) -> tuple[Any, ...]:
    if not isinstance(result, tuple) or len(result) != arity:
        raise TypeError(
            f"{scope}.{primitive_name} ThreadData {context} expects a "
            f"{arity}-tuple result"
        )
    return result


def coerce_thread_payloads_to_thread_data(
    primitive_name: str,
    kwargs: dict[str, Any],
    *,
    scope: str,
) -> None:
    for arg_name in THREAD_PAYLOAD_ARG_NAMES:
        source = kwargs.get(arg_name)
        if source is None:
            continue
        kwargs[arg_name] = _coerce_thread_payload(
            source,
            scope=scope,
            primitive_name=primitive_name,
            arg_name=arg_name,
        )


coerce_register_fragments_to_thread_data = coerce_thread_payloads_to_thread_data


def dispatch_thread_data_aware(
    primitive_name: str,
    impl,
    kwargs: dict[str, Any],
    *,
    scope: str,
    strip_launch_metadata: bool = False,
) -> Any:
    coerce_register_fragments_to_thread_data(primitive_name, kwargs, scope=scope)
    if strip_launch_metadata:
        for key in LAUNCH_METADATA_KEYS:
            kwargs.pop(key, None)

    # Certain provider entrypoints consume ThreadData natively through a packed ABI.
    if bool(getattr(impl, "_supports_native_thread_data", False)):
        if any(
            isinstance(kwargs.get(arg_name), ThreadData)
            for arg_name in THREAD_PAYLOAD_ARG_NAMES
        ):
            return impl(**kwargs)

    if "keys" in kwargs and "values" in kwargs:
        keys = kwargs.get("keys")
        values = kwargs.get("values")
        any_thread_data = isinstance(keys, ThreadData) or isinstance(values, ThreadData)
        if any_thread_data:
            if not isinstance(keys, ThreadData) or not isinstance(values, ThreadData):
                raise TypeError(
                    f"{scope}.{primitive_name} requires both keys and "
                    "values to be ThreadData when one argument uses ThreadData"
                )
            if keys.items_per_thread != values.items_per_thread:
                raise ValueError(
                    f"{scope}.{primitive_name} requires matching "
                    "ThreadData.items_per_thread for keys and values"
                )

            key_values = keys._require_values(primitive_name)
            value_values = values._require_values(primitive_name)
            first_kwargs = dict(kwargs)
            first_kwargs["keys"] = key_values[0]
            first_kwargs["values"] = value_values[0]
            first_result = _require_tuple_result_with_arity(
                primitive_name,
                impl(**first_kwargs),
                scope=scope,
                context="pair mapping",
                arity=2,
            )

            out_keys = keys._new_uninitialized(dtype=keys.dtype)
            out_values = values._new_uninitialized(dtype=values.dtype)
            out_keys[0], out_values[0] = first_result
            for idx, (key, value) in enumerate(
                zip(key_values[1:], value_values[1:]), start=1
            ):
                iter_kwargs = dict(kwargs)
                iter_kwargs["keys"] = key
                iter_kwargs["values"] = value
                mapped_key, mapped_value = _require_tuple_result_with_arity(
                    primitive_name,
                    impl(**iter_kwargs),
                    scope=scope,
                    context="pair mapping",
                    arity=2,
                )
                out_keys[idx] = mapped_key
                out_values[idx] = mapped_value
            return out_keys, out_values

    for arg_name in THREAD_DATA_MAPPING_ARG_NAMES:
        source = kwargs.get(arg_name)
        if not isinstance(source, ThreadData):
            continue
        source_values = source._require_values(primitive_name)
        first_kwargs = dict(kwargs)
        first_kwargs[arg_name] = source_values[0]
        first_result = impl(**first_kwargs)

        if isinstance(first_result, tuple):
            tuple_arity = len(first_result)
            if tuple_arity <= 0:
                raise TypeError(
                    f"{scope}.{primitive_name} ThreadData unary tuple "
                    "mapping expects a non-empty tuple result"
                )
            outs = [source._new_uninitialized() for _ in range(tuple_arity)]
            for out_idx, out_value in enumerate(first_result):
                outs[out_idx][0] = out_value
            for idx, value in enumerate(source_values[1:], start=1):
                iter_kwargs = dict(kwargs)
                iter_kwargs[arg_name] = value
                mapped_values = _require_tuple_result_with_arity(
                    primitive_name,
                    impl(**iter_kwargs),
                    scope=scope,
                    context="unary tuple mapping",
                    arity=tuple_arity,
                )
                for out_idx, out_value in enumerate(mapped_values):
                    outs[out_idx][idx] = out_value
            return tuple(outs)

        out = source._new_uninitialized()
        out[0] = first_result
        for idx, value in enumerate(source_values[1:], start=1):
            iter_kwargs = dict(kwargs)
            iter_kwargs[arg_name] = value
            out[idx] = impl(**iter_kwargs)
        return out
    return impl(**kwargs)
