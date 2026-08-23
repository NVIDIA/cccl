# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deferred-factory helpers for CuTe cooperative scopes."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import Any, Callable

from .._payload import reject_payload_selector_keyword
from ._thread_data import _validate_items_per_thread


class _DefaultSelector:
    __slots__ = ()

    def __repr__(self) -> str:
        return "DEFAULT"


_DEFAULT_SELECTOR = _DefaultSelector()


class _PrimitiveFactory:
    __slots__ = (
        "bound_kwargs",
        "factory_name",
        "launch_metadata_keys",
        "override_aliases",
        "overridable_kwargs",
        "primitive",
        "scope",
    )

    def __init__(
        self,
        *,
        scope: str,
        factory_name: str,
        primitive: Callable[..., Any],
        bound_kwargs: tuple[tuple[str, Any], ...],
        overridable_kwargs: tuple[str, ...] = (),
        override_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (),
        launch_metadata_keys: tuple[str, ...] = (),
    ):
        self.scope = scope
        self.factory_name = factory_name
        self.primitive = primitive
        self.bound_kwargs = bound_kwargs
        self.overridable_kwargs = overridable_kwargs
        self.override_aliases = override_aliases
        self.launch_metadata_keys = launch_metadata_keys

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        merged = dict(self.bound_kwargs)
        for call_name, bound_names in self.override_aliases:
            if call_name in kwargs:
                for bound_name in bound_names:
                    merged.pop(bound_name, None)
        overridable_kwargs = set(self.overridable_kwargs)
        for name in merged.keys() & kwargs.keys() & overridable_kwargs:
            del merged[name]
        duplicates = merged.keys() & kwargs.keys()
        if duplicates:
            names = ", ".join(sorted(duplicates))
            raise TypeError(
                f"{self.scope}.{self.factory_name} call got "
                f"duplicate keyword argument(s): {names}"
            )
        launch_metadata_keys = set(self.launch_metadata_keys)
        bound_launch_aliases = set(merged) & launch_metadata_keys
        call_launch_aliases = set(kwargs) & launch_metadata_keys
        if bound_launch_aliases and call_launch_aliases:
            names = ", ".join(sorted(bound_launch_aliases | call_launch_aliases))
            raise TypeError(
                f"{self.scope}.{self.factory_name} call got "
                f"duplicate launch metadata aliases: {names}"
            )
        merged.update(kwargs)
        return self.primitive(*args, **merged)


def _make_factory(
    scope: str,
    factory_name: str,
    primitive: Callable[..., Any],
    kwargs: Mapping[str, Any],
    *,
    overridable_kwargs: tuple[str, ...] = (),
    override_aliases: tuple[tuple[str, tuple[str, ...]], ...] = (),
    launch_metadata_keys: tuple[str, ...] = (),
) -> _PrimitiveFactory:
    return _PrimitiveFactory(
        scope=scope,
        factory_name=factory_name,
        primitive=_public_factory_primitive(primitive, module_name=scope),
        bound_kwargs=tuple(kwargs.items()),
        overridable_kwargs=overridable_kwargs,
        override_aliases=override_aliases,
        launch_metadata_keys=launch_metadata_keys,
    )


def _public_factory_primitive(
    primitive: Callable[..., Any],
    *,
    module_name: str,
) -> Callable[..., Any]:
    if getattr(primitive, "__module__", None) == module_name:
        return primitive

    @functools.wraps(primitive)
    def _factory_primitive(*args: Any, **kwargs: Any) -> Any:
        return primitive(*args, **kwargs)

    _factory_primitive.__module__ = module_name
    return _factory_primitive


def _bind_if_not_none(kwargs: dict[str, Any], name: str, value: Any) -> None:
    if value is not None:
        kwargs[name] = value


def _reject_if_supplied(scope: str, factory_name: str, name: str, value: Any) -> None:
    if value is not None:
        raise NotImplementedError(
            f"{scope}.{factory_name} does not support "
            f"compatibility factory argument {name!r}"
        )


def _reject_methods(scope: str, factory_name: str, kwargs: dict[str, Any]) -> None:
    methods = kwargs.pop("methods", None)
    _reject_if_supplied(scope, factory_name, "methods", methods)
    reject_payload_selector_keyword(
        kwargs,
        scope=scope,
        primitive_name=factory_name,
    )


def _reject_prims_specific_load_store_factory_kwargs(
    scope: str,
    factory_name: str,
    kwargs: dict[str, Any],
) -> None:
    names = sorted(
        {
            "alignment",
            "bounds_check",
            "ip",
            "is_invariant",
            "is_invariant_group",
            "is_nontemporal",
            "is_volatile",
            "loc",
            "ordering",
            "syncscope",
        }
        & kwargs.keys()
    )
    if names:
        raise TypeError(
            f"{scope}.{factory_name} got Prims-specific factory keyword "
            f"argument(s): {', '.join(names)}"
        )


def _reject_algorithm(
    scope: str,
    factory_name: str,
    algorithm: Any,
    *,
    default: Any,
) -> None:
    if algorithm is not None and algorithm != default:
        raise NotImplementedError(
            f"{scope}.{factory_name} does not support CUB algorithm selection yet"
        )


def _resolve_static_items_per_thread(
    scope: str,
    factory_name: str,
    items_per_thread: Any,
) -> int:
    try:
        return _validate_items_per_thread(items_per_thread)
    except TypeError as exc:
        raise TypeError(
            f"{scope}.{factory_name} items_per_thread must be an int"
        ) from exc
    except ValueError as exc:
        raise ValueError(
            f"{scope}.{factory_name} items_per_thread must be positive"
        ) from exc


def _select_positional_enum_alias(
    scope: str,
    factory_name: str,
    positional_value: Any,
    keyword_value: Any,
    *,
    enum_type: type,
    keyword_name: str,
) -> tuple[Any, Any]:
    if not isinstance(positional_value, enum_type):
        return positional_value, keyword_value
    if (
        keyword_value is not _DEFAULT_SELECTOR
        and keyword_value is not None
        and keyword_value != positional_value
    ):
        raise TypeError(
            f"{scope}.{factory_name} got conflicting "
            f"positional and keyword {keyword_name}"
        )
    return None, positional_value


def _normalize_pair_dtype_aliases(
    scope: str,
    factory_name: str,
    keys: Any,
    values: Any,
    key_dtype: Any,
    value_dtype: Any,
) -> tuple[Any, Any]:
    if keys is not None and key_dtype is not None and keys != key_dtype:
        raise TypeError(f"{scope}.{factory_name} got conflicting keys and key_dtype")
    if values is not None and value_dtype is not None and values != value_dtype:
        raise TypeError(
            f"{scope}.{factory_name} got conflicting values and value_dtype"
        )
    return (
        keys if keys is not None else key_dtype,
        values if values is not None else value_dtype,
    )
