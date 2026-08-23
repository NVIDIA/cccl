# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS load/store dispatch shared by block and warp scopes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ._payload import Payload, normalize_payload_selector
from ._prims import is_cutlass_array_operand

_PRIMS_FACTORY_OVERRIDABLE_KWARGS = {
    "make_load": ("offset",),
    "make_store": ("offset",),
}
_PRIMS_LOAD_STORE_CONTROL_KWARGS = frozenset(
    (
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
    )
)
_LOAD_VALID_OVERRIDABLE_KWARGS = (
    "offset",
    "valid_items",
    "num_valid_items",
    "oob_default",
)
_STORE_VALID_OVERRIDABLE_KWARGS = (
    "offset",
    "valid_items",
    "num_valid_items",
)
_LOAD_STORE_VALID_OVERRIDE_ALIASES = (
    ("valid_items", ("num_valid_items",)),
    ("num_valid_items", ("valid_items",)),
)


def _normalize_payload_selector(
    payload: Any,
    *,
    scope: str,
    primitive_name: str,
) -> Payload | None:
    return normalize_payload_selector(
        payload,
        scope=scope,
        primitive_name=primitive_name,
        allowed=(Payload.PRIMS,),
        choices_text="prims",
    )


def dispatch_load(
    source: Any,
    /,
    *args: Any,
    scope: str,
    tensor_load: Callable[..., Any],
    payload: Any = None,
    **kwargs: Any,
) -> Any:
    payload = _normalize_payload_selector(
        payload,
        scope=scope,
        primitive_name="load",
    )
    prims_signals = _supplied_prims_load_store_controls(kwargs)
    if payload is Payload.PRIMS or (
        payload is None and (is_cutlass_array_operand(source) or prims_signals)
    ):
        from . import _prims_adapter

        return _prims_adapter.load(source, *args, scope=scope, **kwargs)
    return tensor_load(source, *args, **kwargs)


def dispatch_store(
    destination: Any,
    value: Any,
    /,
    *args: Any,
    scope: str,
    tensor_store: Callable[..., Any],
    payload: Any = None,
    **kwargs: Any,
) -> Any:
    payload = _normalize_payload_selector(
        payload,
        scope=scope,
        primitive_name="store",
    )
    prims_signals = _supplied_prims_load_store_controls(kwargs)
    if payload is Payload.PRIMS or (
        payload is None and (is_cutlass_array_operand(destination) or prims_signals)
    ):
        from . import _prims_adapter

        return _prims_adapter.store(
            destination,
            value,
            *args,
            scope=scope,
            **kwargs,
        )
    return tensor_store(destination, value, *args, **kwargs)


def root_factory(
    make_factory: Callable[..., Any],
    primitive: Callable[..., Any],
    /,
    *args: Any,
    **kwargs: Any,
) -> Any:
    payload = kwargs.pop("payload", None)
    factory_scope = getattr(make_factory, "__module__", None) or getattr(
        primitive,
        "__module__",
        "cuda.coop.cutlass",
    )
    factory_name = getattr(make_factory, "__name__", None) or getattr(
        primitive,
        "__name__",
        "factory",
    )
    if payload is not None:
        payload = _normalize_payload_selector(
            payload,
            scope=factory_scope,
            primitive_name=factory_name,
        )
    factory = make_factory(*args, **kwargs)
    factory.primitive = primitive
    bound_kwargs = getattr(factory, "bound_kwargs", ())
    if payload is not None:
        bound_kwargs = (
            ("payload", payload),
            *((name, value) for name, value in bound_kwargs if name != "payload"),
        )
        factory.bound_kwargs = bound_kwargs
    overridable_kwargs = getattr(factory, "overridable_kwargs", ())
    prims_overrides = _PRIMS_FACTORY_OVERRIDABLE_KWARGS.get(factory_name, ())
    if prims_overrides:
        bound_names = {name for name, _ in bound_kwargs}
        factory.overridable_kwargs = (
            *overridable_kwargs,
            *(
                name
                for name in prims_overrides
                if name in bound_names and name not in overridable_kwargs
            ),
        )
        overridable_kwargs = factory.overridable_kwargs
    if (
        any(name == "payload" for name, _ in bound_kwargs)
        and "payload" not in overridable_kwargs
    ):
        factory.overridable_kwargs = (*overridable_kwargs, "payload")
    return factory


def make_block_load_factory(
    make_factory: Callable[..., Any],
    primitive: Callable[..., Any],
    /,
    *args: Any,
    scope: str,
    **kwargs: Any,
) -> Any:
    return _load_store_factory(
        make_factory,
        primitive,
        *args,
        scope=scope,
        factory_name="make_load",
        group_scope="block",
        positional_defaults=(
            ("dtype", None),
            ("threads_per_block", None),
            ("items_per_thread", 1),
            ("algorithm", "direct"),
        ),
        keyword_defaults=(
            ("dim", None),
            ("offset", None),
            ("valid_items", None),
            ("num_valid_items", None),
            ("oob_default", None),
            ("bounds_check", None),
        ),
        explicitly_bound_prims_kwargs=("offset", "bounds_check"),
        overridable_kwargs=_LOAD_VALID_OVERRIDABLE_KWARGS,
        **kwargs,
    )


def make_block_store_factory(
    make_factory: Callable[..., Any],
    primitive: Callable[..., Any],
    /,
    *args: Any,
    scope: str,
    **kwargs: Any,
) -> Any:
    return _load_store_factory(
        make_factory,
        primitive,
        *args,
        scope=scope,
        factory_name="make_store",
        group_scope="block",
        positional_defaults=(
            ("dtype", None),
            ("threads_per_block", None),
            ("items_per_thread", 1),
            ("algorithm", "direct"),
        ),
        keyword_defaults=(
            ("dim", None),
            ("offset", None),
            ("valid_items", None),
            ("num_valid_items", None),
            ("bounds_check", None),
        ),
        explicitly_bound_prims_kwargs=("offset", "bounds_check"),
        overridable_kwargs=_STORE_VALID_OVERRIDABLE_KWARGS,
        **kwargs,
    )


def make_warp_load_factory(
    make_factory: Callable[..., Any],
    primitive: Callable[..., Any],
    /,
    *args: Any,
    scope: str,
    **kwargs: Any,
) -> Any:
    return _load_store_factory(
        make_factory,
        primitive,
        *args,
        scope=scope,
        factory_name="make_load",
        group_scope="warp",
        positional_defaults=(
            ("dtype", None),
            ("items_per_thread", 1),
            ("threads_in_warp", 32),
            ("algorithm", "direct"),
            ("num_valid_items", None),
            ("oob_default", None),
        ),
        keyword_defaults=(
            ("offset", None),
            ("valid_items", None),
            ("bounds_check", None),
        ),
        explicitly_bound_prims_kwargs=("offset", "bounds_check"),
        overridable_kwargs=_LOAD_VALID_OVERRIDABLE_KWARGS,
        **kwargs,
    )


def make_warp_store_factory(
    make_factory: Callable[..., Any],
    primitive: Callable[..., Any],
    /,
    *args: Any,
    scope: str,
    **kwargs: Any,
) -> Any:
    return _load_store_factory(
        make_factory,
        primitive,
        *args,
        scope=scope,
        factory_name="make_store",
        group_scope="warp",
        positional_defaults=(
            ("dtype", None),
            ("items_per_thread", 1),
            ("threads_in_warp", 32),
            ("algorithm", "direct"),
            ("num_valid_items", None),
        ),
        keyword_defaults=(
            ("offset", None),
            ("valid_items", None),
            ("bounds_check", None),
        ),
        explicitly_bound_prims_kwargs=("offset", "bounds_check"),
        overridable_kwargs=_STORE_VALID_OVERRIDABLE_KWARGS,
        **kwargs,
    )


def _load_store_factory(
    make_factory: Callable[..., Any],
    primitive: Callable[..., Any],
    /,
    *args: Any,
    scope: str,
    factory_name: str,
    group_scope: str,
    positional_defaults: tuple[tuple[str, Any], ...],
    keyword_defaults: tuple[tuple[str, Any], ...],
    explicitly_bound_prims_kwargs: tuple[str, ...],
    overridable_kwargs: tuple[str, ...],
    **kwargs: Any,
) -> Any:
    payload = kwargs.get("payload", None)
    normalized_payload = None
    if payload is not None:
        normalized_payload = _normalize_payload_selector(
            payload,
            scope=scope,
            primitive_name=factory_name,
        )
    prims_specific_kwargs = _supplied_prims_load_store_controls(kwargs)
    if normalized_payload is Payload.PRIMS or prims_specific_kwargs:
        payload_default = normalized_payload
        if payload_default is None and prims_specific_kwargs:
            payload_default = Payload.PRIMS
        kwargs = dict(kwargs)
        kwargs.pop("payload", None)
        return _make_root_prims_load_store_factory(
            primitive,
            *args,
            scope=scope,
            factory_name=factory_name,
            group_scope=group_scope,
            payload=payload_default,
            positional_defaults=positional_defaults,
            keyword_defaults=keyword_defaults,
            explicitly_bound_prims_kwargs=explicitly_bound_prims_kwargs,
            overridable_kwargs=overridable_kwargs,
            kwargs=kwargs,
        )
    return root_factory(make_factory, primitive, *args, **kwargs)


def _supplied_prims_load_store_controls(
    kwargs: dict[str, Any],
) -> tuple[str, ...]:
    return tuple(sorted(_PRIMS_LOAD_STORE_CONTROL_KWARGS & kwargs.keys()))


def _make_root_prims_load_store_factory(
    primitive: Callable[..., Any],
    /,
    *args: Any,
    scope: str,
    factory_name: str,
    group_scope: str,
    payload: Payload | None,
    positional_defaults: tuple[tuple[str, Any], ...],
    keyword_defaults: tuple[tuple[str, Any], ...],
    explicitly_bound_prims_kwargs: tuple[str, ...],
    overridable_kwargs: tuple[str, ...],
    kwargs: dict[str, Any],
) -> Any:
    from ._dsl._factory import (
        _bind_if_not_none,
        _make_factory,
        _reject_methods,
        _resolve_static_items_per_thread,
    )
    from ._dsl._launch import (
        LAUNCH_METADATA_KEYS,
        _reject_launch_metadata_kwargs,
        resolve_block_threads,
        resolve_threads_in_warp,
    )

    values, explicit_names, extra_kwargs = _bind_factory_arguments(
        scope,
        factory_name,
        args=args,
        kwargs=kwargs,
        positional_defaults=positional_defaults,
        keyword_defaults=keyword_defaults,
    )
    _reject_methods(scope, factory_name, extra_kwargs)
    if group_scope == "warp":
        _reject_launch_metadata_kwargs(scope, factory_name, extra_kwargs)
    bound = dict(extra_kwargs)

    if group_scope == "block":
        resolved_threads = resolve_block_threads(
            scope,
            factory_name,
            threads_per_block=values["threads_per_block"],
            dim=values["dim"],
        )
        if resolved_threads is not None:
            bound["threads_per_block"] = resolved_threads
        launch_metadata_keys = LAUNCH_METADATA_KEYS
    elif group_scope == "warp":
        bound["threads_in_warp"] = resolve_threads_in_warp(
            scope,
            factory_name,
            values["threads_in_warp"],
        )
        launch_metadata_keys = ()
    else:
        raise AssertionError(f"unhandled load/store group scope {group_scope!r}")

    valid_items = _normalize_valid_items_aliases(
        scope,
        factory_name,
        valid_items=values["valid_items"],
        num_valid_items=values["num_valid_items"],
    )
    bound.update(
        items_per_thread=_resolve_static_items_per_thread(
            scope,
            factory_name,
            values["items_per_thread"],
        ),
        algorithm=values["algorithm"],
    )
    _bind_if_not_none(bound, "dtype", values["dtype"])
    _bind_if_not_none(bound, "valid_items", valid_items)
    if "oob_default" in values:
        _bind_if_not_none(bound, "oob_default", values["oob_default"])
    for name in explicitly_bound_prims_kwargs:
        if name in explicit_names:
            bound[name] = values[name]

    factory = _make_factory(
        scope,
        factory_name=factory_name,
        primitive=primitive,
        kwargs=bound,
        overridable_kwargs=overridable_kwargs,
        override_aliases=_LOAD_STORE_VALID_OVERRIDE_ALIASES,
        launch_metadata_keys=launch_metadata_keys,
    )
    if payload is not None:
        _bind_factory_payload_default(factory, payload)
    return factory


def _bind_factory_arguments(
    scope: str,
    factory_name: str,
    *,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    positional_defaults: tuple[tuple[str, Any], ...],
    keyword_defaults: tuple[tuple[str, Any], ...],
) -> tuple[dict[str, Any], set[str], dict[str, Any]]:
    if len(args) > len(positional_defaults):
        raise TypeError(
            f"{scope}.{factory_name} accepts at most "
            f"{len(positional_defaults)} positional argument(s)"
        )
    remaining = dict(kwargs)
    values: dict[str, Any] = {}
    explicit_names: set[str] = set()
    for (name, _), value in zip(positional_defaults, args):
        if name in remaining:
            raise TypeError(
                f"{scope}.{factory_name} got multiple values for argument {name!r}"
            )
        values[name] = value
        explicit_names.add(name)
    for name, default in positional_defaults[len(args) :]:
        if name in remaining:
            values[name] = remaining.pop(name)
            explicit_names.add(name)
        else:
            values[name] = default
    for name, default in keyword_defaults:
        if name in remaining:
            values[name] = remaining.pop(name)
            explicit_names.add(name)
        else:
            values[name] = default
    return values, explicit_names, remaining


def _normalize_valid_items_aliases(
    scope: str,
    factory_name: str,
    *,
    valid_items: Any,
    num_valid_items: Any,
) -> Any:
    if valid_items is not None and num_valid_items is not None:
        raise TypeError(
            f"{scope}.{factory_name} got both valid_items and num_valid_items"
        )
    return valid_items if valid_items is not None else num_valid_items


def _bind_factory_payload_default(factory: Any, payload: Payload) -> None:
    bound_kwargs = getattr(factory, "bound_kwargs", ())
    factory.bound_kwargs = (
        ("payload", payload),
        *((name, value) for name, value in bound_kwargs if name != "payload"),
    )
    overridable_kwargs = getattr(factory, "overridable_kwargs", ())
    if "payload" not in overridable_kwargs:
        factory.overridable_kwargs = (*overridable_kwargs, "payload")
