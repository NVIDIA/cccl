# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import copy
import functools
from types import ModuleType
from typing import Any

from ._runtime_dependency import raise_for_missing_cutlass_runtime


def _scope_replacements(*, source: str, target: str) -> tuple[tuple[str, str], ...]:
    replacements = [(source, target)]
    source_root = source.rsplit(".", 1)[0]
    target_root = target.rsplit(".", 1)[0]
    if source_root != target_root:
        replacements.extend(
            (
                (f"{source_root}.block", f"{target_root}.block"),
                (f"{source_root}.warp", f"{target_root}.warp"),
                (f"{source_root}.ThreadData", f"{target_root}.ThreadData"),
            )
        )
    return tuple(replacements)


def _rewrite_scope_text(text: str, *, source: str, target: str) -> str:
    replacement = text
    for old, new in _scope_replacements(source=source, target=target):
        replacement = replacement.replace(old, new)
    return replacement


def _rewrite_exception_scope(exc: BaseException, *, source: str, target: str) -> bool:
    rewritten = False
    args = []
    for arg in exc.args:
        if isinstance(arg, str):
            replacement = _rewrite_scope_text(arg, source=source, target=target)
            rewritten = rewritten or replacement != arg
            args.append(replacement)
        else:
            args.append(arg)
    if rewritten:
        exc.args = tuple(args)
    return rewritten


def _wrap_function(function: Any, *, source: str, target: str, module_name: str) -> Any:
    @functools.wraps(function)
    def _scoped_call(*args: Any, **kwargs: Any) -> Any:
        try:
            result = function(*args, **kwargs)
        except ImportError as exc:
            raise_for_missing_cutlass_runtime(exc)
            _rewrite_exception_scope(exc, source=source, target=target)
            raise
        except Exception as exc:
            _rewrite_exception_scope(exc, source=source, target=target)
            raise
        return _rewrite_return_scope(
            result, source=source, target=target, module_name=module_name
        )

    _scoped_call.__module__ = module_name
    if isinstance(function.__doc__, str):
        _scoped_call.__doc__ = _rewrite_scope_text(
            function.__doc__, source=source, target=target
        )
    return _scoped_call


def _is_primitive_factory(value: Any) -> bool:
    try:
        from ._dsl._factory import _PrimitiveFactory
    except Exception:
        return False
    return isinstance(value, _PrimitiveFactory)


def _rewrite_return_scope(
    value: Any,
    *,
    source: str,
    target: str,
    module_name: str,
) -> Any:
    if not _is_primitive_factory(value):
        return value

    value = copy.copy(value)
    scope = getattr(value, "scope", None)
    if isinstance(scope, str):
        value.scope = _rewrite_scope_text(scope, source=source, target=target)

    primitive = getattr(value, "primitive", None)
    if callable(primitive):
        value.primitive = _wrap_function(
            primitive,
            source=source,
            target=target,
            module_name=module_name,
        )
    return value


def _adapt_public_value(
    name: str,
    value: Any,
    *,
    source: str,
    target: str,
    module_name: str,
) -> Any:
    if callable(value) and not isinstance(value, type):
        return _wrap_function(
            value, source=source, target=target, module_name=module_name
        )
    # Keep selector enums and helper classes identity-shared with the backend.
    # Backend code may compare enum members directly, so these are aliases rather
    # than private-scope-specific wrapper types.
    return value


def _is_public_export(name: str, *, backend: ModuleType) -> bool:
    if name.startswith("_"):
        return False
    return name in backend.__all__


def _public_backend_names(backend: ModuleType) -> set[str]:
    return {name for name in dir(backend) if _is_public_export(name, backend=backend)}


def install_public_exports(
    namespace: dict[str, Any],
    *,
    backend: ModuleType,
    source: str,
    target: str,
) -> None:
    module_name = namespace["__name__"]
    for name in backend.__all__:
        namespace[name] = _adapt_public_value(
            name,
            getattr(backend, name),
            source=source,
            target=target,
            module_name=module_name,
        )


def get_public_attr(
    name: str,
    *,
    backend: ModuleType,
    source: str,
    target: str,
    module_name: str,
    namespace: dict[str, Any] | None = None,
) -> Any:
    if not _is_public_export(name, backend=backend):
        raise AttributeError(f"module {module_name!r} has no attribute {name!r}")

    try:
        backend_value = getattr(backend, name)
    except AttributeError as exc:
        raise AttributeError(
            f"module {module_name!r} has no attribute {name!r}"
        ) from exc

    value = _adapt_public_value(
        name,
        backend_value,
        source=source,
        target=target,
        module_name=module_name,
    )
    if namespace is not None:
        namespace[name] = value
    return value


def public_dir(namespace: dict[str, Any], *, backend: ModuleType) -> list[str]:
    public_backend_names = _public_backend_names(backend)
    return sorted({*namespace, *public_backend_names})
