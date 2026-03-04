# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ._stf_bindings import (
    context,
    data_place,
    dep,
    exec_place,
    exec_place_grid,
    get_compile_flags,
    get_include_path,
    get_library_path,
)


@runtime_checkable
class ExecPlaceLike(Protocol):
    """Protocol for objects that can be used as execution places.

    Any object implementing this protocol can be passed to ctx.task(),
    exec_place_grid.create(), task.set_exec_place(), etc.

    Built-in exec_place and exec_place_grid satisfy this protocol.
    External packages can define custom execution places by implementing
    _as_stf_exec_place() (which should return an exec_place wrapping an
    opaque handle obtained from stf_exec_place_opaque_wrap()).
    """

    @property
    def kind(self) -> str: ...

    def _as_stf_exec_place(self) -> "exec_place": ...


@runtime_checkable
class DataPlaceLike(Protocol):
    """Protocol for objects that can be used as data places.

    Any object implementing this protocol can be passed wherever a data_place
    is expected. External packages can define custom data places by implementing
    _as_stf_data_place().
    """

    @property
    def kind(self) -> str: ...

    def _as_stf_data_place(self) -> "data_place": ...


__all__ = [
    "ExecPlaceLike",
    "DataPlaceLike",
    "get_compile_flags",
    "get_include_path",
    "get_library_path",
    "context",
    "dep",
    "exec_place",
    "exec_place_grid",
    "data_place",
]
