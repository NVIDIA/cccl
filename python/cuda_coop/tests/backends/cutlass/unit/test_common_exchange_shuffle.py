# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from enum import Enum, IntEnum
from importlib import import_module
from types import SimpleNamespace

import pytest

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


class _StringMode(str, Enum):
    BLOCKED_TO_STRIPED = "blocked_to_striped"
    DOWN = "down"


class _UnitDistance(IntEnum):
    ONE = 1


class _ReadOnlyPayload:
    items_per_thread = 1
    dtype = int

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return 0


@pytest.mark.parametrize(
    ("operation", "mode"),
    (
        ("exchange", SimpleNamespace(value="blocked_to_striped")),
        ("exchange", _StringMode.BLOCKED_TO_STRIPED),
        ("shuffle", SimpleNamespace(value="down")),
        ("shuffle", _StringMode.DOWN),
    ),
)
def test_common_python_entry_points_require_plain_string_modes(
    monkeypatch, operation, mode
):
    import importlib

    from cuda.coop._core.api import _dispatch
    from cuda.coop._core.api.thread_group import this_block

    api = importlib.import_module(f"cuda.coop._core.api.{operation}")
    group = this_block()
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            _dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.exchange"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.shuffle"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(TypeError, match="mode must be a string"):
            getattr(api, operation)(group, object(), mode=mode)


@pytest.mark.parametrize("operation", ("exchange", "shuffle"))
def test_common_frontends_accept_read_only_thread_data(monkeypatch, operation):
    import importlib

    from cuda.coop._core.api import _dispatch
    from cuda.coop._core.api.thread_group import this_block

    api = importlib.import_module(f"cuda.coop._core.api.{operation}")
    monkeypatch.setattr(
        api, "_group_primitive_marker", lambda *_args, **_kwargs: "validated"
    )
    kwargs = {
        "mode": "blocked_to_striped" if operation == "exchange" else "down"
    }
    group = this_block()
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            _dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.exchange"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.shuffle"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        assert (
            getattr(api, operation)(group, _ReadOnlyPayload(), **kwargs)
            == "validated"
        )


@pytest.mark.parametrize(
    "distance",
    (SimpleNamespace(value=1), _UnitDistance.ONE),
    ids=("value-impostor", "integer-enum"),
)
def test_common_shuffle_validates_the_actual_distance(monkeypatch, distance):
    from cuda.coop._core.api import _dispatch
    from cuda.coop._core.api.shuffle import shuffle
    from cuda.coop._core.api.thread_group import this_block

    group = this_block()
    with monkeypatch.context() as compiler_context:
        compiler_context.setattr(
            _dispatch, "_backend_module_name", lambda: "test.backend"
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.exchange"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        compiler_context.setattr(
            import_module("cuda.coop._core.api.shuffle"),
            "_backend_module_name",
            lambda: "test.backend",
        )
        with pytest.raises(ValueError, match="distance must be exactly 1"):
            shuffle(group, _ReadOnlyPayload(), distance=distance)
