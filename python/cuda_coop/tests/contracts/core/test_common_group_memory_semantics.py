# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


@pytest.mark.evidence_for("group.load", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.store", backend="core", evidence="semantics")
def test_common_load_store_preserve_portable_results_and_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = (1, 2, 3)
    output: list[int] = []
    destination: list[int] = []
    value = (4, 5)
    observed: list[tuple[object, ...]] = []

    def load(group, actual_source, actual_output, **kwargs):
        observed.append(("load", group, actual_source, actual_output, kwargs))
        actual_output.extend(actual_source[:2])
        return actual_output

    def store(group, actual_destination, actual_value, **kwargs):
        observed.append(("store", group, actual_destination, actual_value, kwargs))
        actual_destination.extend(actual_value)
        return object()

    backend = SimpleNamespace(load=load, store=store)

    def import_backend(name: str):
        assert name == "cuda.coop.testing"
        return backend

    monkeypatch.setattr(root_api, "import_module", import_backend)
    group = root_api.this_block()

    with root_api._compiler_scope("cuda.coop.testing"):
        loaded = root_api.load(
            group,
            source,
            output,
            algorithm="direct",
            valid_items=2,
            oob_default=-1,
            offset=1,
            temp_storage="load scratch",
        )
        stored = root_api.store(
            group,
            destination,
            value,
            algorithm="direct",
            valid_items=2,
            offset=3,
            temp_storage="store scratch",
        )

    assert loaded is output
    assert output == [1, 2]
    assert stored is None
    assert destination == [4, 5]
    assert source == (1, 2, 3)
    assert value == (4, 5)
    assert observed == [
        (
            "load",
            group,
            source,
            output,
            {
                "algorithm": "direct",
                "valid_items": 2,
                "oob_default": -1,
                "offset": 1,
                "temp_storage": "load scratch",
            },
        ),
        (
            "store",
            group,
            destination,
            value,
            {
                "algorithm": "direct",
                "valid_items": 2,
                "offset": 3,
                "temp_storage": "store scratch",
            },
        ),
    ]


def test_common_load_store_support_block_physical_and_logical_warps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str]] = []

    def load(group, source, output, **_kwargs):
        observed.append(("load", group.kind))
        output.extend(source)
        return output

    def store(group, destination, value, **_kwargs):
        observed.append(("store", group.kind))
        destination.extend(value)

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(load=load, store=store),
    )
    warp = root_api.this_warp()
    supported_groups = (
        root_api.this_block(),
        warp,
        warp.group_by(8),
    )
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_cluster(),
        root_api.this_grid(),
    )

    with root_api._compiler_scope("cuda.coop.testing"):
        for group in supported_groups:
            assert root_api.load(group, (1,), []) == [1]
            assert root_api.store(group, [], (1,)) is None
        for operation, args in (("load", ((1,), [])), ("store", ([], (1,)))):
            for group in unsupported_groups:
                before = len(observed)
                with pytest.raises(NotImplementedError, match=operation):
                    getattr(root_api, operation)(group, *args)
                assert len(observed) == before

    assert observed == [
        (operation, group_kind)
        for group_kind in ("block", "warp", "threads_within_warp")
        for operation in ("load", "store")
    ]
    for operation in ("load", "store"):
        assert COMMON_PROFILE_MATRIX[operation]["supported_groups"] == (
            "block",
            "physical_warp",
            "threads_within_warp",
        )
