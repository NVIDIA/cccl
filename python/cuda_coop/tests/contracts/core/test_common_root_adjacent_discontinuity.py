# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="core", evidence="semantics"
)
@pytest.mark.evidence_for("group.discontinuity", backend="core", evidence="semantics")
def test_common_adjacent_discontinuity_routes_block_only_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[int, ...], dict[str, object]]] = []

    def adjacent_difference(group, value, /, **kwargs):
        calls.append(("adjacent_difference", group.kind, tuple(value), kwargs))
        return tuple(value)

    def discontinuity(group, value, /, **kwargs):
        calls.append(("discontinuity", group.kind, tuple(value), kwargs))
        return tuple(int(item != value[0]) for item in value)

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(
            adjacent_difference=adjacent_difference,
            discontinuity=discontinuity,
        ),
    )

    thread = root_api.this_thread()
    warp = root_api.this_warp()
    logical_warp = warp.group_by(8)
    block = root_api.this_block()
    cluster = root_api.this_cluster()
    grid = root_api.this_grid()
    payload = [3, 3, 8]
    original_payload = payload.copy()
    temp_storage = object()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.adjacent_difference(block, payload) == (3, 3, 8)
        assert root_api.adjacent_difference(
            block,
            payload,
            direction=" LEFT ",
            valid_items=61,
            tile_predecessor_item=-7,
            temp_storage=temp_storage,
        ) == (3, 3, 8)
        assert root_api.adjacent_difference(
            block,
            payload,
            direction=" RIGHT ",
            tile_successor_item=-7,
            temp_storage=temp_storage,
        ) == (3, 3, 8)
        assert root_api.discontinuity(block, payload) == (0, 0, 1)
        assert root_api.discontinuity(
            block,
            payload,
            mode=" TAILS ",
            tile_successor_item=-7,
            temp_storage=temp_storage,
        ) == (0, 0, 1)

        for operation, kwargs, message in (
            (
                "adjacent_difference",
                {"direction": "subtract_left"},
                "direction must be one of: left, right",
            ),
            (
                "discontinuity",
                {"mode": "heads_and_tails"},
                "mode must be one of: heads, tails",
            ),
        ):
            before = len(calls)
            with pytest.raises(ValueError, match=message):
                getattr(root_api, operation)(block, payload, **kwargs)
            assert len(calls) == before

        for operation in ("adjacent_difference", "discontinuity"):
            for group in (thread, warp, logical_warp, cluster, grid):
                before = len(calls)
                with pytest.raises(
                    NotImplementedError,
                    match=rf"cuda\.coop\.{operation} does not support group kind",
                ):
                    getattr(root_api, operation)(group, payload)
                assert len(calls) == before

    assert payload == original_payload
    assert calls == [
        (
            "adjacent_difference",
            "block",
            tuple(payload),
            {
                "direction": "left",
                "valid_items": None,
                "tile_predecessor_item": None,
                "tile_successor_item": None,
                "temp_storage": None,
            },
        ),
        (
            "adjacent_difference",
            "block",
            tuple(payload),
            {
                "direction": "left",
                "valid_items": 61,
                "tile_predecessor_item": -7,
                "tile_successor_item": None,
                "temp_storage": temp_storage,
            },
        ),
        (
            "adjacent_difference",
            "block",
            tuple(payload),
            {
                "direction": "right",
                "valid_items": None,
                "tile_predecessor_item": None,
                "tile_successor_item": -7,
                "temp_storage": temp_storage,
            },
        ),
        (
            "discontinuity",
            "block",
            tuple(payload),
            {
                "mode": "heads",
                "tile_predecessor_item": None,
                "tile_successor_item": None,
                "temp_storage": None,
            },
        ),
        (
            "discontinuity",
            "block",
            tuple(payload),
            {
                "mode": "tails",
                "tile_predecessor_item": None,
                "tile_successor_item": -7,
                "temp_storage": temp_storage,
            },
        ),
    ]
    assert COMMON_PROFILE_MATRIX["adjacent_difference"]["supported_groups"] == (
        "block",
    )
    assert COMMON_PROFILE_MATRIX["adjacent_difference"]["mutation_rule"] == (
        "does not mutate inputs"
    )
    assert COMMON_PROFILE_MATRIX["discontinuity"]["supported_groups"] == ("block",)
    assert COMMON_PROFILE_MATRIX["discontinuity"]["mutation_rule"] == (
        "does not mutate inputs"
    )
    assert COMMON_PROFILE_MATRIX["discontinuity"]["result_layout"] == (
        "shape-preserving int32 flags"
    )
