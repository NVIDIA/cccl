# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


@pytest.mark.evidence_for("group.shuffle", backend="core", evidence="semantics")
def test_common_shuffle_routes_unit_block_shifts_without_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], str, int]] = []

    def shuffle(group, value, /, *, mode="down", distance=1):
        calls.append((group.kind, tuple(value), mode, distance))
        if mode == "up":
            return (None, *value[:-1])
        return (*value[1:], None)

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(shuffle=shuffle),
    )

    block = root_api.this_block()
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_warp(),
        root_api.this_warp().group_by(8),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    payload = [3, 1, 4]
    original_payload = payload.copy()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.shuffle(block, payload) == (1, 4, None)
        assert root_api.shuffle(block, payload, mode=" UP ") == (None, 3, 1)

        for kwargs, message in (
            (
                {"mode": "rotate"},
                "mode must be one of: down, up",
            ),
            (
                {"mode": "offset"},
                "mode must be one of: down, up",
            ),
            (
                {"distance": 2},
                "distance must be exactly 1 in common V1",
            ),
            (
                {"distance": True},
                "distance must be exactly 1 in common V1",
            ),
            (
                {"distance": 1.0},
                "distance must be exactly 1 in common V1",
            ),
        ):
            before = len(calls)
            with pytest.raises(ValueError, match=message):
                root_api.shuffle(block, payload, **kwargs)
            assert len(calls) == before

        for group in unsupported_groups:
            before = len(calls)
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.shuffle does not support group kind",
            ):
                root_api.shuffle(group, payload)
            assert len(calls) == before

    assert payload == original_payload
    assert calls == [
        ("block", tuple(payload), "down", 1),
        ("block", tuple(payload), "up", 1),
    ]
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.shuffle(block, payload, mode="rotate", distance=2)
    assert COMMON_PROFILE_MATRIX["shuffle"]["supported_groups"] == ("block",)
    assert COMMON_PROFILE_MATRIX["shuffle"]["mutation_rule"] == (
        "does not mutate inputs"
    )
    assert COMMON_PROFILE_MATRIX["shuffle"]["result_layout"] == (
        "shape-preserving payload; vacated first or last flattened item undefined"
    )
