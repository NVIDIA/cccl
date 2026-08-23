# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api


@pytest.mark.evidence_for("group.exchange", backend="core", evidence="semantics")
def test_common_exchange_preserves_payload_and_group_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, list[int], str]] = []

    def exchange(group, value, /, *, mode="striped_to_blocked"):
        calls.append((group.kind, value.copy(), mode))
        return tuple(value if mode == "striped_to_blocked" else reversed(value))

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(exchange=exchange),
    )

    warp = root_api.this_warp()
    logical_warp = warp.group_by(8)
    block = root_api.this_block()
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    payload = [3, 1, 4]

    with root_api._compiler_scope("cuda.coop.testing"):
        for group in (block, warp, logical_warp):
            assert root_api.exchange(group, payload) == (3, 1, 4)
            assert root_api.exchange(
                group,
                payload,
                mode="blocked-to-striped",
            ) == (4, 1, 3)

        for group in unsupported_groups:
            before = len(calls)
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.exchange does not support group kind",
            ):
                root_api.exchange(group, payload)
            assert len(calls) == before

    assert payload == [3, 1, 4]
    assert calls == [
        ("block", payload, "striped_to_blocked"),
        ("block", payload, "blocked_to_striped"),
        ("warp", payload, "striped_to_blocked"),
        ("warp", payload, "blocked_to_striped"),
        ("threads_within_warp", payload, "striped_to_blocked"),
        ("threads_within_warp", payload, "blocked_to_striped"),
    ]
