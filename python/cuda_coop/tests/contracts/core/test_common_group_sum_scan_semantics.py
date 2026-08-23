# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX

_SCAN_OPERATIONS = (
    "scan",
    "exclusive_sum",
    "inclusive_sum",
    "exclusive_scan",
    "inclusive_scan",
)
_DEFAULT_KEYWORDS = {
    "reduce": {
        "binary_op": None,
        "broadcast": True,
        "valid_items": None,
        "algorithm": None,
    },
    "sum": {
        "broadcast": True,
        "valid_items": None,
        "algorithm": None,
    },
    "scan": {
        "mode": "exclusive",
        "scan_op": None,
        "initial_value": None,
        "algorithm": None,
        "temp_storage": None,
    },
    "exclusive_sum": {"algorithm": None, "temp_storage": None},
    "inclusive_sum": {"algorithm": None, "temp_storage": None},
    "exclusive_scan": {
        "scan_op": None,
        "initial_value": None,
        "algorithm": None,
        "temp_storage": None,
    },
    "inclusive_scan": {
        "scan_op": None,
        "algorithm": None,
        "temp_storage": None,
    },
}


@pytest.mark.evidence_for("group.reduce", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.sum", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.scan", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.exclusive_sum", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.inclusive_sum", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.exclusive_scan", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.inclusive_scan", backend="core", evidence="semantics")
def test_common_reduce_sum_scan_alias_defaults_and_group_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, object, dict[str, object]]] = []

    def make_member(operation: str):
        def member(group, value, /, **kwargs):
            calls.append((operation, group.kind, value, kwargs))
            return operation

        return member

    backend = SimpleNamespace(
        **{operation: make_member(operation) for operation in _DEFAULT_KEYWORDS}
    )
    monkeypatch.setattr(root_api, "import_module", lambda _name: backend)

    thread = root_api.this_thread()
    warp = root_api.this_warp()
    logical_warp = warp.group_by(8)
    block = root_api.this_block()
    cluster = root_api.this_cluster()
    value = [3, 1, 4]
    original_value = value.copy()
    temp_storage = object()

    with root_api._compiler_scope("cuda.coop.testing"):
        for operation, expected_keywords in _DEFAULT_KEYWORDS.items():
            assert getattr(root_api, operation)(block, value) == operation
            assert calls[-1] == (operation, "block", value, expected_keywords)

        assert (
            root_api.sum(
                block,
                value,
                broadcast=False,
                valid_items=17,
                algorithm="warp-reductions",
            )
            == "sum"
        )
        assert calls[-1][-1] == {
            "broadcast": False,
            "valid_items": 17,
            "algorithm": "warp_reductions",
        }
        assert (
            root_api.reduce(
                block,
                value,
                binary_op="max",
                broadcast=False,
                algorithm="raking",
            )
            == "reduce"
        )
        assert calls[-1][-1] == {
            "binary_op": "max",
            "broadcast": False,
            "valid_items": None,
            "algorithm": "raking",
        }
        assert (
            root_api.scan(
                block,
                value,
                mode=" INCLUSIVE ",
                algorithm="warp-scans",
            )
            == "scan"
        )
        assert calls[-1][-1] == {
            "mode": "inclusive",
            "scan_op": None,
            "initial_value": None,
            "algorithm": "warp_scans",
            "temp_storage": None,
        }
        assert (
            root_api.exclusive_scan(
                block,
                value,
                scan_op="max",
                initial_value=0,
                algorithm="raking-memoize",
                temp_storage=temp_storage,
            )
            == "exclusive_scan"
        )
        assert calls[-1][-1] == {
            "scan_op": "max",
            "initial_value": 0,
            "algorithm": "raking_memoize",
            "temp_storage": temp_storage,
        }

        for operation in ("reduce", "sum"):
            for group in (thread, warp, logical_warp, block, cluster):
                assert getattr(root_api, operation)(group, value) == operation

        for operation in _SCAN_OPERATIONS:
            function = getattr(root_api, operation)
            for group in (block, warp, logical_warp):
                assert function(group, value) == operation

            for group in (thread, cluster):
                before = len(calls)
                with pytest.raises(
                    NotImplementedError,
                    match=rf"cuda\.coop\.{operation} does not support group kind",
                ):
                    function(group, value)
                assert len(calls) == before

    assert value == original_value


def test_common_scan_profile_requires_fully_defined_results() -> None:
    assert COMMON_PROFILE_MATRIX["exclusive_sum"]["result_layout"] == (
        "shape-preserving payload with the first flattened position equal to zero"
    )
    for operation in ("scan", "exclusive_scan"):
        result_layout = COMMON_PROFILE_MATRIX[operation]["result_layout"]
        assert "every position defined" in result_layout
        assert "require initial_value" in result_layout
    for operation in ("inclusive_sum", "inclusive_scan"):
        assert (
            "every position defined"
            in (COMMON_PROFILE_MATRIX[operation]["result_layout"])
        )
