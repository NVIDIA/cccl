# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


class _Payload:
    def __init__(self, *values: int | float, dtype: object = int):
        self._values = list(values)
        self.items_per_thread = len(values)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> int | float:
        return self._values[index]

    def __setitem__(self, index: int, value: int | float) -> None:
        self._values[index] = value

    def values(self) -> tuple[int | float, ...]:
        return tuple(self._values)


@pytest.mark.evidence_for("group.merge_sort_keys", backend="core", evidence="semantics")
def test_common_merge_sort_keys_owns_block_and_warp_group_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, object]]] = []

    def merge_sort_keys(group, keys, /, **kwargs):
        calls.append((group.kind, keys.values(), kwargs))
        return tuple(sorted(keys.values(), reverse=bool(kwargs["descending"])))

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(merge_sort_keys=merge_sort_keys),
    )

    block = root_api.this_block()
    warp = root_api.this_warp()
    logical_warp = warp.group_by(8)
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    payload = _Payload(3, 1, 3)
    original = payload.values()
    storage = object()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.merge_sort_keys(block, payload) == (1, 3, 3)
        assert calls[-1] == (
            "block",
            original,
            {
                "descending": False,
                "valid_items": None,
                "oob_default": None,
                "temp_storage": None,
            },
        )

        assert root_api.merge_sort_keys(
            warp,
            payload,
            descending=True,
            valid_items=2,
            oob_default=-2_147_483_648,
            temp_storage=storage,
        ) == (3, 3, 1)
        assert calls[-1] == (
            "warp",
            original,
            {
                "descending": True,
                "valid_items": 2,
                "oob_default": -2_147_483_648,
                "temp_storage": storage,
            },
        )
        assert root_api.merge_sort_keys(logical_warp, payload) == (1, 3, 3)
        assert calls[-1][0] == "threads_within_warp"

        before = len(calls)
        for kwargs in (
            {"valid_items": 2},
            {"oob_default": 99},
        ):
            with pytest.raises(
                ValueError,
                match="valid_items and oob_default must be provided together",
            ):
                root_api.merge_sort_keys(block, payload, **kwargs)
            assert len(calls) == before

        with pytest.raises(
            TypeError,
            match=r"cuda\.coop\.merge_sort_keys requires a fixed-size ThreadData",
        ):
            root_api.merge_sort_keys(block, 3)
        assert len(calls) == before

        float_payload = _Payload(3.0, 1.0, dtype=float)
        with pytest.raises(
            TypeError,
            match=(
                r"cuda\.coop\.merge_sort_keys common V1 supports key dtypes "
                r"int32, uint32, int64, uint64"
            ),
        ):
            root_api.merge_sort_keys(warp, float_payload)
        assert len(calls) == before

        for group in unsupported_groups:
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.merge_sort_keys does not support group kind",
            ):
                root_api.merge_sort_keys(group, payload)
            assert len(calls) == before

    assert payload.values() == original
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.merge_sort_keys(block, payload)

    profile = COMMON_PROFILE_MATRIX["merge_sort_keys"]
    assert profile["supported_groups"] == (
        "block",
        "physical_warp",
        "threads_within_warp",
    )
    assert profile["mutation_rule"] == "does not mutate inputs"
    assert profile["result_layout"] == "shape-preserving payload"


def test_common_merge_sort_rejects_lossy_sentinels_before_backend_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []

    def merge_sort_keys(_group, _keys, /, **kwargs):
        calls.append(kwargs["oob_default"])
        return kwargs["oob_default"]

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(merge_sort_keys=merge_sort_keys),
    )

    block = root_api.this_block()
    keys = _Payload(3, 1, dtype=int)
    invalid_cases = (
        (
            1.5,
            TypeError,
            r"oob_default must have the same integer dtype as keys \(int32\); "
            r"got float",
        ),
        (
            np.int64(0),
            TypeError,
            r"oob_default must have the same integer dtype as keys \(int32\); "
            r"got int64",
        ),
        (
            True,
            TypeError,
            r"oob_default must have the same integer dtype as keys \(int32\); "
            r"got bool",
        ),
        (
            1 << 31,
            ValueError,
            r"oob_default=2147483648 is not representable in keys dtype int32",
        ),
        (
            -(1 << 31) - 1,
            ValueError,
            r"oob_default=-2147483649 is not representable in keys dtype int32",
        ),
    )

    with root_api._compiler_scope("cuda.coop.testing"):
        for sentinel, error_type, match in invalid_cases:
            with pytest.raises(error_type, match=match):
                root_api.merge_sort_keys(
                    block,
                    keys,
                    valid_items=63,
                    oob_default=sentinel,
                )
            assert calls == []

        for sentinel in (-(1 << 31), (1 << 31) - 1):
            assert (
                root_api.merge_sort_keys(
                    block,
                    keys,
                    valid_items=63,
                    oob_default=sentinel,
                )
                == sentinel
            )

        unsigned_keys = _Payload(
            np.uint32(3),
            np.uint32(1),
            dtype=np.uint32,
        )
        accepted_calls = list(calls)
        for sentinel in (-1, 1 << 32):
            with pytest.raises(
                ValueError,
                match=(
                    rf"oob_default={sentinel} is not representable in keys "
                    r"dtype uint32"
                ),
            ):
                root_api.merge_sort_keys(
                    block,
                    unsigned_keys,
                    valid_items=63,
                    oob_default=sentinel,
                )
            assert calls == accepted_calls

        for sentinel in (np.uint32(0), np.uint32((1 << 32) - 1)):
            assert (
                root_api.merge_sort_keys(
                    block,
                    unsigned_keys,
                    valid_items=63,
                    oob_default=sentinel,
                )
                == sentinel
            )

    assert calls == [
        -(1 << 31),
        (1 << 31) - 1,
        np.uint32(0),
        np.uint32((1 << 32) - 1),
    ]


@pytest.mark.evidence_for(
    "group.merge_sort_pairs", backend="core", evidence="semantics"
)
def test_common_merge_sort_pairs_preserves_correlation_and_restricts_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def merge_sort_pairs(group, keys, values, /, **kwargs):
        calls.append((group, keys, values, kwargs))
        ordered = sorted(
            zip(keys.values(), values.values()), reverse=kwargs["descending"]
        )
        return tuple(key for key, _ in ordered), tuple(value for _, value in ordered)

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(merge_sort_pairs=merge_sort_pairs),
    )
    block = root_api.this_block()
    warp = root_api.this_warp()
    logical_warp = warp.group_by(8)
    keys = _Payload(3, 1, 3, dtype=np.int32)
    values = _Payload(30.5, 10.5, 31.5, dtype=np.float64)
    originals = keys.values(), values.values()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.merge_sort_pairs(block, keys, values) == (
            (1, 3, 3),
            (10.5, 30.5, 31.5),
        )
        root_api.merge_sort_pairs(
            warp,
            keys,
            values,
            descending=True,
            valid_items=2,
            oob_default=np.int32(-(1 << 31)),
        )
        root_api.merge_sort_pairs(logical_warp, keys, values)
        assert calls[-1][0].kind == "threads_within_warp"

        before = len(calls)
        with pytest.raises(ValueError, match="matching items_per_thread"):
            root_api.merge_sort_pairs(block, keys, _Payload(1.0, dtype=float))
        with pytest.raises(TypeError, match="supports value dtypes"):
            root_api.merge_sort_pairs(
                block,
                keys,
                _Payload(True, False, True, dtype=bool),
            )
        with pytest.raises(
            TypeError,
            match="requires a fixed-size ThreadData values payload",
        ):
            root_api.merge_sort_pairs(block, keys, 1.0)
        assert len(calls) == before

    assert (keys.values(), values.values()) == originals
    profile = COMMON_PROFILE_MATRIX["merge_sort_pairs"]
    assert profile["supported_groups"] == (
        "block",
        "physical_warp",
        "threads_within_warp",
    )
    assert profile["dtype_contract"] == {
        "keys": "integer_key",
        "values": "numeric",
    }
