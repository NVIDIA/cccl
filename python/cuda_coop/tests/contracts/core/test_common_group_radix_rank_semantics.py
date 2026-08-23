# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


class _Payload:
    def __init__(self, *values: int, dtype: object = int):
        self._values = list(values)
        self.items_per_thread = len(values)
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> int:
        return self._values[index]

    def __setitem__(self, index: int, value: int) -> None:
        self._values[index] = value

    def values(self) -> tuple[int, ...]:
        return tuple(self._values)


@pytest.mark.evidence_for("group.radix_rank", backend="core", evidence="semantics")
def test_common_radix_rank_owns_blocked_bit_ordered_integer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, object]]] = []

    def radix_rank(group, keys, /, **kwargs):
        calls.append((group.kind, keys.values(), kwargs))
        return tuple(range(len(keys)))

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(radix_rank=radix_rank),
    )

    block = root_api.this_block()
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_warp(),
        root_api.this_warp().group_by(8),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    payload = _Payload(-(1 << 31), -1, 0, (1 << 31) - 1)
    original = payload.values()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.radix_rank(block, payload) == (0, 1, 2, 3)
        assert calls[-1] == (
            "block",
            original,
            {
                "begin_bit": 0,
                "end_bit": None,
                "radix_bits": None,
                "descending": False,
            },
        )

        uint64_payload = _Payload(
            1 << 63,
            (1 << 64) - 1,
            dtype=SimpleNamespace(name="uint64"),
        )
        root_api.radix_rank(
            block,
            uint64_payload,
            begin_bit=np.int32(56),
            end_bit=np.int32(64),
            radix_bits=np.int32(8),
            descending=True,
        )
        assert calls[-1][2] == {
            "begin_bit": np.int32(56),
            "end_bit": np.int32(64),
            "radix_bits": np.int32(8),
            "descending": True,
        }

        before = len(calls)
        with pytest.raises(
            TypeError,
            match=r"cuda\.coop\.radix_rank requires a fixed-size ThreadData",
        ):
            root_api.radix_rank(block, 3)
        for dtype in (float, SimpleNamespace(name="int16")):
            with pytest.raises(
                TypeError,
                match=(
                    r"cuda\.coop\.radix_rank common V1 supports key dtypes "
                    r"int32, uint32, int64, uint64"
                ),
            ):
                root_api.radix_rank(block, _Payload(1, 2, dtype=dtype))
        with pytest.raises(
            TypeError,
            match="descending must be a compile-time bool",
        ):
            root_api.radix_rank(block, payload, descending=1)
        for name, kwargs in (
            ("begin_bit", {"begin_bit": True}),
            ("begin_bit", {"begin_bit": 1.5}),
            ("end_bit", {"end_bit": object()}),
            ("radix_bits", {"radix_bits": False}),
        ):
            with pytest.raises(
                TypeError,
                match=rf"{name} must be a compile-time integer",
            ):
                root_api.radix_rank(block, payload, **kwargs)
        for kwargs, message in (
            ({"begin_bit": -1}, "begin_bit must be non-negative"),
            ({"begin_bit": 32}, "begin_bit must be < 32"),
            ({"end_bit": 33}, "end_bit must be <= 32"),
            (
                {"begin_bit": 8, "end_bit": 8},
                "end_bit must be greater than begin_bit",
            ),
            ({"radix_bits": 0}, "radix_bits must be positive"),
            ({"radix_bits": 9}, "bit width must be <= 8"),
            (
                {"end_bit": 8, "radix_bits": 4},
                "radix_bits must match end_bit - begin_bit",
            ),
        ):
            with pytest.raises(ValueError, match=message):
                root_api.radix_rank(block, payload, **kwargs)
        assert len(calls) == before

        for group in unsupported_groups:
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.radix_rank does not support group kind",
            ):
                root_api.radix_rank(group, payload)
            assert len(calls) == before

    assert payload.values() == original
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.radix_rank(block, payload)

    profile = COMMON_PROFILE_MATRIX["radix_rank"]
    assert profile["supported_groups"] == ("block",)
    assert profile["mutation_rule"] == "does not mutate inputs"
    assert profile["result_layout"] == "shape-preserving int32 ranks"
