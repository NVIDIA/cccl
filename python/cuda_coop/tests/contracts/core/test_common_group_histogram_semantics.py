# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


class _Payload:
    def __init__(self, *values: int):
        self._values = list(values)
        self.items_per_thread = len(values)
        self.dtype = int

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> int:
        return self._values[index]

    def __setitem__(self, index: int, value: int) -> None:
        self._values[index] = value

    def values(self) -> tuple[int, ...]:
        return tuple(self._values)


@pytest.mark.evidence_for("group.histogram", backend="core", evidence="semantics")
def test_common_histogram_owns_launch_independent_block_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, object]]] = []

    def histogram(group, samples, /, **kwargs):
        calls.append((group.kind, samples.values(), kwargs))
        return tuple(range(kwargs["bins_per_thread"]))

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(histogram=histogram),
    )

    block = root_api.this_block()
    assert block.static_size is None
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_warp(),
        root_api.this_warp().group_by(8),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    payload = _Payload(3, 1, 4)
    original = payload.values()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.histogram(block, payload, bins=65, bins_per_thread=2) == (
            0,
            1,
        )
        assert calls[-1] == (
            "block",
            original,
            {
                "bins": 65,
                "bins_per_thread": 2,
                "counter_dtype": None,
                "algorithm": "atomic",
            },
        )

        assert root_api.histogram(
            block,
            payload,
            bins=32,
            algorithm=" SORT ",
        ) == (0,)
        assert calls[-1][-1]["algorithm"] == "sort"

        before = len(calls)
        with pytest.raises(
            TypeError,
            match=r"cuda\.coop\.histogram requires a fixed-size ThreadData",
        ):
            root_api.histogram(block, 3, bins=8)
        assert len(calls) == before

        with pytest.raises(ValueError, match="algorithm must be one of: atomic, sort"):
            root_api.histogram(block, payload, bins=8, algorithm="custom")
        assert len(calls) == before

        for group in unsupported_groups:
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.histogram does not support group kind",
            ):
                root_api.histogram(group, payload, bins=8)
            assert len(calls) == before

    assert payload.values() == original
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.histogram(block, 3, bins=0, bins_per_thread=0)

    profile = COMMON_PROFILE_MATRIX["histogram"]
    assert profile["supported_groups"] == ("block",)
    assert profile["mutation_rule"] == "does not mutate inputs"
    assert profile["result_layout"] == (
        "striped counters by rank plus i times group size; out-of-range slots are zero"
    )
    assert profile["preconditions"] == (
        "every sample satisfies 0 <= sample < bins; violating this CUB "
        "precondition is undefined behavior",
    )
