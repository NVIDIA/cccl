# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import numpy as np
import pytest


@pytest.mark.evidence_for("group.histogram", backend="cutlass", evidence="lowering")
def test_group_histogram_normalizes_portable_python_and_numpy_dtypes() -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int32, Int64, Uint8, Uint32, Uint64

    from cuda.coop.cutlass._dsl._cub_histogram_provider import (
        _HISTOGRAM_SAMPLE_TYPES,
        _resolve_histogram_type,
    )

    def resolve(value):
        return _resolve_histogram_type(
            value,
            allowed=_HISTOGRAM_SAMPLE_TYPES,
            feature="histogram",
        )

    assert resolve(int) is Int32
    assert resolve(1) is Int32
    for ordinary, expected in (
        (np.uint8, Uint8),
        (np.int32, Int32),
        (np.uint32, Uint32),
        (np.int64, Int64),
        (np.uint64, Uint64),
    ):
        assert resolve(ordinary) is expected
        assert resolve(ordinary(1)) is expected


@pytest.mark.evidence_for("group.histogram", backend="cutlass", evidence="lowering")
def test_group_histogram_routes_to_one_exact_cub_block_plan(
    monkeypatch: pytest.MonkeyPatch,
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts((8, 4, 2))
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int64, Uint8

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupLoweringTarget, LaunchFacts, ResultVisibility
    from cuda.coop.cutlass import _group_histogram
    from cuda.coop.cutlass._dsl import _cub_histogram_provider as provider

    samples = coop.ThreadData.from_values(3, 1, 4, dtype=Uint8)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_histogram",
        lambda **kwargs: observed.append(kwargs) or "counts",
    )

    block = coop.this_block()
    assert (
        coop.histogram(
            block,
            samples,
            bins=65,
            bins_per_thread=2,
            counter_dtype=Int64,
            algorithm="sort",
        )
        == "counts"
    )
    assert len(observed) == 1
    call = observed[0]
    assert call["group"].block_dim == (8, 4, 2)
    assert call["samples"] is samples
    assert call["bins"] == 65
    assert call["bins_per_thread"] == 2
    assert call["counter_dtype"] is Int64
    assert call["algorithm"].value == "sort"
    assert call["source"] == "cutlass_root"

    plan = _group_histogram._make_group_histogram_plan(
        group=block,
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        item_dtype=Uint8,
        counter_dtype=Int64,
        items_per_thread=3,
        bins=65,
        bins_per_thread=2,
        algorithm="sort",
    ).require_supported()
    assert plan.target is GroupLoweringTarget.CUB_BLOCK
    assert plan.call.source == "cutlass_root"
    assert plan.implementation.template_arguments["BINS"] == 65
    assert plan.implementation.template_arguments["ITEMS_PER_THREAD"] == 3
    assert plan.implementation.method_name == "Histogram"
    assert plan.result.visibility is ResultVisibility.PER_MEMBER
    assert plan.result.result_items_per_thread == 2


@pytest.mark.evidence_for("group.histogram", backend="cutlass", evidence="lowering")
def test_group_histogram_rejects_incomplete_projection_and_non_block_group(
    set_cutlass_launch_facts,
) -> None:
    set_cutlass_launch_facts(64)
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Uint8

    import cuda.coop.cutlass as coop

    samples = coop.ThreadData.from_values(Uint8(0), Uint8(1), dtype=Uint8)
    with pytest.raises(
        TypeError,
        match="histogram bins must be a compile-time positive integer",
    ):
        coop.histogram(
            coop.this_block(),
            samples,
            bins="65",
        )

    with pytest.raises(
        ValueError,
        match="histogram bins_per_thread must be a compile-time positive integer",
    ):
        coop.histogram(
            coop.this_block(),
            samples,
            bins=65,
            bins_per_thread=0,
        )

    with pytest.raises(
        ValueError,
        match=(
            "histogram bins_per_thread is too small for 65 bins and block "
            "size 64; need at least 2"
        ),
    ):
        coop.histogram(
            coop.this_block(),
            samples,
            bins=65,
        )

    with pytest.raises(NotImplementedError, match="only this_block"):
        coop.histogram(
            coop.this_warp(),
            samples,
            bins=8,
        )
