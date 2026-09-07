# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect

import pytest


def _provider_dependencies() -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")


def _launch_facts(block_dim: tuple[int, int, int] = (32, 1, 1)):
    from cuda.coop._core import LaunchFactOrigin, LaunchFacts

    return LaunchFacts(
        exact_block_dim=block_dim,
        provenance=LaunchFactOrigin(
            "exact_block_dim",
            "test_kernel",
            verified=True,
        ),
    )


def test_public_exports_and_signatures() -> None:
    _provider_dependencies()

    import cuda.coop.cutlass as coop

    for name in ("histogram", "run_length_decode"):
        assert name in coop.__all__
        function = getattr(coop, name)
        assert function.__module__.startswith("cuda.coop.cutlass._group_")
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(function).parameters
        )


def test_frontends_delegate_complete_block_payloads(monkeypatch) -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32, Uint32, Uint64

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._compiler import _launch
    from cuda.coop.cutlass._lowering import _histogram as histogram_provider
    from cuda.coop.cutlass._lowering import (
        _run_length_decode as run_length_provider,
    )

    monkeypatch.setattr(_launch, "current_kernel_launch_facts", _launch_facts)
    samples = coop.ThreadData.from_values(Int32(3), Int32(1), dtype=Int32)
    run_values = coop.ThreadData.from_values(Int32(7), Int32(9), dtype=Int32)
    run_lengths = coop.ThreadData.from_values(Uint32(2), Uint32(1), dtype=Uint32)
    calls: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(
        histogram_provider,
        "provider_histogram",
        lambda **kwargs: calls.append(("histogram", kwargs)) or "counts",
    )
    monkeypatch.setattr(
        run_length_provider,
        "provider_run_length_decode",
        lambda **kwargs: calls.append(("decode", kwargs)) or "decoded",
    )

    assert (
        coop.histogram(
            coop.this_block(),
            samples,
            bins=32,
            bins_per_thread=1,
            counter_dtype=Uint64,
            algorithm="sort",
        )
        == "counts"
    )
    assert (
        coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=3,
            decoded_window_offset=Uint32(2),
        )
        == "decoded"
    )

    histogram_call = calls[0][1]
    assert histogram_call["group"].hierarchy.block_dim == (32, 1, 1)
    assert histogram_call["samples"] is samples
    assert histogram_call["bins"] == 32
    assert histogram_call["bins_per_thread"] == 1
    assert histogram_call["counter_dtype"] is Uint64
    assert histogram_call["algorithm"].value == "sort"

    decode_call = calls[1][1]
    assert decode_call["group"].hierarchy.block_dim == (32, 1, 1)
    assert decode_call["run_values"] is run_values
    assert decode_call["run_lengths"] is run_lengths
    assert decode_call["decoded_items_per_thread"] == 3
    assert isinstance(decode_call["decoded_window_offset"], Uint32)


def test_plans_render_internal_storage_and_output_projection() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32, Uint8, Uint32, Uint64

    import cuda.coop.cutlass as coop
    from cuda.coop._core import LaunchFacts
    from cuda.coop.cutlass._lowering import _histogram as histogram_provider
    from cuda.coop.cutlass._lowering import (
        _run_length_decode as run_length_provider,
    )

    launch = LaunchFacts(exact_block_dim=(32, 1, 1))
    histogram = histogram_provider._make_request(
        group=coop.this_block(),
        launch=launch,
        sample_type=Uint8,
        counter_type=Uint64,
        items_per_thread=2,
        bins=32,
        bins_per_thread=1,
        algorithm="atomic",
        source="test",
    )
    histogram_source = "\n".join(
        histogram_provider.render_histogram_artifact(histogram)
    )
    assert "::cub::BlockHistogram<unsigned char, 32, 2, 32" in histogram_source
    assert "__shared__ unsigned long long histogram[32]" in histogram_source
    assert "unsigned long long* histogram_result" in histogram_source
    assert "histogram_result[0] = bin_0 < 32u" in histogram_source

    decode = run_length_provider._make_request(
        group=coop.this_block(),
        launch=launch,
        value_type=Int32,
        length_type=Uint32,
        runs_per_thread=2,
        decoded_items_per_thread=4,
        with_relative_offsets=True,
        source="test",
    )
    decode_source = "\n".join(
        run_length_provider.render_run_length_decode_artifact(decode)
    )
    assert "::cub::BlockRunLengthDecode" in decode_source
    assert "decoded_window_offset" in decode_source
    assert "relative_offsets_result" in decode_source
    assert "total_decoded_size_result" in decode_source
    assert "static_cast<unsigned long long>" in decode_source


def test_decode_controls_and_payload_shapes_are_strict() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32, Int64, Uint32

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass import _run_length_controls
    from cuda.coop.cutlass._lowering import (
        _run_length_decode as run_length_provider,
    )

    for invalid in (True, 1.5, "3"):
        with pytest.raises(TypeError, match="must be an integer"):
            _run_length_controls.validate_decoded_window_offset(
                invalid,
                scope="cuda.coop.cutlass",
            )
    with pytest.raises(ValueError, match="nonnegative"):
        _run_length_controls.validate_decoded_window_offset(
            -1,
            scope="cuda.coop.cutlass",
        )

    values = coop.ThreadData.from_values(Int32(3), Int32(5), dtype=Int32)
    short_lengths = coop.ThreadData.from_values(Uint32(2), dtype=Uint32)
    with pytest.raises(ValueError, match="matching ThreadData.items_per_thread"):
        run_length_provider._resolve_run_inputs(
            run_values=values,
            run_lengths=short_lengths,
        )
    with pytest.raises(TypeError, match="both be ThreadData or both be scalar"):
        run_length_provider._resolve_run_inputs(
            run_values=values,
            run_lengths=Uint32(2),
        )

    with pytest.raises(ValueError, match="does not fit"):
        run_length_provider._as_decoded_window_offset(
            1 << 32,
            length_type=Uint32,
        )
    with pytest.raises(TypeError, match="dtype must match"):
        run_length_provider._as_decoded_window_offset(
            Int64(3),
            length_type=Uint32,
        )
