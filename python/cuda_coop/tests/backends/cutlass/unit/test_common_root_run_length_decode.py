# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest


@pytest.mark.evidence_for(
    "group.run_length_decode",
    backend="cutlass",
    evidence="lowering",
)
def test_common_and_qualified_decode_share_one_typed_block_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass._mlir import ir
    from cutlass.base_dsl.typing import Uint64

    import cuda.coop.cutlass as cutlass_coop
    from cuda import coop
    from cuda.coop._core import LaunchFacts, root_api
    from cuda.coop.cutlass import _group_run_length_decode as frontend
    from cuda.coop.cutlass._dsl import _cub_run_length_decode_provider as provider
    from cuda.coop.cutlass._dsl import _launch

    launch = LaunchFacts(exact_block_dim=(8, 4, 2))
    monkeypatch.setattr(_launch, "infer_launch_facts", lambda *_args, **_kwargs: launch)

    run_values = cutlass_coop.ThreadData.from_values(11, 29, dtype=Uint64)
    run_lengths = cutlass_coop.ThreadData.from_values(7, 0, dtype=Uint64)
    relative_offsets = cutlass_coop.ThreadData(3, dtype=Uint64)
    total_decoded_size = cutlass_coop.ThreadData(1, dtype=Uint64)
    decoded_window_offset = Uint64((1 << 32) + 5)
    observed: list[dict[str, object]] = []
    monkeypatch.setattr(
        provider,
        "provider_run_length_decode",
        lambda **kwargs: observed.append(kwargs) or ("decoded", len(observed)),
    )

    qualified_result = cutlass_coop.run_length_decode(
        cutlass_coop.this_block(),
        run_values,
        run_lengths,
        decoded_items_per_thread=3,
        decoded_window_offset=decoded_window_offset,
        relative_offsets=relative_offsets,
        total_decoded_size=total_decoded_size,
    )
    with root_api._compiler_scope("cuda.coop.cutlass"):
        common_result = coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=3,
            decoded_window_offset=decoded_window_offset,
        )

    assert qualified_result[0] == common_result[0] == "decoded"
    qualified_call, common_call = observed
    assert qualified_call["group"] == common_call["group"]
    assert qualified_call["run_values"] is common_call["run_values"] is run_values
    assert qualified_call["run_lengths"] is common_call["run_lengths"] is run_lengths
    assert qualified_call["decoded_items_per_thread"] == 3
    assert common_call["decoded_items_per_thread"] == 3
    assert qualified_call["decoded_window_offset"] is decoded_window_offset
    assert common_call["decoded_window_offset"] is decoded_window_offset
    assert qualified_call["relative_offsets"] is relative_offsets
    assert qualified_call["total_decoded_size"] is total_decoded_size
    assert common_call["relative_offsets"] is None
    assert common_call["total_decoded_size"] is None
    assert qualified_call["source"] == common_call["source"] == "cutlass_root"

    plan = frontend._make_group_run_length_decode_plan(
        group=common_call["group"],
        launch=launch,
        item_dtype=Uint64,
        run_length_dtype=Uint64,
        runs_per_thread=2,
        decoded_items_per_thread=3,
        with_relative_offsets=False,
        source="common_root_test",
    ).require_supported()
    assert plan.implementation.template_arguments["DecodedOffsetT"] is Uint64
    assert plan.implementation.template_arguments["RunLengthT"] is Uint64
    assert plan.implementation.template_arguments["TotalDecodedSizeT"] is Uint64

    artifact = provider._make_request(
        group=common_call["group"],
        launch=launch,
        value_type=Uint64,
        length_type=Uint64,
        runs_per_thread=2,
        decoded_items_per_thread=3,
        with_relative_offsets=False,
        source="common_root_test",
    )
    with ir.Context():
        assert artifact.ffi_param_types[4] is Uint64
    source = "\n".join(provider.render_run_length_decode_artifact(artifact))
    assert "unsigned long long decoded_window_offset" in source
    assert "decoded_window_offset < 0" not in source
    assert "static_cast<unsigned long long>(decoded_window_offset)" in source


@pytest.mark.parametrize(
    ("value", "error", "match"),
    [
        (-1, ValueError, "must be nonnegative"),
        (False, TypeError, "must be an integer"),
        (1.5, TypeError, "must be an integer"),
        ("7", TypeError, "must be an integer"),
    ],
)
def test_qualified_group_and_scoped_decode_reject_invalid_static_offsets(
    value: object,
    error: type[Exception],
    match: str,
) -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Uint32

    import cuda.coop.cutlass as coop

    run_values = coop.ThreadData.from_values(3, 5, dtype=Uint32)
    run_lengths = coop.ThreadData.from_values(1, 1, dtype=Uint32)

    with pytest.raises(error, match=match):
        coop.run_length_decode(
            coop.this_block(),
            run_values,
            run_lengths,
            decoded_items_per_thread=2,
            decoded_window_offset=value,
        )
    with pytest.raises(error, match=match):
        coop._block.run_length_decode(
            run_values,
            run_lengths,
            decoded_items_per_thread=2,
            decoded_window_offset=value,
        )


def test_compiler_integer_literal_offset_is_validated_without_losing_its_type() -> None:
    pytest.importorskip("cutlass.cute.ffi")
    from cutlass.base_dsl.typing import Int64, Uint64

    from cuda.coop.cutlass._run_length_controls import (
        validate_decoded_window_offset,
    )

    high_offset = Uint64((1 << 32) + 9)
    assert (
        validate_decoded_window_offset(high_offset, scope="cuda.coop.cutlass")
        is high_offset
    )
    with pytest.raises(ValueError, match="must be nonnegative"):
        validate_decoded_window_offset(Int64(-1), scope="cuda.coop.cutlass")


@pytest.mark.parametrize("literal", [1.5, "7"])
def test_statically_known_noninteger_compiler_offset_is_rejected(
    literal: object,
) -> None:
    from cuda.coop.cutlass._run_length_controls import (
        validate_decoded_window_offset,
    )

    class StaticCompilerValue:
        width = 32
        signed = False
        dtype = object()

        def __init__(self, value: object) -> None:
            self.value = value

        def ir_value(self) -> object:
            return object()

    with pytest.raises(TypeError, match="must be an integer"):
        validate_decoded_window_offset(
            StaticCompilerValue(literal),
            scope="cuda.coop.cutlass",
        )
