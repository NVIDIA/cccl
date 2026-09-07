# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest

from cuda.coop._core import root_api

from ...support.cases.api_contracts import COMMON_PROFILE_MATRIX


class _Payload:
    def __init__(
        self,
        *values: int,
        dtype: object = int,
        items_per_thread: object | None = None,
    ) -> None:
        self._values = list(values)
        self.items_per_thread = (
            len(values) if items_per_thread is None else items_per_thread
        )
        self.dtype = dtype

    def __len__(self) -> int:
        return len(self._values)

    def __getitem__(self, index: int) -> int:
        return self._values[index]

    def __setitem__(self, index: int, value: int) -> None:
        self._values[index] = value

    def values(self) -> tuple[int, ...]:
        return tuple(self._values)


class _CompilerInteger:
    width = 64
    signed = False
    dtype = object()

    def ir_value(self) -> object:
        return object()


def _decode_blocked_window(
    run_values: tuple[tuple[int, ...], ...],
    run_lengths: tuple[tuple[int, ...], ...],
    *,
    decoded_items_per_thread: int,
    decoded_window_offset: int,
) -> tuple[tuple[int, ...], ...]:
    """Decode valid V1 runs: positive actual lengths, then optional zero padding."""

    flattened_values = tuple(value for member in run_values for value in member)
    flattened_lengths = tuple(length for member in run_lengths for length in member)
    stream = tuple(
        value
        for value, length in zip(flattened_values, flattened_lengths, strict=True)
        for _ in range(length)
    )
    result = []
    for rank in range(len(run_values)):
        first = decoded_window_offset + rank * decoded_items_per_thread
        result.append(
            tuple(
                stream[position] if position < len(stream) else 0
                for position in range(first, first + decoded_items_per_thread)
            )
        )
    return tuple(result)


@pytest.mark.evidence_for(
    "group.run_length_decode", backend="core", evidence="semantics"
)
def test_common_run_length_decode_owns_blocked_window_and_validation_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The final zero is trailing padding. An internal zero followed by a
    # positive length is outside the V1 precondition and is not normalized.
    assert _decode_blocked_window(
        ((10, 11), (20, 21)),
        ((2, 1), (2, 0)),
        decoded_items_per_thread=3,
        decoded_window_offset=1,
    ) == ((10, 11, 20), (20, 0, 0))

    calls: list[tuple[str, tuple[int, ...], tuple[int, ...], dict[str, object]]] = []

    def run_length_decode(group, run_values, run_lengths, /, **kwargs):
        calls.append(
            (group.kind, run_values.values(), run_lengths.values(), dict(kwargs))
        )
        return _Payload(
            *(0 for _ in range(kwargs["decoded_items_per_thread"])),
            dtype=run_values.dtype,
        )

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(run_length_decode=run_length_decode),
    )

    block = root_api.this_block()
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_warp(),
        root_api.this_warp().group_by(8),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    values = _Payload(10, 20, dtype=np.uint8)
    lengths = _Payload(1, 2, dtype=np.uint64)
    original_values = values.values()
    original_lengths = lengths.values()

    with root_api._compiler_scope("cuda.coop.testing"):
        decoded = root_api.run_length_decode(
            block,
            values,
            lengths,
            decoded_items_per_thread=np.int32(3),
            decoded_window_offset=np.uint64(1),
        )
        assert decoded.items_per_thread == 3
        assert decoded.dtype is np.uint8
        assert calls[-1] == (
            "block",
            original_values,
            original_lengths,
            {
                "decoded_items_per_thread": np.int32(3),
                "decoded_window_offset": np.uint64(1),
            },
        )

        dynamic_offset = _CompilerInteger()
        root_api.run_length_decode(
            block,
            _Payload(1, dtype=SimpleNamespace(width=64, signed=True)),
            _Payload(2, dtype=SimpleNamespace(bitwidth=32, signed=False)),
            decoded_items_per_thread=1,
            decoded_window_offset=dynamic_offset,
        )
        assert calls[-1][-1]["decoded_window_offset"] is dynamic_offset

        for value_dtype in (int, np.uint8, np.int32, np.uint32, np.int64, np.uint64):
            root_api.run_length_decode(
                block,
                _Payload(7, dtype=value_dtype),
                _Payload(1, dtype=np.uint32),
                decoded_items_per_thread=1,
            )
        for length_dtype in (int, np.int32, np.uint32, np.int64, np.uint64):
            root_api.run_length_decode(
                block,
                _Payload(7, dtype=np.int32),
                _Payload(1, dtype=length_dtype),
                decoded_items_per_thread=1,
            )
        for length_dtype, maximum_offset in (
            (int, (1 << 31) - 1),
            (np.int32, (1 << 31) - 1),
            (np.uint32, (1 << 32) - 1),
            (np.int64, (1 << 63) - 1),
            (np.uint64, (1 << 64) - 1),
        ):
            calls_before_offset = len(calls)
            root_api.run_length_decode(
                block,
                _Payload(7, dtype=np.int32),
                _Payload(1, dtype=length_dtype),
                decoded_items_per_thread=1,
                decoded_window_offset=maximum_offset,
            )
            assert len(calls) == calls_before_offset + 1
            with pytest.raises(
                ValueError,
                match="decoded_window_offset must be representable",
            ):
                root_api.run_length_decode(
                    block,
                    _Payload(7, dtype=np.int32),
                    _Payload(1, dtype=length_dtype),
                    decoded_items_per_thread=1,
                    decoded_window_offset=maximum_offset + 1,
                )
            assert len(calls) == calls_before_offset + 1

        before = len(calls)
        for parameter, run_values, run_lengths in (
            ("run_values", 7, lengths),
            ("run_lengths", values, 2),
        ):
            with pytest.raises(
                TypeError,
                match=rf"requires a fixed-size ThreadData {parameter} payload",
            ):
                root_api.run_length_decode(
                    block,
                    run_values,
                    run_lengths,
                    decoded_items_per_thread=1,
                )
        with pytest.raises(ValueError, match="matching items_per_thread"):
            root_api.run_length_decode(
                block,
                _Payload(1, 2),
                _Payload(1),
                decoded_items_per_thread=1,
            )
        for parameter, payload in (
            ("run_values", _Payload(dtype=int, items_per_thread=0)),
            ("run_lengths", _Payload(dtype=int, items_per_thread=False)),
            ("run_values", _Payload(1, dtype=int, items_per_thread=2)),
        ):
            exception = TypeError if payload.items_per_thread is False else ValueError
            with pytest.raises(
                exception,
                match=rf"{parameter}\.items_per_thread",
            ):
                root_api.run_length_decode(
                    block,
                    payload if parameter == "run_values" else _Payload(1),
                    payload if parameter == "run_lengths" else _Payload(1),
                    decoded_items_per_thread=1,
                )
        for parameter, value_dtype, length_dtype in (
            ("run_values", np.uint16, np.uint32),
            ("run_values", np.float32, np.uint32),
            ("run_lengths", np.int32, np.uint8),
            ("run_lengths", np.int32, np.float64),
        ):
            with pytest.raises(TypeError, match=rf"{parameter} dtypes"):
                root_api.run_length_decode(
                    block,
                    _Payload(1, dtype=value_dtype),
                    _Payload(1, dtype=length_dtype),
                    decoded_items_per_thread=1,
                )
        for count, exception in (
            (True, TypeError),
            (1.5, TypeError),
            ("2", TypeError),
            (0, ValueError),
            (-1, ValueError),
        ):
            with pytest.raises(
                exception,
                match="decoded_items_per_thread must be a compile-time positive integer",
            ):
                root_api.run_length_decode(
                    block,
                    values,
                    lengths,
                    decoded_items_per_thread=count,
                )
        for offset, exception, message in (
            (True, TypeError, "must be an int-like scalar"),
            (1.5, TypeError, "must be an int-like scalar"),
            (object(), TypeError, "must be an int-like scalar"),
            (-1, ValueError, "must be non-negative"),
        ):
            with pytest.raises(exception, match=message):
                root_api.run_length_decode(
                    block,
                    values,
                    lengths,
                    decoded_items_per_thread=1,
                    decoded_window_offset=offset,
                )
        for group in unsupported_groups:
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.run_length_decode does not support group kind",
            ):
                root_api.run_length_decode(
                    group,
                    values,
                    lengths,
                    decoded_items_per_thread=1,
                )
        assert len(calls) == before

    assert values.values() == original_values
    assert lengths.values() == original_lengths
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.run_length_decode(
            block,
            7,
            2,
            decoded_items_per_thread=0,
            decoded_window_offset=-1,
        )

    profile = COMMON_PROFILE_MATRIX["run_length_decode"]
    assert profile["supported_groups"] == ("block",)
    assert profile["mutation_rule"] == "does not mutate inputs"
    assert profile["result_layout"] == (
        "decoded_items_per_thread values per member in blocked window order; "
        "out-of-range positions are zero"
    )
