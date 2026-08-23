# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

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


class _CompilerInteger:
    width = 32
    signed = True
    dtype = object()

    def ir_value(self) -> object:
        return object()


@pytest.mark.evidence_for("group.radix_sort_keys", backend="core", evidence="semantics")
def test_common_radix_sort_keys_owns_complete_block_integer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, tuple[int, ...], dict[str, object]]] = []

    def radix_sort_keys(group, keys, /, **kwargs):
        calls.append((group.kind, keys.values(), kwargs))
        return tuple(sorted(keys.values(), reverse=bool(kwargs["descending"])))

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(radix_sort_keys=radix_sort_keys),
    )

    block = root_api.this_block()
    warp = root_api.this_warp()
    unsupported_groups = (
        root_api.this_thread(),
        warp,
        warp.group_by(8),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    payload = _Payload(7, -3, 7, 2)
    original = payload.values()
    storage = object()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.radix_sort_keys(block, payload) == (-3, 2, 7, 7)
        assert calls[-1] == (
            "block",
            original,
            {
                "begin_bit": 0,
                "end_bit": None,
                "descending": False,
                "temp_storage": None,
            },
        )

        uint64_payload = _Payload(
            1 << 63,
            3,
            dtype=SimpleNamespace(name="uint64"),
        )
        assert root_api.radix_sort_keys(
            block,
            uint64_payload,
            begin_bit=8,
            descending=True,
            temp_storage=storage,
        ) == (1 << 63, 3)
        assert calls[-1] == (
            "block",
            uint64_payload.values(),
            {
                "begin_bit": 8,
                "end_bit": None,
                "descending": True,
                "temp_storage": storage,
            },
        )

        compiler_begin = _CompilerInteger()
        compiler_end = _CompilerInteger()
        root_api.radix_sort_keys(
            block,
            payload,
            begin_bit=compiler_begin,
            end_bit=compiler_end,
        )
        assert calls[-1][2]["begin_bit"] is compiler_begin
        assert calls[-1][2]["end_bit"] is compiler_end

        before = len(calls)
        with pytest.raises(
            TypeError,
            match=r"cuda\.coop\.radix_sort_keys requires a fixed-size ThreadData",
        ):
            root_api.radix_sort_keys(block, 3)
        with pytest.raises(
            TypeError,
            match=(
                r"cuda\.coop\.radix_sort_keys common V1 supports key dtypes "
                r"int32, uint32, int64, uint64"
            ),
        ):
            root_api.radix_sort_keys(block, _Payload(1, 2, dtype=float))
        with pytest.raises(TypeError, match="descending must be a compile-time bool"):
            root_api.radix_sort_keys(block, payload, descending=1)
        for name, kwargs in (
            ("begin_bit", {"begin_bit": True}),
            ("begin_bit", {"begin_bit": 1.5}),
            ("begin_bit", {"begin_bit": object()}),
            ("end_bit", {"end_bit": False}),
            ("end_bit", {"end_bit": "32"}),
            ("end_bit", {"end_bit": object()}),
        ):
            with pytest.raises(TypeError, match=rf"{name} must be an int-like scalar"):
                root_api.radix_sort_keys(block, payload, **kwargs)
        for kwargs, message in (
            ({"begin_bit": -1}, "begin_bit must be non-negative"),
            ({"begin_bit": 32}, "begin_bit must be < 32"),
            ({"end_bit": 33}, "end_bit must be <= 32"),
            ({"end_bit": 0}, "end_bit must be positive"),
            (
                {"begin_bit": 8, "end_bit": 8},
                "end_bit must be greater than begin_bit",
            ),
        ):
            with pytest.raises(ValueError, match=message):
                root_api.radix_sort_keys(block, payload, **kwargs)
        assert len(calls) == before

        for group in unsupported_groups:
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.radix_sort_keys does not support group kind",
            ):
                root_api.radix_sort_keys(group, payload)
            assert len(calls) == before

    assert payload.values() == original
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.radix_sort_keys(block, payload)

    profile = COMMON_PROFILE_MATRIX["radix_sort_keys"]
    assert profile["supported_groups"] == ("block",)
    assert profile["mutation_rule"] == "does not mutate inputs"
    assert profile["result_layout"] == "shape-preserving payload"


@pytest.mark.evidence_for(
    "group.radix_sort_pairs", backend="core", evidence="semantics"
)
def test_common_radix_sort_pairs_preserves_correlation_and_validates_both_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, ...]] = []

    def radix_sort_pairs(group, keys, values, /, **kwargs):
        calls.append((group, keys, values, kwargs))
        ordered = sorted(
            zip(keys.values(), values.values()), reverse=kwargs["descending"]
        )
        return tuple(key for key, _ in ordered), tuple(value for _, value in ordered)

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(radix_sort_pairs=radix_sort_pairs),
    )
    block = root_api.this_block()
    warp = root_api.this_warp()
    keys = _Payload(7, 2, 7, -1, dtype=SimpleNamespace(name="int64"))
    values = _Payload(70, 20, 71, -10, dtype=SimpleNamespace(name="uint8"))
    originals = keys.values(), values.values()

    with root_api._compiler_scope("cuda.coop.testing"):
        assert root_api.radix_sort_pairs(
            block,
            keys,
            values,
            descending=True,
            begin_bit=4,
            end_bit=48,
        ) == ((7, 7, 2, -1), (71, 70, 20, -10))
        before = len(calls)
        with pytest.raises(ValueError, match="matching items_per_thread"):
            root_api.radix_sort_pairs(block, keys, _Payload(1, dtype=int))
        with pytest.raises(TypeError, match="supports value dtypes"):
            root_api.radix_sort_pairs(
                block,
                keys,
                _Payload(1j, 2j, 3j, 4j, dtype=complex),
            )
        with pytest.raises(NotImplementedError, match="does not support group kind"):
            root_api.radix_sort_pairs(warp, keys, values)
        assert len(calls) == before

    assert (keys.values(), values.values()) == originals
    assert COMMON_PROFILE_MATRIX["radix_sort_pairs"]["result_layout"] == (
        "correlated shape-preserving key/value payloads"
    )
