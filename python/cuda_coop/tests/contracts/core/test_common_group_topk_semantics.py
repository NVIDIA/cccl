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


@pytest.mark.evidence_for("group.topk_max_keys", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.topk_min_keys", backend="core", evidence="semantics")
def test_common_topk_owns_complete_block_integer_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, tuple[int, ...], int, dict[str, object]]] = []

    def select(operation: str, group, keys, k, /, **kwargs):
        calls.append((operation, group.kind, keys.values(), k, kwargs))
        reverse = operation == "topk_max_keys"
        selected = sorted(keys.values(), reverse=reverse)
        return _Payload(*selected, dtype=keys.dtype)

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(
            topk_max_keys=lambda group, keys, k, /, **kwargs: select(
                "topk_max_keys", group, keys, k, **kwargs
            ),
            topk_min_keys=lambda group, keys, k, /, **kwargs: select(
                "topk_min_keys", group, keys, k, **kwargs
            ),
        ),
    )

    block = root_api.ThreadGroup(
        kind="block",
        hierarchy=root_api.ThreadHierarchy._resolved(block_dim=64),
    )
    unsupported_groups = (
        root_api.this_thread(),
        root_api.this_warp(),
        root_api.this_warp().group_by(8),
        root_api.this_cluster(),
        root_api.this_grid(),
    )
    multidimensional = root_api.ThreadGroup(
        kind="block",
        hierarchy=root_api.ThreadHierarchy._resolved(block_dim=(32, 2, 1)),
    )
    oversized = root_api.ThreadGroup(
        kind="block",
        hierarchy=root_api.ThreadHierarchy._resolved(block_dim=1025),
    )
    payload = _Payload(7, -3, 7, 2)
    original = payload.values()
    storage = object()

    with root_api._compiler_scope("cuda.coop.testing"):
        largest = root_api.topk_max_keys(block, payload, 2)
        smallest = root_api.topk_min_keys(
            block,
            payload,
            2,
            valid_items=4,
            begin_bit=8,
            end_bit=24,
            temp_storage=storage,
        )

        assert largest.values()[:2] == (7, 7)
        assert smallest.values()[:2] == (-3, 2)
        assert largest.dtype is payload.dtype
        assert smallest.dtype is payload.dtype
        assert len(largest) == len(payload)
        assert len(smallest) == len(payload)
        assert calls == [
            (
                "topk_max_keys",
                "block",
                original,
                2,
                {
                    "valid_items": None,
                    "begin_bit": 0,
                    "end_bit": None,
                    "temp_storage": None,
                },
            ),
            (
                "topk_min_keys",
                "block",
                original,
                2,
                {
                    "valid_items": 4,
                    "begin_bit": 8,
                    "end_bit": 24,
                    "temp_storage": storage,
                },
            ),
        ]

        compiler_k = _CompilerInteger()
        compiler_valid = _CompilerInteger()
        compiler_begin = _CompilerInteger()
        compiler_end = _CompilerInteger()
        root_api.topk_max_keys(
            block,
            payload,
            compiler_k,
            valid_items=compiler_valid,
            begin_bit=compiler_begin,
            end_bit=compiler_end,
        )
        assert calls[-1][3] is compiler_k
        assert calls[-1][4] == {
            "valid_items": compiler_valid,
            "begin_bit": compiler_begin,
            "end_bit": compiler_end,
            "temp_storage": None,
        }

        before = len(calls)
        with pytest.raises(
            TypeError,
            match=r"cuda\.coop\.topk_max_keys requires a fixed-size ThreadData",
        ):
            root_api.topk_max_keys(block, 3, 1)
        with pytest.raises(
            TypeError,
            match=(
                r"cuda\.coop\.topk_max_keys common V1 supports key dtypes "
                r"int32, uint32, int64, uint64"
            ),
        ):
            root_api.topk_max_keys(block, _Payload(1, 2, dtype=float), 1)
        with pytest.raises(
            TypeError,
            match=(
                r"cuda\.coop\.topk_max_keys common V1 supports key dtypes "
                r"int32, uint32, int64, uint64"
            ),
        ):
            root_api.topk_max_keys(
                block,
                _Payload(1, 2, dtype=SimpleNamespace(name="uint8")),
                1,
            )
        for name, kwargs in (
            ("k", {"k": True}),
            ("k", {"k": 1.5}),
            ("valid_items", {"valid_items": False}),
            ("valid_items", {"valid_items": "4"}),
            ("begin_bit", {"begin_bit": True}),
            ("begin_bit", {"begin_bit": object()}),
            ("end_bit", {"end_bit": False}),
            ("end_bit", {"end_bit": 1.5}),
        ):
            k = kwargs.pop("k", 1)
            with pytest.raises(TypeError, match=rf"{name} must be an int-like scalar"):
                root_api.topk_max_keys(block, payload, k, **kwargs)
        for kwargs, message in (
            ({"k": 0}, "k must be positive"),
            ({"k": 5, "valid_items": 4}, "k must be <= valid_items"),
            ({"valid_items": 0}, "valid_items must be positive"),
            ({"valid_items": 257}, "valid_items must be <= tile size 256"),
            ({"begin_bit": -1}, "begin_bit must be non-negative"),
            ({"begin_bit": 32}, "begin_bit must be < 32"),
            ({"end_bit": 0}, "end_bit must be positive"),
            ({"end_bit": 33}, "end_bit must be <= 32"),
            (
                {"begin_bit": 8, "end_bit": 8},
                "end_bit must be greater than begin_bit",
            ),
        ):
            k = kwargs.pop("k", 1)
            with pytest.raises(ValueError, match=message):
                root_api.topk_max_keys(block, payload, k, **kwargs)

        for group in unsupported_groups:
            with pytest.raises(
                NotImplementedError,
                match=r"cuda\.coop\.topk_max_keys does not support group kind",
            ):
                root_api.topk_max_keys(group, payload, 1)

        with pytest.raises(ValueError, match="requires a one-dimensional block"):
            root_api.topk_max_keys(multidimensional, payload, 1)

        with pytest.raises(ValueError, match="block thread count must be <= 1024"):
            root_api.topk_max_keys(oversized, payload, 1)
        assert len(calls) == before

    assert payload.values() == original
    with pytest.raises(
        root_api.CoopCompilerContextRequiredError,
        match="requires a Python DSL compiler context",
    ):
        root_api.topk_max_keys(block, payload, 1)

    for operation in ("topk_max_keys", "topk_min_keys"):
        profile = COMMON_PROFILE_MATRIX[operation]
        assert profile["supported_groups"] == ("block",)
        assert profile["mutation_rule"] == "does not mutate inputs"
        assert profile["result_layout"].startswith("shape- and dtype-preserving")


@pytest.mark.evidence_for("group.topk_max_pairs", backend="core", evidence="semantics")
@pytest.mark.evidence_for("group.topk_min_pairs", backend="core", evidence="semantics")
def test_common_topk_pairs_preserve_correlation_and_prefix_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, object, object, object, object]] = []

    def select(operation, group, keys, values, k, /, **kwargs):
        calls.append((operation, group, keys, values, kwargs))
        ordered = sorted(
            zip(keys.values(), values.values()),
            reverse=operation == "topk_max_pairs",
        )
        return (
            _Payload(*(key for key, _ in ordered), dtype=keys.dtype),
            _Payload(*(value for _, value in ordered), dtype=values.dtype),
        )

    monkeypatch.setattr(
        root_api,
        "import_module",
        lambda _name: SimpleNamespace(
            topk_max_pairs=lambda group, keys, values, k, /, **kwargs: select(
                "topk_max_pairs", group, keys, values, k, **kwargs
            ),
            topk_min_pairs=lambda group, keys, values, k, /, **kwargs: select(
                "topk_min_pairs", group, keys, values, k, **kwargs
            ),
        ),
    )
    block = root_api.ThreadGroup(
        kind="block",
        hierarchy=root_api.ThreadHierarchy._resolved(block_dim=32),
    )
    keys = _Payload(9, 1, 9, 4, dtype=SimpleNamespace(name="uint32"))
    values = _Payload(90.0, 10.0, 91.0, 40.0, dtype=float)
    originals = keys.values(), values.values()

    with root_api._compiler_scope("cuda.coop.testing"):
        max_keys, max_values = root_api.topk_max_pairs(block, keys, values, 2)
        min_keys, min_values = root_api.topk_min_pairs(
            block, keys, values, 2, valid_items=4
        )
        assert set(zip(max_keys.values()[:2], max_values.values()[:2])) == {
            (9, 90.0),
            (9, 91.0),
        }
        assert set(zip(min_keys.values()[:2], min_values.values()[:2])) == {
            (1, 10.0),
            (4, 40.0),
        }

        before = len(calls)
        with pytest.raises(ValueError, match="matching items_per_thread"):
            root_api.topk_max_pairs(block, keys, _Payload(1.0, dtype=float), 1)
        with pytest.raises(TypeError, match="supports value dtypes"):
            root_api.topk_min_pairs(
                block,
                keys,
                _Payload(True, False, True, False, dtype=bool),
                1,
            )
        with pytest.raises(ValueError, match="k must be <= valid_items"):
            root_api.topk_max_pairs(block, keys, values, 3, valid_items=2)
        assert len(calls) == before

    assert (keys.values(), values.values()) == originals
    for operation in ("topk_max_pairs", "topk_min_pairs"):
        profile = COMMON_PROFILE_MATRIX[operation]
        assert profile["supported_groups"] == ("block",)
        assert profile["result_layout"].startswith("correlated shape-")
