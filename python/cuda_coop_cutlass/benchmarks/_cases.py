# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Deterministic streaming workloads and independent NumPy references."""

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    operation: str
    dtype: str = "int32"
    group: str = "block"
    width: int = 32
    threads: int = 128
    items: int = 4
    options: dict = field(default_factory=dict)

    @property
    def group_size(self):
        return self.threads if self.group == "block" else self.width

    @property
    def groups_per_block(self):
        return self.threads // self.group_size


def core_cases():
    result = []

    def add(operation, *, suffix="", **kwargs):
        case = BenchmarkCase("", operation, **kwargs)
        name = (
            f"{case.group}{case.group_size}_{operation}_"
            f"{case.dtype}_i{case.items}"
        )
        if suffix:
            name += f"_{suffix}"
        result.append(BenchmarkCase(name, operation, **kwargs))

    for operation in ("load", "store"):
        for algorithm in ("direct", "transpose"):
            add(operation, suffix=algorithm, options={"algorithm": algorithm})
        add(
            operation,
            group="warp",
            suffix="transpose",
            options={"algorithm": "transpose"},
        )
        add(
            operation,
            group="warp",
            width=8,
            suffix="direct",
            options={"algorithm": "direct"},
        )
    for dtype in ("int32", "float32"):
        add("sum", dtype=dtype, options={"algorithm": "raking"})
        add("sum", dtype=dtype, group="warp", items=1)
    add("sum", group="warp", width=8, items=1)
    add(
        "reduce",
        suffix="min",
        options={"operator": "min", "algorithm": "raking"},
    )
    for operation in (
        "inclusive_sum",
        "exclusive_sum",
        "inclusive_scan",
        "exclusive_scan",
    ):
        options = {"algorithm": "raking"}
        if operation.endswith("scan"):
            options["operator"] = "max"
        add(operation, options=options)
    for operation in ("inclusive_sum", "exclusive_sum"):
        add(operation, group="warp", items=1)
    add(
        "inclusive_scan",
        group="warp",
        width=8,
        items=1,
        suffix="max",
        options={"operator": "max"},
    )
    for layout in ("blocked", "striped"):
        add(
            "reduce_batched",
            group="warp",
            items=64,
            suffix=layout,
            options={"output_layout": layout},
        )
    for group, width in (("block", 32), ("warp", 32), ("warp", 8)):
        for mode in ("blocked_to_striped", "striped_to_blocked"):
            add(
                "exchange",
                group=group,
                width=width,
                suffix=mode,
                options={"mode": mode},
            )
    for mode in ("up", "down", "offset", "rotate"):
        add(
            "shuffle",
            items=4 if mode in ("up", "down") else 1,
            suffix=mode,
            options={"mode": mode},
        )
    for group, width in (("block", 32), ("warp", 32), ("warp", 8)):
        for operation in ("merge_sort_keys", "merge_sort_pairs"):
            add(
                operation,
                group=group,
                width=width,
                options={"descending": operation.endswith("pairs")},
            )
    return result


def all_cases():
    from _advanced import cases

    result = core_cases() + cases()
    names = [case.name for case in result]
    if len(set(names)) != len(names):
        raise ValueError("Benchmark case names must be unique")
    return result


_CORE_OPERATIONS = {case.operation for case in core_cases()}


def make_arrays(case, blocks):
    if case.operation not in _CORE_OPERATIONS:
        from _advanced import make_arrays as advanced_arrays

        return advanced_arrays(case, blocks)
    size = blocks * case.threads * case.items
    source = ((np.arange(size, dtype=np.int64) * 13 + 5) % 17 - 8).astype(
        case.dtype
    )
    group_size = case.group_size
    tile = group_size * case.items
    groups = blocks * case.groups_per_block
    values = source.reshape(groups, tile)
    aux = np.zeros(1, dtype=np.int32)
    out_aux = np.zeros(1, dtype=np.int32)
    operation = case.operation
    if operation in ("load", "store"):
        expected = source.copy()
    elif operation in ("sum", "reduce"):
        expected = (
            values.sum(axis=1, dtype=source.dtype)
            if operation == "sum"
            else values.min(axis=1)
        )
    elif operation in (
        "inclusive_sum",
        "exclusive_sum",
        "inclusive_scan",
        "exclusive_scan",
    ):
        inclusive = (
            np.add.accumulate(values, axis=1, dtype=source.dtype)
            if operation.endswith("sum")
            else np.maximum.accumulate(values, axis=1)
        )
        if operation.startswith("exclusive"):
            initial = 0 if operation.endswith("sum") else np.iinfo(np.int32).min
            expected = np.concatenate(
                (
                    np.full((groups, 1), initial, dtype=source.dtype),
                    inclusive[:, :-1],
                ),
                axis=1,
            )
        else:
            expected = inclusive
    elif operation == "reduce_batched":
        expected = values.reshape(groups, group_size, case.items).sum(
            axis=1, dtype=source.dtype
        )
    elif operation == "exchange":
        if case.options["mode"] == "blocked_to_striped":
            expected = values.reshape(groups, case.items, group_size).transpose(
                0, 2, 1
            )
        else:
            expected = values.reshape(groups, group_size, case.items).transpose(
                0, 2, 1
            )
    elif operation == "shuffle":
        expected = np.zeros_like(values)
        mode = case.options["mode"]
        if mode == "up":
            expected[:, 1:] = values[:, :-1]
        elif mode == "down":
            expected[:, :-1] = values[:, 1:]
        elif mode == "offset":
            expected[:, :-3] = values[:, 3:]
        else:
            expected = np.roll(values, -3, axis=1)
    else:
        # An odd multiplier permutes our power-of-two tiles, making pair
        # association exact without relying on unspecified ordering of ties.
        source = np.tile((np.arange(tile) * 17 + 3) % tile, groups).astype(
            case.dtype
        )
        aux = np.arange(size, dtype=np.int32)
        order = np.argsort(source.reshape(groups, tile), axis=1)
        if case.options["descending"]:
            order = order[:, ::-1]
        expected = np.take_along_axis(
            source.reshape(groups, tile), order, axis=1
        )
        if operation == "merge_sort_pairs":
            out_aux = np.take_along_axis(
                aux.reshape(groups, tile), order, axis=1
            ).ravel()
    return {"input": source, "input_aux": aux}, {
        "output": np.ascontiguousarray(expected).ravel(),
        "output_aux": out_aux,
    }


def assert_result(case, observed, expected):
    if case.operation not in _CORE_OPERATIONS:
        from _advanced import assert_result as advanced_assert

        return advanced_assert(case, observed, expected)
    for key in expected:
        np.testing.assert_array_equal(
            observed[key], expected[key], err_msg=f"{case.name}: {key}"
        )
