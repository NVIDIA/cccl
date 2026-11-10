# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from collections.abc import Callable
from dataclasses import dataclass

from .adjacent_shuffle_exchange import (
    _adjacent_difference_callsite,
    _exchange_callsite,
    _resolved_adjacent_difference_call,
    _resolved_exchange_call,
    _resolved_shuffle_call,
    _shuffle_callsite,
)
from .discontinuity import _discontinuity_callsite, _resolved_discontinuity_call
from .histogram_rld import (
    _histogram_callsite,
    _resolved_histogram_call,
    _resolved_run_length_decode_call,
    _run_length_decode_callsite,
)
from .radix import (
    _radix_rank_callsite,
    _radix_sort_callsite,
    _resolved_radix_rank_call,
    _resolved_radix_sort_call,
)
from .reduce_scan import (
    _reduce_callsite,
    _resolved_reduce_call,
    _resolved_scan_call,
    _scan_callsite,
)
from .topk_sort import (
    _merge_sort_callsite,
    _resolved_merge_sort_call,
    _resolved_topk_call,
    _topk_callsite,
)


@dataclass(frozen=True)
class CanonicalFamilyCase:
    name: str
    build_callsite: Callable
    build_resolved_call: Callable


CANONICAL_FAMILY_CASES = (
    CanonicalFamilyCase(
        "sum",
        lambda: _reduce_callsite("sum", 1, "f32"),
        lambda: _resolved_reduce_call("sum", 1, "f32"),
    ),
    CanonicalFamilyCase(
        "reduce",
        lambda: _reduce_callsite("reduce", 1, "i32", binary_op="bit_xor"),
        lambda: _resolved_reduce_call("reduce", 1, "i32", binary_op="bit_xor"),
    ),
    CanonicalFamilyCase(
        "scan",
        lambda: _scan_callsite(1, "i32", initial_present=1, prefix_present=0),
        lambda: _resolved_scan_call(
            1, "i32", initial_present=1, prefix_present=0
        ),
    ),
    CanonicalFamilyCase(
        "histogram",
        lambda: _histogram_callsite("histogram", bins=128),
        lambda: _resolved_histogram_call("histogram", bins=128),
    ),
    CanonicalFamilyCase(
        "histogram_256",
        lambda: _histogram_callsite("histogram_256"),
        lambda: _resolved_histogram_call("histogram_256"),
    ),
    CanonicalFamilyCase(
        "run_length_decode",
        lambda: _run_length_decode_callsite("items"),
        lambda: _resolved_run_length_decode_call("items"),
    ),
    CanonicalFamilyCase(
        "run_length_decode_with_offsets",
        lambda: _run_length_decode_callsite("items_and_offsets"),
        lambda: _resolved_run_length_decode_call("items_and_offsets"),
    ),
    CanonicalFamilyCase(
        "discontinuity_flags",
        lambda: _discontinuity_callsite("flags", "heads"),
        lambda: _resolved_discontinuity_call("flags", "heads"),
    ),
    CanonicalFamilyCase(
        "discontinuity_heads_and_tails",
        lambda: _discontinuity_callsite("heads_and_tails", "heads_and_tails"),
        lambda: _resolved_discontinuity_call(
            "heads_and_tails", "heads_and_tails"
        ),
    ),
    CanonicalFamilyCase(
        "adjacent_difference",
        lambda: _adjacent_difference_callsite("subtract_left"),
        lambda: _resolved_adjacent_difference_call("subtract_left"),
    ),
    CanonicalFamilyCase(
        "shuffle",
        lambda: _shuffle_callsite("rotate", num_items=128, distance=3),
        lambda: _resolved_shuffle_call("rotate", num_items=128, distance=3),
    ),
    CanonicalFamilyCase(
        "exchange",
        lambda: _exchange_callsite("blocked_to_striped"),
        lambda: _resolved_exchange_call("blocked_to_striped"),
    ),
    CanonicalFamilyCase(
        "radix_rank",
        _radix_rank_callsite,
        _resolved_radix_rank_call,
    ),
    CanonicalFamilyCase(
        "merge_sort_keys",
        lambda: _merge_sort_callsite("keys", 1, "i32", None, False, True),
        lambda: _resolved_merge_sort_call("keys", 1, "i32", None, False, True),
    ),
    CanonicalFamilyCase(
        "merge_sort_pairs",
        lambda: _merge_sort_callsite("pairs", 1, "i32", "f32", False, True),
        lambda: _resolved_merge_sort_call(
            "pairs", 1, "i32", "f32", False, True
        ),
    ),
    CanonicalFamilyCase(
        "radix_sort_keys",
        lambda: _radix_sort_callsite("keys", 1, "i32", None),
        lambda: _resolved_radix_sort_call("keys", 1, "i32", None),
    ),
    CanonicalFamilyCase(
        "radix_sort_pairs",
        lambda: _radix_sort_callsite("pairs", 1, "i32", "f32"),
        lambda: _resolved_radix_sort_call("pairs", 1, "i32", "f32"),
    ),
    CanonicalFamilyCase(
        "topk_keys",
        lambda: _topk_callsite("keys", 1, "i32", None, True, True),
        lambda: _resolved_topk_call("keys", 1, "i32", None, True, True),
    ),
    CanonicalFamilyCase(
        "topk_pairs",
        lambda: _topk_callsite("pairs", 1, "i32", "f32", False, False),
        lambda: _resolved_topk_call("pairs", 1, "i32", "f32", False, False),
    ),
)
