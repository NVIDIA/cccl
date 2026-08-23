# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral public API records shared by frontend-specific tests.

These records deliberately do not import a frontend implementation. Each backend
proves its own surface against the same semantic obligation, so a test module never
needs another optional backend merely to state the expected contract.
"""

GROUP_CONSTRUCTOR_SIGNATURES = {
    "this_thread": "() -> 'ThreadGroup'",
    "this_warp": "() -> 'ThreadGroup'",
    "this_block": "() -> 'ThreadGroup'",
    "this_cluster": "() -> 'ThreadGroup'",
    "this_grid": "() -> 'ThreadGroup'",
}

GROUP_METHOD_SIGNATURES = {
    "rank": "(self, level: 'str' = 'thread') -> 'Any'",
    "count": "(self, level: 'str' = 'thread') -> 'Any'",
    "rank_as": ("(self, dtype: 'Any' = None, level: 'str' = 'thread') -> 'Any'"),
    "count_as": ("(self, dtype: 'Any' = None, level: 'str' = 'thread') -> 'Any'"),
    "sync": "(self) -> 'None'",
    "sync_aligned": "(self) -> 'None'",
    "group_by": ("(self, count: 'int', *, exhaustive: 'bool' = True) -> 'ThreadGroup'"),
    "is_member": "(self) -> 'Any'",
}

PORTABLE_GROUP_PRIMITIVE_SIGNATURES = {
    "load": (
        "(group: 'ThreadGroup', source: 'Any', output: 'ThreadDataLike[Any]', /, *, "
        "algorithm: 'Any' = 'direct', valid_items: 'Any' = None, "
        "oob_default: 'Any' = None, offset: 'Any' = None, "
        "temp_storage: 'Any' = None) -> 'ThreadDataLike[Any]'"
    ),
    "store": (
        "(group: 'ThreadGroup', destination: 'Any', value: 'Any', /, *, "
        "algorithm: 'Any' = 'direct', valid_items: 'Any' = None, "
        "offset: 'Any' = None, temp_storage: 'Any' = None) -> 'None'"
    ),
    "reduce": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "binary_op: 'Any' = None, broadcast: 'bool' = True, "
        "valid_items: 'Any' = None, algorithm: 'Any' = None) -> 'Any'"
    ),
    "sum": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "broadcast: 'bool' = True, valid_items: 'Any' = None, "
        "algorithm: 'Any' = None) -> 'Any'"
    ),
    "scan": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "mode: 'str' = 'exclusive', scan_op: 'Any' = None, "
        "initial_value: 'Any' = None, algorithm: 'Any' = None, "
        "temp_storage: 'Any' = None) -> 'Any'"
    ),
    "exclusive_sum": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "algorithm: 'Any' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "inclusive_sum": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "algorithm: 'Any' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "exclusive_scan": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "scan_op: 'Any' = None, initial_value: 'Any' = None, "
        "algorithm: 'Any' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "inclusive_scan": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "scan_op: 'Any' = None, algorithm: 'Any' = None, "
        "temp_storage: 'Any' = None) -> 'Any'"
    ),
    "exchange": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "mode: 'str' = 'striped_to_blocked') -> 'Any'"
    ),
    "adjacent_difference": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "direction: 'Any' = <BlockAdjacentDifferenceDirection.LEFT: 'left'>, "
        "valid_items: 'Any' = None, tile_predecessor_item: 'Any' = None, "
        "tile_successor_item: 'Any' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "discontinuity": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "mode: 'Any' = <BlockDiscontinuityMode.HEADS: 'heads'>, "
        "tile_predecessor_item: 'Any' = None, "
        "tile_successor_item: 'Any' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "shuffle": (
        "(group: 'ThreadGroup', value: 'Any', /, *, "
        "mode: 'Any' = <BlockShuffleMode.DOWN: 'down'>, "
        "distance: 'Any' = 1) -> 'Any'"
    ),
    "merge_sort_keys": (
        "(group: 'ThreadGroup', keys: 'Any', /, *, descending: 'bool' = False, "
        "valid_items: 'Any' = None, oob_default: 'Any' = None, "
        "temp_storage: 'Any' = None) -> 'Any'"
    ),
    "merge_sort_pairs": (
        "(group: 'ThreadGroup', keys: 'Any', values: 'Any', /, *, "
        "descending: 'bool' = False, valid_items: 'Any' = None, "
        "oob_default: 'Any' = None, temp_storage: 'Any' = None) "
        "-> 'tuple[Any, Any]'"
    ),
    "radix_sort_keys": (
        "(group: 'ThreadGroup', keys: 'Any', /, *, begin_bit: 'Any' = 0, "
        "end_bit: 'Any | None' = None, descending: 'bool' = False, "
        "temp_storage: 'Any' = None) -> 'Any'"
    ),
    "radix_sort_pairs": (
        "(group: 'ThreadGroup', keys: 'Any', values: 'Any', /, *, "
        "begin_bit: 'Any' = 0, end_bit: 'Any | None' = None, "
        "descending: 'bool' = False, temp_storage: 'Any' = None) "
        "-> 'tuple[Any, Any]'"
    ),
    "radix_rank": (
        "(group: 'ThreadGroup', keys: 'Any', /, *, begin_bit: 'Any' = 0, "
        "end_bit: 'Any | None' = None, radix_bits: 'Any | None' = None, "
        "descending: 'bool' = False) -> 'Any'"
    ),
    "histogram": (
        "(group: 'ThreadGroup', samples: 'Any', /, *, bins: 'Any', "
        "bins_per_thread: 'Any' = 1, counter_dtype: 'Any' = None, "
        "algorithm: 'Any' = 'atomic') -> 'Any'"
    ),
    "run_length_decode": (
        "(group: 'ThreadGroup', run_values: 'Any', run_lengths: 'Any', /, *, "
        "decoded_items_per_thread: 'Any', decoded_window_offset: 'Any' = 0) "
        "-> 'Any'"
    ),
    "topk_max_keys": (
        "(group: 'ThreadGroup', keys: 'Any', k: 'Any', /, *, "
        "valid_items: 'Any' = None, begin_bit: 'Any' = 0, "
        "end_bit: 'Any | None' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "topk_max_pairs": (
        "(group: 'ThreadGroup', keys: 'Any', values: 'Any', k: 'Any', /, *, "
        "valid_items: 'Any' = None, begin_bit: 'Any' = 0, "
        "end_bit: 'Any | None' = None, temp_storage: 'Any' = None) "
        "-> 'tuple[Any, Any]'"
    ),
    "topk_min_keys": (
        "(group: 'ThreadGroup', keys: 'Any', k: 'Any', /, *, "
        "valid_items: 'Any' = None, begin_bit: 'Any' = 0, "
        "end_bit: 'Any | None' = None, temp_storage: 'Any' = None) -> 'Any'"
    ),
    "topk_min_pairs": (
        "(group: 'ThreadGroup', keys: 'Any', values: 'Any', k: 'Any', /, *, "
        "valid_items: 'Any' = None, begin_bit: 'Any' = 0, "
        "end_bit: 'Any | None' = None, temp_storage: 'Any' = None) "
        "-> 'tuple[Any, Any]'"
    ),
}

REQUIRED_PARAMETER = "<required>"

PORTABLE_GROUP_PRIMITIVE_POSITIONALS = {
    "load": ("group", "source", "output"),
    "store": ("group", "destination", "value"),
    "reduce": ("group", "value"),
    "sum": ("group", "value"),
    "scan": ("group", "value"),
    "exclusive_sum": ("group", "value"),
    "inclusive_sum": ("group", "value"),
    "exclusive_scan": ("group", "value"),
    "inclusive_scan": ("group", "value"),
    "exchange": ("group", "value"),
    "adjacent_difference": ("group", "value"),
    "discontinuity": ("group", "value"),
    "shuffle": ("group", "value"),
    "merge_sort_keys": ("group", "keys"),
    "merge_sort_pairs": ("group", "keys", "values"),
    "radix_sort_keys": ("group", "keys"),
    "radix_sort_pairs": ("group", "keys", "values"),
    "radix_rank": ("group", "keys"),
    "histogram": ("group", "samples"),
    "run_length_decode": ("group", "run_values", "run_lengths"),
    "topk_max_keys": ("group", "keys", "k"),
    "topk_max_pairs": ("group", "keys", "values", "k"),
    "topk_min_keys": ("group", "keys", "k"),
    "topk_min_pairs": ("group", "keys", "values", "k"),
}

PORTABLE_GROUP_PRIMITIVE_KEYWORDS = {
    "load": ("algorithm", "valid_items", "oob_default", "offset", "temp_storage"),
    "store": ("algorithm", "valid_items", "offset", "temp_storage"),
    "reduce": ("binary_op", "broadcast", "valid_items", "algorithm"),
    "sum": ("broadcast", "valid_items", "algorithm"),
    "scan": ("mode", "scan_op", "initial_value", "algorithm", "temp_storage"),
    "exclusive_sum": ("algorithm", "temp_storage"),
    "inclusive_sum": ("algorithm", "temp_storage"),
    "exclusive_scan": ("scan_op", "initial_value", "algorithm", "temp_storage"),
    "inclusive_scan": ("scan_op", "algorithm", "temp_storage"),
    "exchange": ("mode",),
    "adjacent_difference": (
        "direction",
        "valid_items",
        "tile_predecessor_item",
        "tile_successor_item",
        "temp_storage",
    ),
    "discontinuity": (
        "mode",
        "tile_predecessor_item",
        "tile_successor_item",
        "temp_storage",
    ),
    "shuffle": ("mode", "distance"),
    "merge_sort_keys": ("descending", "valid_items", "oob_default", "temp_storage"),
    "merge_sort_pairs": (
        "descending",
        "valid_items",
        "oob_default",
        "temp_storage",
    ),
    "radix_sort_keys": ("begin_bit", "end_bit", "descending", "temp_storage"),
    "radix_sort_pairs": (
        "begin_bit",
        "end_bit",
        "descending",
        "temp_storage",
    ),
    "radix_rank": ("begin_bit", "end_bit", "radix_bits", "descending"),
    "histogram": ("bins", "bins_per_thread", "counter_dtype", "algorithm"),
    "run_length_decode": ("decoded_items_per_thread", "decoded_window_offset"),
    "topk_max_keys": ("valid_items", "begin_bit", "end_bit", "temp_storage"),
    "topk_max_pairs": ("valid_items", "begin_bit", "end_bit", "temp_storage"),
    "topk_min_keys": ("valid_items", "begin_bit", "end_bit", "temp_storage"),
    "topk_min_pairs": ("valid_items", "begin_bit", "end_bit", "temp_storage"),
}

PORTABLE_GROUP_PRIMITIVE_DEFAULTS = {
    "load": ("direct", None, None, None, None),
    "store": ("direct", None, None, None),
    "reduce": (None, True, None, None),
    "sum": (True, None, None),
    "scan": ("exclusive", None, None, None, None),
    "exclusive_sum": (None, None),
    "inclusive_sum": (None, None),
    "exclusive_scan": (None, None, None, None),
    "inclusive_scan": (None, None, None),
    "exchange": ("striped_to_blocked",),
    "adjacent_difference": ("left", None, None, None, None),
    "discontinuity": ("heads", None, None, None),
    "shuffle": ("down", 1),
    "merge_sort_keys": (False, None, None, None),
    "merge_sort_pairs": (False, None, None, None),
    "radix_sort_keys": (0, None, False, None),
    "radix_sort_pairs": (0, None, False, None),
    "radix_rank": (0, None, None, False),
    "histogram": (REQUIRED_PARAMETER, 1, None, "atomic"),
    "run_length_decode": (REQUIRED_PARAMETER, 0),
    "topk_max_keys": (None, 0, None, None),
    "topk_max_pairs": (None, 0, None, None),
    "topk_min_keys": (None, 0, None, None),
    "topk_min_pairs": (None, 0, None, None),
}

QUALIFIED_GROUP_PRIMITIVE_SUFFIXES = {
    "cutlass": {
        "load": (),
        "store": (),
        "reduce": (),
        "sum": (),
        "scan": ("valid_items", "aggregate_output"),
        "exclusive_sum": ("valid_items", "aggregate_output"),
        "inclusive_sum": ("valid_items", "aggregate_output"),
        "exclusive_scan": ("valid_items", "aggregate_output"),
        "inclusive_scan": ("valid_items", "aggregate_output"),
        "exchange": (
            "ranks",
            "valid_flags",
            ("warp_time_slicing", False),
        ),
        "adjacent_difference": ("difference_op",),
        "discontinuity": ("flag_op",),
        "shuffle": ("block_prefix", "block_suffix"),
        "merge_sort_keys": ("compare_op",),
        "merge_sort_pairs": ("compare_op",),
        "radix_sort_keys": (),
        "radix_sort_pairs": (),
        "radix_rank": ("exclusive_digit_prefix",),
        "histogram": (),
        "run_length_decode": (
            "relative_offsets",
            "total_decoded_size",
            "decoded_offset_dtype",
        ),
        "topk_max_keys": (),
        "topk_max_pairs": (),
        "topk_min_keys": (),
        "topk_min_pairs": (),
    },
    "numba_mlir": {
        "scan": ("valid_items", "aggregate_output"),
        "exclusive_sum": ("valid_items", "aggregate_output"),
        "inclusive_sum": ("valid_items", "aggregate_output"),
        "exclusive_scan": ("valid_items", "aggregate_output"),
        "inclusive_scan": ("valid_items", "aggregate_output"),
        "exchange": (
            "ranks",
            "valid_flags",
            ("warp_time_slicing", False),
        ),
        "adjacent_difference": ("difference_op",),
        "discontinuity": ("flag_op",),
        "shuffle": ("block_prefix", "block_suffix"),
        "merge_sort_keys": ("compare_op",),
        "merge_sort_pairs": ("compare_op",),
        "radix_sort_keys": (("blocked_to_striped", False),),
        "radix_sort_pairs": (("blocked_to_striped", False),),
        "radix_rank": ("exclusive_digit_prefix",),
        "run_length_decode": (
            "relative_offsets",
            "total_decoded_size",
            "decoded_offset_dtype",
        ),
    },
}


def qualified_group_primitive_suffix_contract(
    suffix: tuple[str | tuple[str, object], ...],
) -> tuple[tuple[str, object], ...]:
    """Normalize qualified-only parameter names and their defaults."""

    return tuple((item, None) if isinstance(item, str) else item for item in suffix)


def portable_group_primitive_parameter_contract(
    name: str,
    *,
    suffix: tuple[str | tuple[str, object], ...] = (),
) -> tuple[tuple[str, str, object], ...]:
    """Return the canonical runtime parameter contract for one v1 operation."""

    return (
        *(
            (parameter, "POSITIONAL_ONLY", REQUIRED_PARAMETER)
            for parameter in PORTABLE_GROUP_PRIMITIVE_POSITIONALS[name]
        ),
        *(
            (parameter, "KEYWORD_ONLY", default)
            for parameter, default in zip(
                PORTABLE_GROUP_PRIMITIVE_KEYWORDS[name],
                PORTABLE_GROUP_PRIMITIVE_DEFAULTS[name],
            )
        ),
        *(
            (parameter, "KEYWORD_ONLY", default)
            for parameter, default in qualified_group_primitive_suffix_contract(suffix)
        ),
    )


_BLOCK_AND_WARP_GROUPS = (
    "block",
    "physical_warp",
    "threads_within_warp",
)
_REDUCTION_GROUPS = (
    "thread",
    "physical_warp",
    "threads_within_warp",
    "block",
    "cluster",
)
_BLOCK_ONLY = ("block",)
_COMMON_PROFILE_GROUPS = {
    "load": _BLOCK_AND_WARP_GROUPS,
    "store": _BLOCK_AND_WARP_GROUPS,
    "reduce": _REDUCTION_GROUPS,
    "sum": _REDUCTION_GROUPS,
    "scan": _BLOCK_AND_WARP_GROUPS,
    "exclusive_sum": _BLOCK_AND_WARP_GROUPS,
    "inclusive_sum": _BLOCK_AND_WARP_GROUPS,
    "exclusive_scan": _BLOCK_AND_WARP_GROUPS,
    "inclusive_scan": _BLOCK_AND_WARP_GROUPS,
    "exchange": _BLOCK_AND_WARP_GROUPS,
    "adjacent_difference": _BLOCK_ONLY,
    "discontinuity": _BLOCK_ONLY,
    "shuffle": _BLOCK_ONLY,
    "merge_sort_keys": _BLOCK_AND_WARP_GROUPS,
    "merge_sort_pairs": _BLOCK_AND_WARP_GROUPS,
    "radix_sort_keys": _BLOCK_ONLY,
    "radix_sort_pairs": _BLOCK_ONLY,
    "radix_rank": _BLOCK_ONLY,
    "histogram": _BLOCK_ONLY,
    "run_length_decode": _BLOCK_ONLY,
    "topk_max_keys": _BLOCK_ONLY,
    "topk_max_pairs": _BLOCK_ONLY,
    "topk_min_keys": _BLOCK_ONLY,
    "topk_min_pairs": _BLOCK_ONLY,
}
_COMMON_PROFILE_RESULTS = {
    "load": "populated output payload",
    "store": "None",
    "reduce": "one scalar with documented ownership",
    "sum": "one scalar with documented ownership",
    "scan": (
        "shape-preserving payload with every position defined; exclusive sum "
        "uses zero or an explicit initial_value, while other exclusive operators "
        "require initial_value; inclusive scans reject initial_value"
    ),
    "exclusive_sum": (
        "shape-preserving payload with the first flattened position equal to zero"
    ),
    "inclusive_sum": "shape-preserving payload with every position defined",
    "exclusive_scan": (
        "shape-preserving payload with every position defined; sum uses zero or "
        "an explicit initial_value, while other operators require initial_value"
    ),
    "inclusive_scan": (
        "shape-preserving payload with every position defined and no initial_value"
    ),
    "exchange": "shape-preserving payload",
    "adjacent_difference": "shape-preserving payload",
    "discontinuity": "shape-preserving int32 flags",
    "shuffle": (
        "shape-preserving payload; vacated first or last flattened item undefined"
    ),
    "merge_sort_keys": "shape-preserving payload",
    "merge_sort_pairs": (
        "correlated shape-preserving key/value payloads; only the valid sorted "
        "prefix is defined for a partial tile"
    ),
    "radix_sort_keys": "shape-preserving payload",
    "radix_sort_pairs": "correlated shape-preserving key/value payloads",
    "radix_rank": "shape-preserving int32 ranks",
    "histogram": (
        "striped counters by rank plus i times group size; out-of-range slots are zero"
    ),
    "run_length_decode": (
        "decoded_items_per_thread values per member in blocked window order; "
        "out-of-range positions are zero"
    ),
    "topk_max_keys": (
        "shape- and dtype-preserving payload; exactly the first k blocked "
        "positions are defined and unordered; remaining positions are unspecified"
    ),
    "topk_max_pairs": (
        "correlated shape- and dtype-preserving key/value payloads; exactly the "
        "first k blocked positions are defined and unordered"
    ),
    "topk_min_keys": (
        "shape- and dtype-preserving payload; exactly the first k blocked "
        "positions are defined and unordered; remaining positions are unspecified"
    ),
    "topk_min_pairs": (
        "correlated shape- and dtype-preserving key/value payloads; exactly the "
        "first k blocked positions are defined and unordered"
    ),
}
_COMMON_PROFILE_MUTATION = {
    "load": "populates and returns output",
    "store": "writes destination and returns None",
    **{
        name: "does not mutate inputs"
        for name in PORTABLE_GROUP_PRIMITIVE_SIGNATURES
        if name not in {"load", "store"}
    },
}
_COMMON_PROFILE_PRECONDITIONS = {
    "histogram": (
        "every sample satisfies 0 <= sample < bins; violating this CUB "
        "precondition is undefined behavior",
    ),
}

COMMON_PROFILE_DTYPE_FAMILIES = {
    "numeric": {
        "canonical_dtypes": (
            "uint8",
            "int32",
            "uint32",
            "int64",
            "uint64",
            "float32",
            "float64",
        ),
        "ordinary_aliases": (("int", "int32"), ("float", "float32")),
    },
    "integer_value": {
        "canonical_dtypes": ("uint8", "int32", "uint32", "int64", "uint64"),
        "ordinary_aliases": (("int", "int32"),),
    },
    "integer_key": {
        "canonical_dtypes": ("int32", "uint32", "int64", "uint64"),
        "ordinary_aliases": (("int", "int32"),),
    },
}

_COMMON_PROFILE_DTYPE_CONTRACTS = {
    "load": {"source": "numeric", "output": "numeric"},
    "store": {"destination": "numeric", "value": "numeric"},
    **{
        operation: {"value": "numeric"}
        for operation in (
            "reduce",
            "sum",
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
            "exchange",
            "adjacent_difference",
            "discontinuity",
            "shuffle",
        )
    },
    "merge_sort_keys": {"keys": "integer_key"},
    "merge_sort_pairs": {"keys": "integer_key", "values": "numeric"},
    "radix_sort_keys": {"keys": "integer_key"},
    "radix_sort_pairs": {"keys": "integer_key", "values": "numeric"},
    "radix_rank": {"keys": "integer_key"},
    "histogram": {"samples": "integer_value", "counters": "integer_key"},
    "run_length_decode": {
        "run_values": "integer_value",
        "run_lengths": "integer_key",
    },
    "topk_max_keys": {"keys": "integer_key"},
    "topk_max_pairs": {"keys": "integer_key", "values": "numeric"},
    "topk_min_keys": {"keys": "integer_key"},
    "topk_min_pairs": {"keys": "integer_key", "values": "numeric"},
}

_COMMON_PROFILE_REQUIRED_EVIDENCE = {
    "core": ("api", "semantics"),
    "cutlass": ("lowering", "compile", "runtime", "link"),
    "numba_mlir": ("lowering", "compile", "runtime", "link"),
}
_COMMON_REDUCE_SUM_SCAN_OPERATIONS = (
    "reduce",
    "sum",
    "scan",
    "exclusive_sum",
    "inclusive_sum",
    "exclusive_scan",
    "inclusive_scan",
)
_COMMON_ADJACENT_DISCONTINUITY_OPERATIONS = (
    "adjacent_difference",
    "discontinuity",
)
_COMMON_TOPK_OPERATIONS = (
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
)
_COMMON_DIRECT_EVIDENCE_OPERATIONS = frozenset(PORTABLE_GROUP_PRIMITIVE_SIGNATURES)
_COMMON_PROFILE_REQUIRED_OPERATIONS_BY_ROLE = {
    "core": _COMMON_DIRECT_EVIDENCE_OPERATIONS,
    # Backend evidence becomes required as each backend commit lands its
    # implementation and test lanes.
    "cutlass": frozenset(
        {
            "load",
            "store",
            "reduce",
            "sum",
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }
    ),
    "numba_mlir": frozenset(
        {
            "load",
            "store",
            "reduce",
            "sum",
            "scan",
            "exclusive_sum",
            "inclusive_sum",
            "exclusive_scan",
            "inclusive_scan",
        }
    ),
}

COMMON_PROFILE_ROLE_METADATA = {
    "core": {
        "public_surface": "cuda.coop",
        "provider": "compiler-selected backend API",
        "migration_sources": (
            "tests/contracts/public_api/**/*.py",
            "tests/contracts/core/**/*.py",
        ),
        "collection_paths_by_evidence": {
            "api": ("tests/contracts/public_api/**/*.py",),
            "semantics": ("tests/contracts/core/**/*.py",),
        },
    },
    "cutlass": {
        "public_surface": "cuda.coop.cutlass",
        "provider": "shared group planner and public CUB/CUDAX",
        "migration_sources": (
            "tests/backends/cutlass/**/*.py",
            "tests/providers/cutlass/**/*.py",
            "tests/providers/qualification/cutlass/**/*.py",
        ),
        "collection_paths_by_evidence": {
            "lowering": (
                "tests/backends/cutlass/unit/**/*.py",
                "tests/backends/cutlass/compile/**/*.py",
            ),
            "compile": ("tests/backends/cutlass/compile/**/*.py",),
            "runtime": ("tests/backends/cutlass/runtime/**/*.py",),
            "link": (
                "tests/providers/cutlass/**/*.py",
                "tests/providers/qualification/cutlass/**/*.py",
            ),
        },
    },
    "numba_mlir": {
        "public_surface": "cuda.coop.numba_mlir",
        "provider": "shared group planner and Numba-CUDA-MLIR rewrite",
        "migration_sources": (
            "tests/backends/numba_mlir/**/*.py",
            "tests/providers/qualification/numba_mlir/**/*.py",
        ),
        "collection_paths_by_evidence": {
            "lowering": (
                "tests/backends/numba_mlir/unit/**/*.py",
                "tests/backends/numba_mlir/compile/**/*.py",
            ),
            "compile": ("tests/backends/numba_mlir/compile/**/*.py",),
            "runtime": ("tests/backends/numba_mlir/runtime/**/*.py",),
            "link": ("tests/providers/qualification/numba_mlir/**/*.py",),
        },
    },
}

_COMMON_LOAD_STORE_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": ("tests/contracts/core/test_common_group_memory_semantics.py",),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/compile/test_group_load_store.py",),
        "compile": (
            "tests/backends/cutlass/compile/test_common_group_load_store_compile.py",
        ),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_load_store.py",),
        "link": ("tests/providers/cutlass/test_common_root_load_store_final_link.py",),
    },
    "numba_mlir": {
        "lowering": (
            "tests/backends/numba_mlir/unit/test_common_root_load_store_lowering.py",
        ),
        "compile": (
            "tests/backends/numba_mlir/compile/test_common_root_load_store.py",
        ),
        "runtime": (
            "tests/backends/numba_mlir/runtime/test_common_root_load_store.py",
        ),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_root_load_store_final_link.py",
        ),
    },
}

_COMMON_SUM_SCAN_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": ("tests/contracts/core/test_common_group_sum_scan_semantics.py",),
    },
    "cutlass": {
        "lowering": (
            "tests/backends/cutlass/unit/test_common_root_sum_scan_lowering.py",
        ),
        "compile": ("tests/backends/cutlass/compile/test_common_root_sum_scan.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_sum_scan.py",),
        "link": ("tests/providers/cutlass/test_common_root_sum_scan_final_link.py",),
    },
    "numba_mlir": {
        "lowering": (
            "tests/backends/numba_mlir/unit/test_common_root_sum_scan_lowering.py",
        ),
        "compile": ("tests/backends/numba_mlir/compile/test_common_root_sum_scan.py",),
        "runtime": ("tests/backends/numba_mlir/runtime/test_common_root_sum_scan.py",),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_sum_scan_final_link.py",
        ),
    },
}
_COMMON_SUM_EVIDENCE_COLLECTION_PATHS = {
    **_COMMON_SUM_SCAN_EVIDENCE_COLLECTION_PATHS,
    "cutlass": {
        **_COMMON_SUM_SCAN_EVIDENCE_COLLECTION_PATHS["cutlass"],
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_sum_scan.py",),
    },
    "numba_mlir": {
        **_COMMON_SUM_SCAN_EVIDENCE_COLLECTION_PATHS["numba_mlir"],
        "runtime": ("tests/backends/numba_mlir/runtime/test_common_root_sum_scan.py",),
    },
}
_COMMON_EXCHANGE_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": ("tests/contracts/core/test_common_group_exchange_semantics.py",),
    },
    "cutlass": {
        "lowering": (
            "tests/backends/cutlass/unit/test_common_root_exchange_lowering.py",
        ),
        "compile": ("tests/backends/cutlass/compile/test_common_root_exchange.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_exchange.py",),
        "link": ("tests/providers/cutlass/test_common_root_exchange_final_link.py",),
    },
    "numba_mlir": {
        "lowering": (
            "tests/backends/numba_mlir/unit/test_common_root_exchange_lowering.py",
        ),
        "compile": ("tests/backends/numba_mlir/compile/test_common_root_exchange.py",),
        "runtime": ("tests/backends/numba_mlir/runtime/test_common_root_exchange.py",),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_exchange_final_link.py",
        ),
    },
}
_COMMON_ADJACENT_DISCONTINUITY_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": (
            "tests/contracts/core/test_common_root_adjacent_discontinuity.py",
        ),
    },
    "cutlass": {
        "lowering": (
            "tests/backends/cutlass/unit/test_common_root_adjacent_discontinuity.py",
        ),
        "compile": (
            "tests/backends/cutlass/compile/test_common_root_adjacent_discontinuity.py",
        ),
        "runtime": (
            "tests/backends/cutlass/runtime/test_common_root_adjacent_discontinuity.py",
        ),
        "link": (
            "tests/providers/cutlass/"
            "test_common_root_adjacent_discontinuity_final_link.py",
        ),
    },
    "numba_mlir": {
        "lowering": (
            "tests/backends/numba_mlir/unit/test_common_root_adjacent_discontinuity.py",
        ),
        "compile": (
            "tests/backends/numba_mlir/compile/"
            "test_common_root_adjacent_discontinuity.py",
        ),
        "runtime": (
            "tests/backends/numba_mlir/runtime/"
            "test_common_root_adjacent_discontinuity.py",
        ),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_adjacent_discontinuity_final_link.py",
        ),
    },
}
_COMMON_SHUFFLE_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": ("tests/contracts/core/test_common_group_shuffle_semantics.py",),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/unit/test_common_root_shuffle.py",),
        "compile": ("tests/backends/cutlass/compile/test_common_root_shuffle.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_shuffle.py",),
        "link": ("tests/providers/cutlass/test_common_root_shuffle_final_link.py",),
    },
    "numba_mlir": {
        "lowering": ("tests/backends/numba_mlir/unit/test_common_root_shuffle.py",),
        "compile": ("tests/backends/numba_mlir/compile/test_common_root_shuffle.py",),
        "runtime": ("tests/backends/numba_mlir/runtime/test_common_root_shuffle.py",),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_shuffle_final_link.py",
        ),
    },
}
_COMMON_HISTOGRAM_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": ("tests/contracts/core/test_common_group_histogram_semantics.py",),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/unit/test_common_root_histogram.py",),
        "compile": ("tests/backends/cutlass/compile/test_common_root_histogram.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_histogram.py",),
        "link": ("tests/providers/cutlass/test_common_root_histogram_final_link.py",),
    },
    "numba_mlir": {
        "lowering": ("tests/backends/numba_mlir/unit/test_common_root_histogram.py",),
        "compile": ("tests/backends/numba_mlir/compile/test_common_root_histogram.py",),
        "runtime": ("tests/backends/numba_mlir/runtime/test_common_root_histogram.py",),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_histogram_final_link.py",
        ),
    },
}
_COMMON_MERGE_SORT_KEYS_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": (
            "tests/contracts/core/test_common_group_merge_sort_semantics.py",
        ),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/unit/test_common_root_merge_sort.py",),
        "compile": ("tests/backends/cutlass/compile/test_common_root_merge_sort.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_merge_sort.py",),
        "link": ("tests/providers/cutlass/test_common_root_merge_sort_final_link.py",),
    },
    "numba_mlir": {
        "lowering": ("tests/backends/numba_mlir/unit/test_common_root_merge_sort.py",),
        "compile": (
            "tests/backends/numba_mlir/compile/test_common_root_merge_sort.py",
        ),
        "runtime": (
            "tests/backends/numba_mlir/runtime/test_common_root_merge_sort.py",
        ),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_merge_sort_final_link.py",
        ),
    },
}
_COMMON_RADIX_SORT_KEYS_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": (
            "tests/contracts/core/test_common_group_radix_sort_semantics.py",
        ),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/unit/test_common_root_radix_sort.py",),
        "compile": ("tests/backends/cutlass/compile/test_common_root_radix_sort.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_radix_sort.py",),
        "link": ("tests/providers/cutlass/test_common_root_radix_sort_final_link.py",),
    },
    "numba_mlir": {
        "lowering": ("tests/backends/numba_mlir/unit/test_common_root_radix_sort.py",),
        "compile": (
            "tests/backends/numba_mlir/compile/test_common_root_radix_sort.py",
        ),
        "runtime": (
            "tests/backends/numba_mlir/runtime/test_common_root_radix_sort.py",
        ),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_radix_sort_final_link.py",
        ),
    },
}
_COMMON_RADIX_RANK_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": (
            "tests/contracts/core/test_common_group_radix_rank_semantics.py",
        ),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/unit/test_common_root_radix_rank.py",),
        "compile": ("tests/backends/cutlass/compile/test_common_root_radix_rank.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_radix_rank.py",),
        "link": ("tests/providers/cutlass/test_common_root_radix_rank_final_link.py",),
    },
    "numba_mlir": {
        "lowering": ("tests/backends/numba_mlir/unit/test_common_root_radix_rank.py",),
        "compile": (
            "tests/backends/numba_mlir/compile/test_common_root_radix_rank.py",
        ),
        "runtime": (
            "tests/backends/numba_mlir/runtime/test_common_root_radix_rank.py",
        ),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_radix_rank_final_link.py",
        ),
    },
}
_COMMON_RUN_LENGTH_DECODE_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": (
            "tests/contracts/core/test_common_group_run_length_decode_semantics.py",
        ),
    },
    "cutlass": {
        "lowering": (
            "tests/backends/cutlass/unit/test_common_root_run_length_decode.py",
        ),
        "compile": (
            "tests/backends/cutlass/compile/test_common_root_run_length_decode.py",
        ),
        "runtime": (
            "tests/backends/cutlass/runtime/test_common_root_run_length_decode.py",
        ),
        "link": (
            "tests/providers/cutlass/test_common_root_run_length_decode_final_link.py",
        ),
    },
    "numba_mlir": {
        "lowering": (
            "tests/backends/numba_mlir/unit/test_common_root_run_length_decode.py",
        ),
        "compile": (
            "tests/backends/numba_mlir/compile/test_common_root_run_length_decode.py",
        ),
        "runtime": (
            "tests/backends/numba_mlir/runtime/test_common_root_run_length_decode.py",
        ),
        "link": (
            "tests/providers/qualification/numba_mlir/"
            "test_common_root_run_length_decode_final_link.py",
        ),
    },
}
_COMMON_TOPK_EVIDENCE_COLLECTION_PATHS = {
    "core": {
        "api": ("tests/contracts/public_api/test_root_api.py",),
        "semantics": ("tests/contracts/core/test_common_group_topk_semantics.py",),
    },
    "cutlass": {
        "lowering": ("tests/backends/cutlass/unit/test_common_root_topk.py",),
        "compile": ("tests/backends/cutlass/compile/test_common_root_topk.py",),
        "runtime": ("tests/backends/cutlass/runtime/test_common_root_topk.py",),
        "link": ("tests/providers/cutlass/test_common_root_topk_final_link.py",),
    },
    "numba_mlir": {
        "lowering": ("tests/backends/numba_mlir/unit/test_common_root_topk.py",),
        "compile": ("tests/backends/numba_mlir/compile/test_common_root_topk.py",),
        "runtime": ("tests/backends/numba_mlir/runtime/test_common_root_topk.py",),
        "link": (
            "tests/providers/qualification/numba_mlir/test_root_topk_final_link.py",
        ),
    },
}


def _pair_evidence_collection_paths(
    base: dict[str, dict[str, tuple[str, ...]]],
    *,
    numba_lowering: tuple[str, ...] | None = None,
    cutlass_runtime_extra: tuple[str, ...] = (),
) -> dict[str, dict[str, tuple[str, ...]]]:
    """Return family evidence owners with the shared CUTLASS pair probes."""

    return {
        "core": base["core"],
        "cutlass": {
            "lowering": base["cutlass"]["lowering"],
            "compile": ("tests/backends/cutlass/compile/test_common_root_pairs.py",),
            "runtime": (
                "tests/backends/cutlass/runtime/test_common_profile.py",
                *cutlass_runtime_extra,
            ),
            "link": ("tests/providers/cutlass/test_common_root_pairs_final_link.py",),
        },
        "numba_mlir": {
            **base["numba_mlir"],
            "lowering": numba_lowering or base["numba_mlir"]["lowering"],
        },
    }


_COMMON_MERGE_SORT_PAIRS_EVIDENCE_COLLECTION_PATHS = _pair_evidence_collection_paths(
    _COMMON_MERGE_SORT_KEYS_EVIDENCE_COLLECTION_PATHS,
    numba_lowering=(
        "tests/backends/numba_mlir/unit/test_common_root_merge_sort_lowering.py",
    ),
)
_COMMON_RADIX_SORT_PAIRS_EVIDENCE_COLLECTION_PATHS = _pair_evidence_collection_paths(
    _COMMON_RADIX_SORT_KEYS_EVIDENCE_COLLECTION_PATHS,
    cutlass_runtime_extra=("tests/backends/cutlass/runtime/test_radix.py",),
)
_COMMON_TOPK_PAIRS_EVIDENCE_COLLECTION_PATHS = _pair_evidence_collection_paths(
    _COMMON_TOPK_EVIDENCE_COLLECTION_PATHS
)
_COMMON_OPERATION_EVIDENCE_COLLECTION_PATHS = {
    **{
        operation: _COMMON_LOAD_STORE_EVIDENCE_COLLECTION_PATHS
        for operation in ("load", "store")
    },
    "sum": _COMMON_SUM_EVIDENCE_COLLECTION_PATHS,
    **{
        operation: _COMMON_SUM_SCAN_EVIDENCE_COLLECTION_PATHS
        for operation in _COMMON_REDUCE_SUM_SCAN_OPERATIONS
        if operation != "sum"
    },
    "exchange": _COMMON_EXCHANGE_EVIDENCE_COLLECTION_PATHS,
    **{
        operation: _COMMON_ADJACENT_DISCONTINUITY_EVIDENCE_COLLECTION_PATHS
        for operation in _COMMON_ADJACENT_DISCONTINUITY_OPERATIONS
    },
    "shuffle": _COMMON_SHUFFLE_EVIDENCE_COLLECTION_PATHS,
    "histogram": _COMMON_HISTOGRAM_EVIDENCE_COLLECTION_PATHS,
    "merge_sort_keys": _COMMON_MERGE_SORT_KEYS_EVIDENCE_COLLECTION_PATHS,
    "merge_sort_pairs": _COMMON_MERGE_SORT_PAIRS_EVIDENCE_COLLECTION_PATHS,
    "radix_rank": _COMMON_RADIX_RANK_EVIDENCE_COLLECTION_PATHS,
    "radix_sort_keys": _COMMON_RADIX_SORT_KEYS_EVIDENCE_COLLECTION_PATHS,
    "radix_sort_pairs": _COMMON_RADIX_SORT_PAIRS_EVIDENCE_COLLECTION_PATHS,
    "run_length_decode": _COMMON_RUN_LENGTH_DECODE_EVIDENCE_COLLECTION_PATHS,
    **{
        operation: (
            _COMMON_TOPK_PAIRS_EVIDENCE_COLLECTION_PATHS
            if operation.endswith("_pairs")
            else _COMMON_TOPK_EVIDENCE_COLLECTION_PATHS
        )
        for operation in _COMMON_TOPK_OPERATIONS
    },
}

COMMON_PROFILE_MATRIX = {
    name: {
        "signature": signature,
        "supported_groups": _COMMON_PROFILE_GROUPS[name],
        "result_layout": _COMMON_PROFILE_RESULTS[name],
        "mutation_rule": _COMMON_PROFILE_MUTATION[name],
        "preconditions": _COMMON_PROFILE_PRECONDITIONS.get(name, ()),
        "dtype_contract": dict(_COMMON_PROFILE_DTYPE_CONTRACTS[name]),
        "certified_backends": ("cutlass", "numba_mlir"),
        "required_evidence": dict(_COMMON_PROFILE_REQUIRED_EVIDENCE),
        "evidence_collection_paths": {
            role: dict(lanes)
            for role, lanes in _COMMON_OPERATION_EVIDENCE_COLLECTION_PATHS.get(
                name, {}
            ).items()
        },
        "evidence_enforcement": {
            role: (
                "required"
                if name in _COMMON_PROFILE_REQUIRED_OPERATIONS_BY_ROLE[role]
                else "migration"
            )
            for role in _COMMON_PROFILE_REQUIRED_EVIDENCE
        },
    }
    for name, signature in PORTABLE_GROUP_PRIMITIVE_SIGNATURES.items()
}

PORTABLE_GROUP_FIRST_EXPORTS = (
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    *GROUP_CONSTRUCTOR_SIGNATURES,
    *PORTABLE_GROUP_PRIMITIVE_SIGNATURES,
)

FULL_GROUP_FIRST_EXPORTS = (
    "Hierarchy",
    "ThreadGroup",
    "ThreadHierarchy",
    *GROUP_CONSTRUCTOR_SIGNATURES,
    *PORTABLE_GROUP_PRIMITIVE_SIGNATURES,
)

SCOPED_ROOT_EXPORTS = frozenset(
    {
        "BlockHistogramAlgorithm",
        "BlockLoadAlgorithm",
        "BlockScanAlgorithm",
        "BlockStoreAlgorithm",
        "gpu_dataclass",
        "local",
        "NoAlgorithm",
        "shared",
        "StatefulFunction",
        "TempStorage",
        "ThreadData",
        "WarpLoadAlgorithm",
        "WarpStoreAlgorithm",
        *PORTABLE_GROUP_FIRST_EXPORTS,
    }
)

SCOPED_BLOCK_EXPORTS = frozenset(
    {
        "BlockExchangeType",
        "BlockDiscontinuityType",
        "BlockAdjacentDifferenceType",
        "BlockShuffleType",
        "adjacent_difference",
        "shuffle",
        "exchange",
        "discontinuity",
        "exclusive_scan",
        "exclusive_sum",
        "histogram",
        "inclusive_scan",
        "inclusive_sum",
        "load",
        "merge_sort_keys",
        "merge_sort_pairs",
        "radix_sort_keys",
        "radix_sort_keys_descending",
        "radix_sort_pairs",
        "radix_sort_pairs_descending",
        "topk_max_keys",
        "topk_min_keys",
        "topk_max_pairs",
        "topk_min_pairs",
        "radix_rank",
        "reduce",
        "run_length",
        "scan",
        "store",
        "sum",
        "make_adjacent_difference",
        "make_discontinuity",
        "make_exchange",
        "make_exclusive_scan",
        "make_exclusive_sum",
        "make_histogram",
        "make_inclusive_scan",
        "make_inclusive_sum",
        "make_load",
        "make_merge_sort_keys",
        "make_merge_sort_pairs",
        "make_radix_rank",
        "make_radix_sort_keys",
        "make_radix_sort_keys_descending",
        "make_radix_sort_pairs",
        "make_radix_sort_pairs_descending",
        "make_topk_max_keys",
        "make_topk_min_keys",
        "make_topk_max_pairs",
        "make_topk_min_pairs",
        "make_reduce",
        "make_run_length",
        "make_scan",
        "make_shuffle",
        "make_store",
        "make_sum",
    }
)

SCOPED_WARP_EXPORTS = frozenset(
    {
        "exclusive_scan",
        "exclusive_sum",
        "inclusive_scan",
        "inclusive_sum",
        "reduce",
        "sum",
        "max",
        "min",
        "merge_sort_keys",
        "merge_sort_pairs",
        "load",
        "store",
        "exchange",
        "WarpExchangeType",
        "make_exchange",
        "make_exclusive_scan",
        "make_exclusive_sum",
        "make_inclusive_scan",
        "make_inclusive_sum",
        "make_load",
        "make_merge_sort_keys",
        "make_merge_sort_pairs",
        "make_reduce",
        "make_store",
        "make_sum",
        "make_max",
        "make_min",
    }
)
