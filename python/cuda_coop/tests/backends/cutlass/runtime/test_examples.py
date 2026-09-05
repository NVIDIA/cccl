# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from pathlib import Path

import pytest

from ..support.prims_runtime import _run_prims_example
from ..support.runtime import runtime_pytestmark

pytestmark = [
    *runtime_pytestmark,
    pytest.mark.usefixtures("qualified_cutlass_backend"),
]


def test_portable_root_sum_example_runtime(source_examples):
    from examples.cutlass import portable_root_sum

    assert portable_root_sum.run_example() == sum(
        range(1, portable_root_sum.TILE_ITEMS + 1)
    )


def test_provider_warp_prefix_reduce_example_runtime(source_examples):
    from examples.cutlass import cute_warp_prefix_reduce

    result = cute_warp_prefix_reduce.run_example()

    assert result["warp_totals"] == [528, 2576]
    assert result["prefix_out"][:8] == [0, 1, 3, 6, 10, 15, 21, 28]
    assert result["prefix_out"][32:40] == [0, 65, 131, 198, 266, 335, 405, 476]


def test_provider_thread_group_reduce_example_runtime(source_examples):
    from examples.cutlass import cute_thread_group_reduce

    result = cute_thread_group_reduce.run_example()

    assert result["block_sum"][:4] == [2080, 2080, 2080, 2080]
    assert result["block_items_sum"][:4] == [8256, 8256, 8256, 8256]
    assert result["warp_max"][:4] == [32, 32, 32, 32]
    assert result["warp_max"][32:36] == [64, 64, 64, 64]


def test_provider_thread_group_descriptor_reduce_example_runtime(source_examples):
    from examples.cutlass import cute_thread_group_descriptor_reduce

    result = cute_thread_group_descriptor_reduce.run_example()

    assert result["block_sum"][:4] == [2080, 2080, 2080, 2080]
    assert result["block_items_sum"][:4] == [8256, 8256, 8256, 8256]
    assert result["warp_max"][:4] == [32, 32, 32, 32]
    assert result["warp_max"][32:36] == [64, 64, 64, 64]


def test_provider_thread_hierarchy_reduce_example_runtime(source_examples):
    from examples.cutlass import cute_thread_hierarchy_reduce

    result = cute_thread_hierarchy_reduce.run_example()

    assert result["block_sum"][:4] == [2080, 2080, 2080, 2080]
    assert result["block_items_sum"][:4] == [8256, 8256, 8256, 8256]
    assert result["warp_max"][:4] == [32, 32, 32, 32]
    assert result["warp_max"][32:36] == [64, 64, 64, 64]


def test_provider_thread_group_query_example_runtime(source_examples):
    from examples.cutlass import cute_thread_group_query

    result = cute_thread_group_query.run_example()

    assert result["thread_rank"][:8] == list(range(8))
    assert result["thread_rank"][32:40] == list(range(32, 40))
    assert result["thread_count"][:4] == [64, 64, 64, 64]
    assert result["warp_rank"][:4] == [0, 0, 0, 0]
    assert result["warp_rank"][32:36] == [1, 1, 1, 1]
    assert result["warp_count"][:4] == [2, 2, 2, 2]


def test_provider_legacy_reduce_compare_example_runtime(source_examples):
    from examples.cutlass import cute_legacy_reduce_compare

    result = cute_legacy_reduce_compare.run_example()

    assert result["block_sum"][:4] == [2080, 2080, 2080, 2080]
    assert result["block_items_sum"][:4] == [8256, 8256, 8256, 8256]
    assert result["warp_max"][:4] == [32, 32, 32, 32]
    assert result["warp_max"][32:36] == [64, 64, 64, 64]


def test_prims_vector_rank_merge_example_runtime(tmp_path: Path):
    result = _run_prims_example("prims_vector_rank_merge.py", tmp_path)

    assert result.returncode == 0, result.stderr
    assert "'merge_pairs':" in result.stdout
    assert "'prefix':" in result.stdout
    assert "'ranks':" in result.stdout


def _assert_kmeans_assign_first_query_result(
    result,
    cute_kmeans_assign_topk,
    *,
    expected_k,
    score_only=False,
):
    query = [
        (feature_idx * 7 + feature_idx // 3) % 19
        for feature_idx in range(cute_kmeans_assign_topk.WIDE_FEATURE_DIM)
    ]
    expected_by_centroid = {}
    for centroid_idx in range(cute_kmeans_assign_topk.WIDE_CENTROIDS_PER_TILE):
        centroid = [
            (centroid_idx * 7 + feature_idx * 3 + feature_idx // 2) % 17
            for feature_idx in range(cute_kmeans_assign_topk.WIDE_FEATURE_DIM)
        ]
        if score_only:
            value = (
                sum(entry * entry for entry in centroid)
                - 2 * sum(x * c for x, c in zip(query, centroid, strict=True))
                + cute_kmeans_assign_topk.FEATURE_SPLIT_SCORE_OFFSET
            )
        else:
            value = sum((x - c) * (x - c) for x, c in zip(query, centroid, strict=True))
        expected_by_centroid[centroid_idx] = value

    threshold = sorted(expected_by_centroid.values())[expected_k - 1]
    required_centroids = {
        centroid
        for centroid, value in expected_by_centroid.items()
        if value < threshold
    }
    actual_distances = result["top_distances"]
    actual_centroids = result["top_centroids"]
    actual_centroid_set = set(actual_centroids)

    assert len(actual_distances) == expected_k
    assert len(actual_centroids) == expected_k
    assert len(actual_centroid_set) == expected_k
    assert required_centroids <= actual_centroid_set
    for distance, centroid in zip(actual_distances, actual_centroids, strict=True):
        assert 0 <= centroid < cute_kmeans_assign_topk.WIDE_CENTROIDS_PER_TILE
        assert distance == expected_by_centroid[centroid]
        assert distance <= threshold


def test_provider_kmeans_assign_topk_feature_split_example_runtime(source_examples):
    from examples.cutlass import cute_kmeans_assign_topk

    prepared = cute_kmeans_assign_topk.prepare_feature_split_batched_example()
    prepared.step()
    result = prepared.validate()

    assert result["query_count"] == cute_kmeans_assign_topk.BATCHED_QUERY_COUNT
    assert result["topk_k"] == cute_kmeans_assign_topk.TOPK_K
    _assert_kmeans_assign_first_query_result(
        result,
        cute_kmeans_assign_topk,
        expected_k=cute_kmeans_assign_topk.TOPK_K,
    )


def test_provider_kmeans_assign_topk_feature_split_score_example_runtime(
    source_examples,
):
    from examples.cutlass import cute_kmeans_assign_topk

    prepared = cute_kmeans_assign_topk.prepare_feature_split_score_batched_example()
    prepared.step()
    result = prepared.validate()

    assert result["query_count"] == cute_kmeans_assign_topk.BATCHED_QUERY_COUNT
    assert result["topk_k"] == cute_kmeans_assign_topk.TOPK_K
    _assert_kmeans_assign_first_query_result(
        result,
        cute_kmeans_assign_topk,
        expected_k=cute_kmeans_assign_topk.TOPK_K,
        score_only=True,
    )


def test_provider_kmeans_assign_topk_feature_split_top1_score_example_runtime(
    source_examples,
):
    from examples.cutlass import cute_kmeans_assign_topk

    prepared = (
        cute_kmeans_assign_topk.prepare_feature_split_top1_score_batched_example()
    )
    prepared.step()
    result = prepared.validate()

    assert result["query_count"] == cute_kmeans_assign_topk.BATCHED_QUERY_COUNT
    assert result["topk_k"] == cute_kmeans_assign_topk.FEATURE_SPLIT_SCORE_TOP1_K
    _assert_kmeans_assign_first_query_result(
        result,
        cute_kmeans_assign_topk,
        expected_k=cute_kmeans_assign_topk.FEATURE_SPLIT_SCORE_TOP1_K,
        score_only=True,
    )


def test_provider_kmeans_assign_topk_feature_split_top1_score_warp_example_runtime(
    source_examples,
):
    from examples.cutlass import cute_kmeans_assign_topk

    prepared = (
        cute_kmeans_assign_topk.prepare_feature_split_top1_score_warp_batched_example()
    )
    prepared.step()
    result = prepared.validate()

    assert result["query_count"] == cute_kmeans_assign_topk.BATCHED_QUERY_COUNT
    assert result["topk_k"] == cute_kmeans_assign_topk.FEATURE_SPLIT_SCORE_TOP1_K
    _assert_kmeans_assign_first_query_result(
        result,
        cute_kmeans_assign_topk,
        expected_k=cute_kmeans_assign_topk.FEATURE_SPLIT_SCORE_TOP1_K,
        score_only=True,
    )
