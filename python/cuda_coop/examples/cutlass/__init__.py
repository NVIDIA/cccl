# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS examples for cuda.coop."""

__all__ = [
    "cute_kmeans_assign_gemm_argmin",
    "cute_kmeans_assign_topk",
    "cute_legacy_reduce_compare",
    "cute_mma_amax_sm100",
    "cute_mma_topk",
    "cute_mma_topk_sm100",
    "cute_run_length_decode_window",
    "cute_scheduler_prefix",
    "cute_sort_and_segment",
    "cute_sort_and_segment_thread_data",
    "cute_sort_register_fragment",
    "cute_thread_group_descriptor_reduce",
    "cute_thread_group_query",
    "cute_thread_group_reduce",
    "cute_thread_hierarchy_reduce",
    "cute_topk_score_window",
    "cute_warp_merge_sort",
    "cute_warp_prefix_reduce",
    "mixed_payload_factory_sort_topk",
    "mixed_payload_sort_topk",
    "mixed_tensor_vector_scan",
    "portable_root_sum",
    "prims_vector_block_exchange",
    "prims_vector_block_prefix_segment",
    "prims_vector_histogram_run_length",
    "prims_vector_pair_sort_topk",
    "prims_vector_rank_merge",
    "prims_vector_sort_topk",
    "prims_vector_warp_merge_sort",
    "prims_vector_warp_prefix",
]
