// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/agent/agent_batched_topk_cluster.cuh> // smem_block_tile_layout
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh> // make_cluster_policy()

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/cstdint>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace
{
template <typename KeyT, int ChunkBytes, int LoadAlignBytes>
void check_layout_case(int dynamic_smem_bytes, int cluster_blocks)
{
  using layout_t = cub::detail::batched_topk_cluster::smem_block_tile_layout<KeyT, ChunkBytes, LoadAlignBytes>;

  const int usable_bytes = dynamic_smem_bytes - layout_t::base_padding_bytes;
  REQUIRE(usable_bytes > 0);

  const int slots = usable_bytes / layout_t::chunk_bytes;
  REQUIRE(slots > 0);

  const auto max_block_resident_items = layout_t::max_block_resident_items(dynamic_smem_bytes);
  REQUIRE(max_block_resident_items == static_cast<cuda::std::uint32_t>(slots * layout_t::max_chunk_items));

  // The head is an edge (static SMEM), not a reserved chunk, so a cluster's coverage is the full physical per-CTA
  // capacity across its CTAs.
  const auto cluster_tile_capacity = static_cast<cuda::std::int64_t>(cluster_blocks) * max_block_resident_items;

  // Worst-case resident chunks on any single rank, mirroring the agent: only the aligned region `[head_items,
  // segment_size)` is chunked (`num_chunks`), then spread across `blocks` ranks (the `ceil_div` max for both the
  // strided and blocked partitions).
  const auto max_rank_chunks = [](cuda::std::int64_t segment_size, int head_items, int blocks) {
    using count_t            = cuda::std::int64_t;
    const count_t tail_items = segment_size - head_items;
    const count_t chunks     = ::cuda::ceil_div(tail_items, count_t{layout_t::max_chunk_items});
    return ::cuda::ceil_div(chunks, static_cast<count_t>(blocks));
  };

  const int heads[] = {0, 1, layout_t::max_chunk_items / 2, layout_t::max_chunk_items - 1};
  for (const int head_items : heads)
  {
    CAPTURE(c2h::type_name<KeyT>(),
            ChunkBytes,
            LoadAlignBytes,
            dynamic_smem_bytes,
            cluster_blocks,
            slots,
            max_block_resident_items,
            cluster_tile_capacity,
            head_items);
    REQUIRE(max_rank_chunks(cluster_tile_capacity, head_items, cluster_blocks) <= slots);
  }

  // Tightness: the full physical capacity fits resident (`slots` chunks per rank), one item beyond overflows.
  CAPTURE(c2h::type_name<KeyT>(),
          ChunkBytes,
          LoadAlignBytes,
          dynamic_smem_bytes,
          cluster_blocks,
          slots,
          max_block_resident_items);
  REQUIRE(max_rank_chunks(cluster_tile_capacity, 0, cluster_blocks) == slots);
  REQUIRE(max_rank_chunks(cluster_tile_capacity + 1, 0, cluster_blocks) == slots + 1);
}

template <typename KeyT, int ChunkBytes, int LoadAlignBytes>
void check_layout_matrix()
{
  constexpr int dynamic_smem_cases[]  = {48 * 1024, 96 * 1024, 160 * 1024, 228 * 1024};
  constexpr int cluster_block_cases[] = {8, 16};

  for (const int dynamic_smem_bytes : dynamic_smem_cases)
  {
    for (const int cluster_blocks : cluster_block_cases)
    {
      check_layout_case<KeyT, ChunkBytes, LoadAlignBytes>(dynamic_smem_bytes, cluster_blocks);
    }
  }
}
} // namespace

CUB_TEST("Segmented TopK cluster SMEM layout exposes the full physical capacity (head is an edge, not a chunk)",
         "[keys][segmented][topk][cluster][layout]",
         CUB_SMALL)
{
  constexpr auto policy = cub::detail::batched_topk::make_cluster_policy();

  using default_float_layout =
    cub::detail::batched_topk_cluster::smem_block_tile_layout<float, policy.chunk_bytes, policy.load_align_bytes>;
  static_assert(default_float_layout::max_block_resident_items(0) == 0);

  check_layout_matrix<cuda::std::uint8_t, policy.chunk_bytes, policy.load_align_bytes>();
  check_layout_matrix<float, policy.chunk_bytes, policy.load_align_bytes>();
  check_layout_matrix<cuda::std::uint64_t, policy.chunk_bytes, policy.load_align_bytes>();

  check_layout_matrix<float, 4 * 1024, 128>();
  check_layout_matrix<float, 16 * 1024, 128>();
  check_layout_matrix<cuda::std::uint64_t, 16 * 1024, 256>();
}
