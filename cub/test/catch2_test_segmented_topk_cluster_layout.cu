// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/dispatch/dispatch_batched_topk_cluster.cuh>

#include <cuda/std/cstdint>

#include <c2h/catch2_test_helper.h>

namespace
{
template <typename SizeT>
[[nodiscard]] constexpr SizeT host_ceil_div(SizeT numerator, SizeT denominator)
{
  return (numerator + denominator - 1) / denominator;
}

template <typename KeyT, int ChunkBytes, int LoadAlignBytes>
void check_layout_case(int dynamic_smem_bytes, int cluster_blocks)
{
  using layout_t = cub::detail::batched_topk_cluster::smem_tile_layout<KeyT, ChunkBytes, LoadAlignBytes>;

  const int usable_bytes = dynamic_smem_bytes - layout_t::base_padding_bytes;
  REQUIRE(usable_bytes > 0);

  const int slots = usable_bytes / layout_t::slot_stride_bytes;
  REQUIRE(slots > 0);

  const auto tile_capacity = layout_t::tile_capacity(dynamic_smem_bytes);
  REQUIRE(tile_capacity == static_cast<cuda::std::uint32_t>(slots * layout_t::chunk_items));

  const auto coverage = layout_t::template cluster_coverage<cuda::std::int64_t>(cluster_blocks, tile_capacity);
  REQUIRE(coverage > 0);

  const auto max_rank_chunks = [](cuda::std::int64_t segment_size, int head_items, int blocks) {
    using size_t             = cuda::std::int64_t;
    const size_t head_chunks = head_items == 0 ? size_t{0} : size_t{1};
    const size_t tail_items  = segment_size - head_items;
    const size_t chunks      = head_chunks + host_ceil_div(tail_items, size_t{layout_t::chunk_items});
    return host_ceil_div(chunks, static_cast<size_t>(blocks));
  };

  const int heads[] = {0, 1, layout_t::chunk_items / 2, layout_t::chunk_items - 1};
  for (const int head_items : heads)
  {
    CAPTURE(c2h::type_name<KeyT>(),
            ChunkBytes,
            LoadAlignBytes,
            dynamic_smem_bytes,
            cluster_blocks,
            slots,
            tile_capacity,
            coverage,
            head_items);
    REQUIRE(max_rank_chunks(coverage, head_items, cluster_blocks) <= slots);
  }

  const auto unreserved_coverage = static_cast<cuda::std::int64_t>(cluster_blocks) * tile_capacity;
  CAPTURE(c2h::type_name<KeyT>(), ChunkBytes, LoadAlignBytes, dynamic_smem_bytes, cluster_blocks, slots, tile_capacity);
  REQUIRE(max_rank_chunks(unreserved_coverage, 1, cluster_blocks) == slots + 1);
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

TEST_CASE("Segmented TopK cluster SMEM layout reserves the unaligned head chunk",
          "[keys][segmented][topk][cluster][layout]")
{
  using default_policy  = cub::detail::batched_topk_cluster::policy_selector;
  constexpr auto policy = default_policy{}(cuda::compute_capability{9, 0});

  using default_float_layout =
    cub::detail::batched_topk_cluster::smem_tile_layout<float, policy.chunk_bytes, policy.load_align_bytes>;
  static_assert(default_float_layout::tile_capacity(0) == 0);
  static_assert(default_float_layout::template cluster_coverage<int>(8, 0) == 0);
  static_assert(default_float_layout::template cluster_coverage<int>(1, default_float_layout::chunk_items) == 0);

  check_layout_matrix<cuda::std::uint8_t, policy.chunk_bytes, policy.load_align_bytes>();
  check_layout_matrix<float, policy.chunk_bytes, policy.load_align_bytes>();
  check_layout_matrix<cuda::std::uint64_t, policy.chunk_bytes, policy.load_align_bytes>();

  check_layout_matrix<float, 4 * 1024, 128>();
  check_layout_matrix<float, 16 * 1024, 128>();
  check_layout_matrix<cuda::std::uint64_t, 16 * 1024, 256>();
}
