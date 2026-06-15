// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Block-level tests for `block_topk` / `block_topk_air`: deterministic
// boundary fixtures (incl. `+/-0.0` ties) and both selection directions.

#include <cub/block/block_load.cuh>
#include <cub/block/block_topk.cuh>

#include <thrust/count.h>
#include <thrust/shuffle.h>

#include <cuda/std/span>

#include <algorithm>
#include <cmath>

#include "catch2_test_block_topk_common.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
template <typename KeyT, int BlockDim, int ItemsPerThread, bool IsFullTile, bool BlockedInput, bool SelectMax>
__global__ void topk_kernel(cuda::std::span<const KeyT> g_in, cuda::std::span<KeyT> g_top, int k, int num_valid)
{
  using topk_t = cub::detail::block_topk<KeyT, BlockDim, ItemsPerThread>;

  __shared__ typename topk_t::TempStorage smem;

  KeyT keys[ItemsPerThread];
  // Sentinel values are adversarial by design to surface bugs in partial tile handling.
  constexpr KeyT oob_sentinel =
    SelectMax ? cuda::std::numeric_limits<KeyT>::max() : cuda::std::numeric_limits<KeyT>::lowest();
  if constexpr (BlockedInput)
  {
    cub::LoadDirectBlocked(static_cast<int>(threadIdx.x), g_in.data(), keys, num_valid, oob_sentinel);
  }
  else
  {
    cub::LoadDirectStriped<BlockDim>(static_cast<int>(threadIdx.x), g_in.data(), keys, num_valid, oob_sentinel);
  }

  if constexpr (SelectMax)
  {
    topk_t(smem).template max_keys<IsFullTile>(keys, k, num_valid);
  }
  else
  {
    topk_t(smem).template min_keys<IsFullTile>(keys, k, num_valid);
  }

  for (int i = 0; i < ItemsPerThread; ++i)
  {
    const int idx = static_cast<int>(threadIdx.x) * ItemsPerThread + i;
    if (idx < k)
    {
      g_top[idx] = keys[i];
    }
  }
}

template <typename KeyT, int BlockDim, int ItemsPerThread, bool IsFullTile, bool BlockedInput, bool SelectMax>
void check_topk(const c2h::host_vector<KeyT>& h_in, cuda::std::span<const KeyT> h_ref, int k)
{
  constexpr int tile  = BlockDim * ItemsPerThread;
  const int num_valid = static_cast<int>(h_in.size());
  REQUIRE(0 < k);
  REQUIRE(num_valid <= tile);
  REQUIRE((!IsFullTile || num_valid == tile));

  const int top_size = cuda::std::min(k, num_valid);
  REQUIRE(static_cast<int>(h_ref.size()) == top_size);

  c2h::device_vector<KeyT> d_in(h_in);
  c2h::device_vector<KeyT> d_top(k, KeyT{});

  topk_kernel<KeyT, BlockDim, ItemsPerThread, IsFullTile, BlockedInput, SelectMax>
    <<<1, BlockDim>>>(to_span(d_in), to_span(d_top), k, num_valid);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<KeyT> h_top(d_top);
  h_top.resize(top_size);
  std::sort(h_top.begin(), h_top.end(), comparator_t<SelectMax>{});
  c2h::host_vector<KeyT> h_ref_vec(h_ref.begin(), h_ref.end());
  CAPTURE(bit_repr(h_top), bit_repr(h_ref_vec));
  REQUIRE(h_top == h_ref_vec);
}
} // namespace

using select_direction_max = c2h::type_list<cuda::std::false_type, cuda::std::true_type>;
using fp_key_types         = c2h::type_list<float, double>;

template <int BlockDim, int ItemsPerThread>
struct block_shape
{
  static constexpr int threads_per_block = BlockDim;
  static constexpr int items_per_thread  = ItemsPerThread;
};

using block_shapes_full_tile =
  c2h::type_list<block_shape<64, 8>, block_shape<256, 2>, block_shape<32, 16>, block_shape<128, 4>>;

C2H_TEST("block_topk preserves keys across FP edge cases", "[block][topk]", fp_key_types, select_direction_max)
{
  using key_t                            = c2h::get<0, TestType>;
  static constexpr bool select_max       = c2h::get<1, TestType>::value;
  static constexpr int threads_per_block = 128;
  static constexpr int items_per_thread  = 4;
  static constexpr int tile_size         = threads_per_block * items_per_thread;

  rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(1).get()));
  c2h::host_vector<key_t> h_in = distinct_keys<key_t>(tile_size, rng);
  h_in[0]                      = static_cast<key_t>(-0.0);
  h_in[1]                      = static_cast<key_t>(+0.0);
  h_in[2]                      = cuda::std::numeric_limits<key_t>::infinity();
  h_in[3]                      = -cuda::std::numeric_limits<key_t>::infinity();
  thrust::shuffle(h_in.begin(), h_in.end(), rng);

  CAPTURE(c2h::type_name<key_t>(), select_max);

  const int num_valid = tile_size / 2 + 7;
  c2h::host_vector<key_t> h_in_partial(h_in.begin(), h_in.begin() + num_valid);

  SECTION("full tile, blocked input")
  {
    static constexpr bool is_full_tile  = true;
    static constexpr bool blocked_input = true;
    const int k                         = GENERATE_COPY(values<int>({1, tile_size / 4, tile_size - 1}));
    CAPTURE(k);
    const auto h_ref = sorted_top_k<select_max>(h_in, k);
    check_topk<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
      h_in, to_span(h_ref), k);
  }

  SECTION("partial tile, blocked input")
  {
    static constexpr bool is_full_tile  = false;
    static constexpr bool blocked_input = true;
    const int k                         = GENERATE_COPY(values<int>({1, num_valid / 4, num_valid - 1}));
    CAPTURE(num_valid, k);
    const auto h_ref = sorted_top_k<select_max>(h_in_partial, k);
    check_topk<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
      h_in_partial, to_span(h_ref), k);
  }
}

C2H_TEST("block_topk::select_* selects the right top-k on a full tile",
         "[block][topk]",
         fp_key_types,
         block_shapes_full_tile,
         select_direction_max)
{
  using key_t                            = c2h::get<0, TestType>;
  using shape_t                          = c2h::get<1, TestType>;
  static constexpr bool select_max       = c2h::get<2, TestType>::value;
  static constexpr int threads_per_block = shape_t::threads_per_block;
  static constexpr int items_per_thread  = shape_t::items_per_thread;
  static constexpr int tile_size         = threads_per_block * items_per_thread;

  rng_t rng(static_cast<cuda::std::uint32_t>(C2H_SEED(2).get()));
  const int k = GENERATE_COPY(values<int>({1, tile_size / 4, tile_size / 2, tile_size - 1}));

  static constexpr bool is_full_tile  = true;
  static constexpr bool blocked_input = true;

  const int overhang = GENERATE_COPY(overhang_generator(tile_size - k <= 1, {0, 1, tile_size - k}));

  auto run_check = [&](rng_t& local_rng, key_t boundary_key) {
    CAPTURE(c2h::type_name<key_t>(), select_max, k, overhang, boundary_key);
    c2h::host_vector<key_t> h_in =
      gen_keys_from_boundary_key<select_max>(tile_size, k, overhang, boundary_key, local_rng);
    const auto h_ref = sorted_top_k<select_max>(h_in, k);
    check_topk<key_t, threads_per_block, items_per_thread, is_full_tile, blocked_input, select_max>(
      h_in, to_span(h_ref), k);
  };

  SECTION("fixed boundary_key")
  {
    const key_t boundary_key = GENERATE_COPY(boundary_key_generator<key_t>());
    run_check(rng, boundary_key);
  }

  SECTION("random boundary_key")
  {
    const key_t boundary_key = random_boundary_key<key_t>(rng);
    run_check(rng, boundary_key);
  }
}

C2H_TEST("block_topk::{Min,Max}Keys preserve -0.0 in output", "[block][topk][float]", select_direction_max)
{
  static constexpr bool select_max       = c2h::get<0, TestType>::value;
  static constexpr int threads_per_block = 128;
  static constexpr int items_per_thread  = 1;
  static constexpr int tile_size         = 8;

  const c2h::host_vector<float> h_in =
    select_max ? c2h::host_vector<float>{-2.0f, -0.0f, -3.0f, 0.0f, -1.0f, -4.0f, -5.0f, -6.0f}
               : c2h::host_vector<float>{3.0f, -0.0f, 1.0f, 2.0f, 0.0f, -1.0f, 4.0f, 5.0f};
  const int k = select_max ? 3 : 5;

  const auto h_ref = sorted_top_k<select_max>(h_in, k);
  check_topk<float, threads_per_block, items_per_thread, false, true, select_max>(h_in, to_span(h_ref), k);

  c2h::device_vector<float> d_in(h_in);
  c2h::device_vector<float> d_top(k);
  topk_kernel<float, threads_per_block, items_per_thread, false, true, select_max>
    <<<1, threads_per_block>>>(to_span(d_in), to_span(d_top), k, tile_size);
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<float> h_top(d_top);
  const int num_minus_zero = static_cast<int>(thrust::count_if(h_top.begin(), h_top.end(), [](float x) {
    return x == 0.0f && std::signbit(x);
  }));
  REQUIRE(num_minus_zero >= 1);
}
