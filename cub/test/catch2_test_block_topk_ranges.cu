// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/specializations/block_topk_air.cuh>

#include <thrust/memory.h>

#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/span>

#include <algorithm>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace
{
template <bool Unroll, bool Memoize, bool SelectMax, bool Pairs, bool Full>
__global__ void topk_range_kernel(
  cuda::std::span<const unsigned> input,
  cuda::std::span<unsigned> output,
  cuda::std::span<unsigned> payload,
  int k,
  int valid,
  int begin_bit,
  int end_bit,
  bool default_range)
{
  using value_t            = cuda::std::conditional_t<Pairs, unsigned, cub::NullType>;
  using topk_t             = cub::detail::block_topk_air<unsigned, 128, 2, value_t, 8, Unroll, Memoize>;
  constexpr auto direction = SelectMax ? cub::detail::topk::select::max : cub::detail::topk::select::min;
  __shared__ typename topk_t::TempStorage storage;
  unsigned keys[2];
  unsigned values[2];
  for (int i = 0; i < 2; ++i)
  {
    const int index = static_cast<int>(threadIdx.x) * 2 + i;
    keys[i]         = index < valid ? input[index] : (SelectMax ? ~0u : 0u);
    values[i]       = static_cast<unsigned>(index);
  }
  if constexpr (Pairs)
  {
    if (default_range)
    {
      topk_t(storage).template select_pairs<direction, Full>(keys, values, k, valid);
    }
    else
    {
      topk_t(storage).template select_pairs<direction, Full>(keys, values, k, valid, begin_bit, end_bit);
    }
  }
  else
  {
    if (default_range)
    {
      topk_t(storage).template select_keys<direction, Full>(keys, k, valid);
    }
    else
    {
      topk_t(storage).template select_keys<direction, Full>(keys, k, valid, begin_bit, end_bit);
    }
  }
  for (int i = 0; i < 2; ++i)
  {
    const int index = static_cast<int>(threadIdx.x) * 2 + i;
    if (index < k)
    {
      output[index] = keys[i];
      if constexpr (Pairs)
      {
        payload[index] = values[i];
      }
    }
  }
}
} // namespace

using topk_option = c2h::type_list<cuda::std::false_type, cuda::std::true_type>;

CUB_TEST("block_topk_air preserves bit ranges across implementation options",
         "[block][topk][bit-ranges]",
         CUB_SMALL,
         topk_option,
         topk_option,
         topk_option,
         topk_option,
         topk_option)
{
  constexpr bool unroll     = c2h::get<0, TestType>::value;
  constexpr bool memoize    = c2h::get<1, TestType>::value;
  constexpr bool select_max = c2h::get<2, TestType>::value;
  constexpr bool pairs      = c2h::get<3, TestType>::value;
  constexpr bool full       = c2h::get<4, TestType>::value;
  constexpr int valid       = full ? 256 : 157;
  const int range           = GENERATE(0, 1, 2, 3);
  const int k               = range == 3 ? 47 : 17;
  const int begin_bit       = range >= 2 ? 3 : 0;
  const int end_bit         = range == 3 ? 6 : (range == 2 ? 12 : 32);
  const unsigned mask       = range == 3 ? 7u : (range == 2 ? 511u : ~0u);
  CAPTURE(unroll, memoize, select_max, pairs, full, range);

  c2h::host_vector<unsigned> input(valid);
  for (int i = 0; i < valid; ++i)
  {
    // The nine-bit range has unique digits; the three-bit range ties at the k boundary.
    // The high bits reverse the full-key order.
    input[i] = (static_cast<unsigned>(255 - i) << 20) | (static_cast<unsigned>((i * 73) % 256 + 1) << 3)
             | static_cast<unsigned>(i % 8);
  }
  auto reference = input;
  std::sort(reference.begin(), reference.end(), [=](unsigned lhs, unsigned rhs) {
    const unsigned lhs_rank = (lhs >> begin_bit) & mask;
    const unsigned rhs_rank = (rhs >> begin_bit) & mask;
    return select_max ? lhs_rank > rhs_rank : lhs_rank < rhs_rank;
  });
  reference.resize(k);
  if (range == 3)
  {
    for (auto& key : reference)
    {
      key = (key >> begin_bit) & mask;
    }
  }
  std::sort(reference.begin(), reference.end());

  c2h::device_vector<unsigned> device_input(input);
  c2h::device_vector<unsigned> device_output(k);
  c2h::device_vector<unsigned> device_payload(k);
  topk_range_kernel<unroll, memoize, select_max, pairs, full><<<1, 128>>>(
    cuda::std::span<const unsigned>(thrust::raw_pointer_cast(device_input.data()), device_input.size()),
    cuda::std::span<unsigned>(thrust::raw_pointer_cast(device_output.data()), device_output.size()),
    cuda::std::span<unsigned>(thrust::raw_pointer_cast(device_payload.data()), device_payload.size()),
    k,
    valid,
    begin_bit,
    end_bit,
    range == 0);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<unsigned> output(device_output);
  if constexpr (pairs)
  {
    const c2h::host_vector<unsigned> payload(device_payload);
    for (int i = 0; i < k; ++i)
    {
      REQUIRE(payload[i] < static_cast<unsigned>(valid));
      REQUIRE(output[i] == input[payload[i]]);
    }
  }
  std::sort(output.begin(), output.end());
  if (range == 3)
  {
    REQUIRE(std::adjacent_find(output.begin(), output.end()) == output.end());
    for (auto& key : output)
    {
      REQUIRE(std::find(input.begin(), input.end(), key) != input.end());
      key = (key >> begin_bit) & mask;
    }
    std::sort(output.begin(), output.end());
  }
  REQUIRE(output == reference);
}
