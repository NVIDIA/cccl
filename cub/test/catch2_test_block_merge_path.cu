// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_merge.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/std/iterator>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "cub_test_macros.h"
#include <c2h/vector.h>

// Virtual sorted sequences exercise large counts without allocating huge tiles.
struct merge_sequence_iterator
{
  using value_type        = std::uint64_t;
  using difference_type   = std::int64_t;
  using pointer           = void;
  using reference         = value_type;
  using iterator_category = cuda::std::random_access_iterator_tag;
  std::uint64_t origin;
  std::uint64_t stride;

  __host__ __device__ value_type operator[](std::uint64_t index) const
  {
    return origin + stride * index;
  }
  __host__ __device__ merge_sequence_iterator operator+(std::uint64_t offset) const
  {
    return {origin + stride * offset, stride};
  }
};

struct merge_sequence_less
{
  __host__ __device__ bool operator()(std::uint64_t lhs, std::uint64_t rhs) const
  {
    return lhs < rhs;
  }
};

template <typename OffsetT>
struct merge_path_case
{
  OffsetT count1, count2, diagonal;
  merge_sequence_iterator first, second;
};

template <typename OffsetT>
_CCCL_KERNEL_ATTRIBUTES void check_merge_paths(const merge_path_case<OffsetT>* cases, OffsetT* result, int count)
{
  const int tid = static_cast<int>(threadIdx.x);

  if (tid < count)
  {
    const auto test = cases[tid];
    result[tid]     = cub::detail::block_merge_path(
      test.first, test.second, test.count1, test.count2, test.diagonal, merge_sequence_less{});
  }
}

using merge_path_offsets =
  c2h::type_list<std::uint8_t, std::uint16_t, std::int16_t, int, std::uint32_t, std::int64_t, std::uint64_t>;

CUB_TEST(
  "BlockMerge partitioning supports integral offset widths", "[block][merge][offset]", CUB_SMALL, merge_path_offsets)
{
  using offset_t = c2h::get<0, TestType>;
  c2h::host_vector<merge_path_case<offset_t>> cases;
  c2h::host_vector<offset_t> expected;
  auto add =
    [&](offset_t a,
        offset_t b,
        offset_t diagonal,
        merge_sequence_iterator first,
        merge_sequence_iterator second,
        offset_t answer) {
      cases.push_back({a, b, diagonal, first, second});
      expected.push_back(answer);
    };
  // Interleaved even and odd sequences have a closed-form partition.
  for (offset_t diagonal = 0; diagonal <= 18; ++diagonal)
  {
    add(9, 9, diagonal, {0, 2}, {1, 2}, (diagonal + 1) / 2);
  }
  add(0, 0, 0, {0, 0}, {0, 0}, 0);
  add(0, 9, 7, {0, 0}, {0, 0}, 0);
  add(9, 0, 7, {0, 0}, {0, 0}, 7);
  // Equal keys consume the earlier run first. Exercise rounding near the
  // maximum offset value, where a direct float-to-integer cast can overflow.
  if constexpr ((std::numeric_limits<offset_t>::max)() >= 2048)
  {
    const offset_t maximum = (std::numeric_limits<offset_t>::max)();

    for (offset_t diagonal : {offset_t{0}, offset_t{1024}, offset_t(maximum - 2048), offset_t(maximum - 512), maximum})
    {
      add(maximum - 1024, 1024, diagonal, {0, 0}, {0, 0}, (std::min) (diagonal, offset_t(maximum - 1024)));
    }
  }
  if constexpr (sizeof(offset_t) >= 8)
  {
    // Each input run exceeds INT_MAX; no actual memory is read by the iterator.
    const offset_t count = (offset_t{1} << 31) + 17;

    for (offset_t diagonal : {offset_t{0}, offset_t{1}, count - 1, count, count + 257, 2 * count})
    {
      add(count, count, diagonal, {0, 2}, {1, 2}, (diagonal + 1) / 2);
    }
  }
  c2h::device_vector<merge_path_case<offset_t>> d_cases = cases;
  c2h::device_vector<offset_t> result(cases.size());

  check_merge_paths<<<1, 64>>>(
    thrust::raw_pointer_cast(d_cases.data()), thrust::raw_pointer_cast(result.data()), static_cast<int>(cases.size()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(expected == result);
}
