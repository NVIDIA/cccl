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
_CCCL_KERNEL_ATTRIBUTES void check_merge_path(merge_path_case<OffsetT> test, OffsetT* result)
{
  *result = cub::detail::block_merge_path(
    test.first, test.second, test.count1, test.count2, test.diagonal, merge_sequence_less{});
}

template <typename OffsetT>
void check_partition(merge_path_case<OffsetT> test, OffsetT expected)
{
  CAPTURE(test.count1, test.count2, test.diagonal);
  c2h::device_vector<OffsetT> result(1);

  check_merge_path<<<1, 1>>>(test, thrust::raw_pointer_cast(result.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(expected == result[0]);
}

using merge_path_offsets =
  c2h::type_list<std::uint8_t, std::uint16_t, std::int16_t, int, std::uint32_t, std::int64_t, std::uint64_t>;
using merge_path_large_offsets =
  c2h::type_list<std::uint16_t, std::int16_t, int, std::uint32_t, std::int64_t, std::uint64_t>;
using merge_path_wide_offsets = c2h::type_list<std::int64_t, std::uint64_t>;

CUB_TEST("BlockMerge partitions interleaved runs", "[block][merge][offset]", CUB_SMALL, merge_path_offsets)
{
  using offset_t      = c2h::get<0, TestType>;
  const auto diagonal = static_cast<offset_t>(GENERATE(Catch::Generators::range(0, 19)));

  check_partition<offset_t>({9, 9, diagonal, {0, 2}, {1, 2}}, (diagonal + 1) / 2);
}

CUB_TEST("BlockMerge partitions two empty runs", "[block][merge][offset]", CUB_SMALL, merge_path_offsets)
{
  using offset_t = c2h::get<0, TestType>;

  check_partition<offset_t>({0, 0, 0, {0, 0}, {0, 0}}, 0);
}

CUB_TEST("BlockMerge partitions an empty left run", "[block][merge][offset]", CUB_SMALL, merge_path_offsets)
{
  using offset_t = c2h::get<0, TestType>;

  check_partition<offset_t>({0, 9, 7, {0, 0}, {0, 0}}, 0);
}

CUB_TEST("BlockMerge partitions an empty right run", "[block][merge][offset]", CUB_SMALL, merge_path_offsets)
{
  using offset_t = c2h::get<0, TestType>;

  check_partition<offset_t>({9, 0, 7, {0, 0}, {0, 0}}, 7);
}

CUB_TEST("BlockMerge partitions equivalent keys near the offset limit",
         "[block][merge][offset]",
         CUB_SMALL,
         merge_path_large_offsets)
{
  using offset_t             = c2h::get<0, TestType>;
  constexpr offset_t maximum = (std::numeric_limits<offset_t>::max)();
  const offset_t diagonal =
    GENERATE_COPY(offset_t{0}, offset_t{1024}, offset_t(maximum - 2048), offset_t(maximum - 512), maximum);
  const offset_t count1 = maximum - 1024;

  check_partition<offset_t>({count1, 1024, diagonal, {0, 0}, {0, 0}}, (std::min) (diagonal, count1));
}

CUB_TEST("BlockMerge partitions runs larger than INT_MAX", "[block][merge][offset]", CUB_SMALL, merge_path_wide_offsets)
{
  using offset_t           = c2h::get<0, TestType>;
  constexpr offset_t count = static_cast<offset_t>((std::numeric_limits<int>::max)()) + 18;
  const offset_t diagonal  = GENERATE_COPY(offset_t{0}, offset_t{1}, count - 1, count, count + 257, 2 * count);

  check_partition<offset_t>({count, count, diagonal, {0, 2}, {1, 2}}, (diagonal + 1) / 2);
}
