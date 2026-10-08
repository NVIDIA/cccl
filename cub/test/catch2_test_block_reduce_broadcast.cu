// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/block/block_reduce.cuh>
#include <cub/util_ptx.cuh>

#include <thrust/memory.h>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cstdint>
#include <numeric>

#include "cub_test_macros.h"
#include <c2h/custom_type.h>
#include <c2h/generators.h>
#include <c2h/vector.h>

// %PARAM% TEST_DIM_X dimx 1:7:32:65:128:256
// %PARAM% TEST_DIM_YZ dimyz 1:2

template <int BlockDimX, int BlockDimYZ, class T, class ReductionOp>
__global__ void broadcast_kernel(const T* input, T* output, int valid_items, bool full_tile, ReductionOp reduction_op)
{
  using block_reduce_t = cub::BlockReduce<T, BlockDimX, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BlockDimYZ, BlockDimYZ>;
  __shared__ typename block_reduce_t::TempStorage storage;
  constexpr int block_threads = BlockDimX * BlockDimYZ * BlockDimYZ;
  const int tid               = static_cast<int>(cub::RowMajorTid(BlockDimX, BlockDimYZ, BlockDimYZ));
  block_reduce_t reduce(storage);
  output[tid] = full_tile ? reduce.ReduceBroadcast(input[tid], reduction_op)
                          : reduce.ReduceBroadcast(input[tid], reduction_op, valid_items);
  __syncthreads();
  const T aggregate =
    reduce.Reduce(input[tid], reduction_op, valid_items < block_threads ? valid_items : block_threads);
  if (tid == 0)
  {
    output[block_threads] = aggregate;
  }
}

template <int BlockDimX, int BlockDimYZ, class T, class ReductionOp>
void check_broadcast(c2h::device_vector<T>& input, ReductionOp reduction_op)
{
  constexpr int block_threads = BlockDimX * BlockDimYZ * BlockDimYZ;
  const bool full_tile        = GENERATE(true, false);
  const int valid_items =
    full_tile
      ? block_threads
      : GENERATE_COPY(
          1,
          std::min(7, block_threads),
          std::min(31, block_threads),
          std::min(33, block_threads),
          std::max(1, block_threads - 1),
          block_threads,
          block_threads + 3);
  CAPTURE(full_tile, valid_items);
  c2h::device_vector<T> output(block_threads + 1);
  broadcast_kernel<BlockDimX, BlockDimYZ><<<1, dim3(BlockDimX, BlockDimYZ, BlockDimYZ)>>>(
    thrust::raw_pointer_cast(input.data()),
    thrust::raw_pointer_cast(output.data()),
    valid_items,
    full_tile,
    reduction_op);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  const c2h::host_vector<T> host_input = input;
  const T expected                     = std::accumulate(
    host_input.begin() + 1, host_input.begin() + std::min(valid_items, block_threads), host_input[0], reduction_op);
  const c2h::host_vector<T> reference(block_threads + 1, expected);
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    REQUIRE_APPROX_EQ(reference, output);
  }
  else
  {
    REQUIRE(reference == output);
  }
  const c2h::host_vector<T> host_output = output;
  REQUIRE(std::all_of(host_output.begin(), host_output.end(), [&](const T& value) {
    return value == host_output[0];
  }));
}

using types = c2h::type_list<std::int32_t, float, double>;
using unsigned_ops =
  c2h::type_list<cuda::std::plus<>,
                 cuda::minimum<>,
                 cuda::maximum<>,
                 cuda::std::bit_and<>,
                 cuda::std::bit_or<>,
                 cuda::std::bit_xor<>>;

struct take_first_t
{
  template <class T>
  __host__ __device__ T operator()(const T& first, const T&) const
  {
    return first;
  }
};

struct compose_affine_t
{
  __host__ __device__ std::uint64_t operator()(std::uint64_t left, std::uint64_t right) const
  {
    const auto left_a     = static_cast<std::uint32_t>(left >> 32);
    const auto left_b     = static_cast<std::uint32_t>(left);
    const auto right_a    = static_cast<std::uint32_t>(right >> 32);
    const auto right_b    = static_cast<std::uint32_t>(right);
    const std::uint32_t a = left_a * right_a;
    const std::uint32_t b = left_a * right_b + left_b;
    return (static_cast<std::uint64_t>(a) << 32) | b;
  }
};

CUB_TEST("Block reduce broadcasts numeric aggregates", "[reduce][block][broadcast]", CUB_SMALL, types)
{
  using type = c2h::get<0, TestType>;
  c2h::device_vector<type> input(TEST_DIM_X * TEST_DIM_YZ * TEST_DIM_YZ);
  // Avoid cancellation in the sequential CPU reference; all GPU outputs still match Reduce exactly.
  const type minimum = cuda::std::is_floating_point_v<type> ? type{1} : type{-7};
  c2h::gen(C2H_SEED(3), input, minimum, type{7});
  if (GENERATE(true, false))
  {
    check_broadcast<TEST_DIM_X, TEST_DIM_YZ>(input, cuda::std::plus<>{});
  }
  else
  {
    check_broadcast<TEST_DIM_X, TEST_DIM_YZ>(input, cuda::maximum<>{});
  }
}

CUB_TEST("Block reduce broadcasts custom aggregates", "[reduce][block][broadcast]", CUB_SMALL)
{
  using type = c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t>;
  c2h::device_vector<type> input(TEST_DIM_X * TEST_DIM_YZ * TEST_DIM_YZ);
  c2h::gen(C2H_SEED(3), input);
  check_broadcast<TEST_DIM_X, TEST_DIM_YZ>(input, cuda::std::plus<>{});
}

CUB_TEST("Block reduce broadcasts noncommutative aggregates", "[reduce][block][broadcast]", CUB_SMALL, types)
{
  using type = c2h::get<0, TestType>;
  c2h::device_vector<type> input(TEST_DIM_X * TEST_DIM_YZ * TEST_DIM_YZ);
  c2h::gen(C2H_SEED(3), input, type{-7}, type{7});
  check_broadcast<TEST_DIM_X, TEST_DIM_YZ>(input, take_first_t{});
}

CUB_TEST("Block reduce preserves affine composition order", "[reduce][block][broadcast]", CUB_SMALL)
{
  // Composition modulo 2^32 is associative and depends on every item's order.
  c2h::host_vector<std::uint64_t> values(TEST_DIM_X * TEST_DIM_YZ * TEST_DIM_YZ);
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    values[i] = (static_cast<std::uint64_t>(2 * i + 3) << 32) | (i + 1);
  }
  c2h::device_vector<std::uint64_t> input = values;
  check_broadcast<TEST_DIM_X, TEST_DIM_YZ>(input, compose_affine_t{});
}

CUB_TEST("Block reduce broadcasts unsigned hardware aggregates", "[reduce][block][broadcast]", CUB_SMALL, unsigned_ops)
{
  using reduction_op = c2h::get<0, TestType>;
  c2h::host_vector<std::uint32_t> values(TEST_DIM_X * TEST_DIM_YZ * TEST_DIM_YZ);
  for (std::size_t i = 0; i < values.size(); ++i)
  {
    values[i] = 0x80000000u | (static_cast<std::uint32_t>(i) * 2654435761u);
  }
  c2h::device_vector<std::uint32_t> input = values;
  check_broadcast<TEST_DIM_X, TEST_DIM_YZ>(input, reduction_op{});
}
