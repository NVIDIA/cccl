// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/find.h>
#include <thrust/sequence.h>

#include <cuda/cmath>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "cub_test_macros.h"

// A copy of `size` bytes from `in_alignment` bytes into an input buffer to `out_alignment` bytes into an output buffer
struct vectorized_copy_case
{
  std::uint8_t* in_buffer;
  std::uint8_t* out_buffer;
  std::size_t in_alignment;
  std::size_t out_alignment;
  std::size_t size;
};

// Value of the output bytes in front of a copy, which the copy must not change
constexpr std::uint8_t untouched_value = 0x42;

template <typename VectorT>
static std::vector<vectorized_copy_case> make_copy_cases(
  const std::vector<std::size_t> sizes,
  c2h::seed_t seed,
  std::vector<c2h::device_vector<std::uint8_t>>& in_buffers,
  std::vector<c2h::device_vector<std::uint8_t>>& out_buffers)
{
  // The input is loaded in 4-byte words and the output is stored in `VectorT`s, so alignments modulo `VectorT` cover
  // every case
  static_assert(sizeof(VectorT) >= sizeof(std::uint32_t));
  constexpr std::size_t max_alignment = sizeof(VectorT);

  // The input buffers are filled from the same random bytes and the output buffers with `untouched_value`
  const std::size_t max_buffer_size = max_alignment + *std::max_element(sizes.begin(), sizes.end());
  c2h::device_vector<std::uint8_t> random_bytes(max_buffer_size, thrust::no_init);
  c2h::gen(seed, random_bytes);

  std::vector<vectorized_copy_case> cases;
  for (std::size_t size : sizes)
  {
    for (std::size_t in_alignment = 0; in_alignment < max_alignment; ++in_alignment)
    {
      // The copies to all output alignments read from the same input buffer
      auto& in_buffer = in_buffers.emplace_back(in_alignment + size, thrust::no_init);
      thrust::copy(random_bytes.cbegin(), random_bytes.cbegin() + in_buffer.size(), in_buffer.begin());
      for (std::size_t out_alignment = 0; out_alignment < max_alignment; ++out_alignment)
      {
        auto& out_buffer = out_buffers.emplace_back(out_alignment + size, untouched_value);
        cases.push_back(
          {thrust::raw_pointer_cast(in_buffer.data()),
           thrust::raw_pointer_cast(out_buffer.data()),
           in_alignment,
           out_alignment,
           size});
      }
    }
  }
  return cases;
}

// Each group of `LogicalWarpSize` threads performs one copy, so that the groups within a warp take different code paths
template <int LogicalWarpSize, typename VectorT, typename ByteOffsetT>
__global__ void test_vectorized_copy_cases_kernel(const vectorized_copy_case* cases, std::size_t num_cases)
{
  const std::size_t case_id = (std::size_t{blockIdx.x} * blockDim.x + threadIdx.x) / LogicalWarpSize;
  if (case_id < num_cases)
  {
    const vectorized_copy_case copy_case = cases[case_id];
    cub::detail::batch_memcpy::vectorized_copy<LogicalWarpSize, VectorT>(
      static_cast<int>(threadIdx.x % LogicalWarpSize),
      copy_case.out_buffer + copy_case.out_alignment,
      static_cast<ByteOffsetT>(copy_case.size),
      copy_case.in_buffer + copy_case.in_alignment);
  }
}

// Returns whether a copy is wrong or changed the output bytes in front of it
struct copy_case_mismatch_op
{
  __device__ bool operator()(const vectorized_copy_case& copy_case) const
  {
    for (std::size_t i = 0; i < copy_case.out_alignment; ++i)
    {
      if (copy_case.out_buffer[i] != untouched_value)
      {
        return true;
      }
    }
    for (std::size_t i = 0; i < copy_case.size; ++i)
    {
      if (copy_case.out_buffer[copy_case.out_alignment + i] != copy_case.in_buffer[copy_case.in_alignment + i])
      {
        return true;
      }
    }
    return false;
  }
};

/**
 * @brief Checks every copy on the device and reports the first wrong one.
 */
static void verify_copy_cases(const c2h::device_vector<vectorized_copy_case>& d_cases)
{
  const auto first_wrong_case =
    thrust::find_if(c2h::device_policy, d_cases.begin(), d_cases.end(), copy_case_mismatch_op{});
  if (first_wrong_case != d_cases.end())
  {
    const vectorized_copy_case copy_case = *first_wrong_case;
    const auto size                      = copy_case.size;
    const auto in_alignment              = copy_case.in_alignment;
    const auto out_alignment             = copy_case.out_alignment;
    CAPTURE(size, in_alignment, out_alignment);
    FAIL("The copied bytes or the output bytes in front of them are wrong");
  }
}

using vector_type_list       = c2h::type_list<uint32_t, uint4>;
using logical_warp_size_list = c2h::enum_type_list<int, 4, 8, 16, 32>;
using byte_offset_type_list  = c2h::type_list<std::int32_t, std::uint64_t>;

CUB_TEST("The vectorized copy used by DeviceMemcpy works at the boundaries of its copy modes",
         "[memcpy]",
         CUB_SMALL,
         vector_type_list,
         logical_warp_size_list,
         byte_offset_type_list)
{
  using vector_t                          = typename c2h::get<0, TestType>;
  constexpr int logical_warp_size         = c2h::get<1, TestType>::value;
  using byte_offset_t                     = typename c2h::get<2, TestType>;
  constexpr std::size_t vector_size       = sizeof(vector_t);
  constexpr std::size_t max_bytewise_size = cub::detail::batch_memcpy::max_bytewise_size<vector_t>;

  std::vector<std::size_t> sizes;
  // Byte-wise copies, the switch to vectorized copies, and vectorized copies with heads and tails of up to
  // `vector_size + 2` bytes each
  for (std::size_t size = 0; size <= max_bytewise_size + vector_size + 4; ++size)
  {
    sizes.push_back(size);
  }
  // Vectorized copies whose number of vectors crosses one and two rounds of the logical warp
  for (std::size_t rounds : {1, 2})
  {
    const std::size_t round_size = rounds * logical_warp_size * vector_size;
    for (std::size_t size = round_size - vector_size; size <= round_size + max_bytewise_size + vector_size; ++size)
    {
      sizes.push_back(size);
    }
  }

  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());

  std::vector<c2h::device_vector<std::uint8_t>> in_buffers;
  std::vector<c2h::device_vector<std::uint8_t>> out_buffers;
  const auto h_cases = make_copy_cases<vector_t>(sizes, C2H_SEED(1), in_buffers, out_buffers);
  c2h::device_vector<vectorized_copy_case> d_cases(h_cases.begin(), h_cases.end());
  const auto num_cases = h_cases.size();

  constexpr int block_threads = 128;
  const auto num_blocks       = cuda::ceil_div(num_cases * logical_warp_size, std::size_t{block_threads});
  test_vectorized_copy_cases_kernel<logical_warp_size, vector_t, byte_offset_t>
    <<<static_cast<unsigned int>(num_blocks), block_threads>>>(thrust::raw_pointer_cast(d_cases.data()), num_cases);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  verify_copy_cases(d_cases);
}
