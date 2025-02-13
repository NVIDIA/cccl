// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/device/device_memcpy.cuh>
#include <cub/util_macro.cuh>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>

#include <c2h/catch2_test_helper.h>

template <int LocigalWarpSize, typename VectorT, typename ByteOffsetT>
__global__ void test_vectorized_copy_kernel(const void* d_in, void* d_out, ByteOffsetT copy_size)
{
  cub::detail::batch_memcpy::vectorized_copy<LocigalWarpSize, VectorT>(threadIdx.x, d_out, copy_size, d_in);
}

using vector_type_list = c2h::type_list<uint32_t, uint4>;

C2H_TEST("The vectorized copy used by DeviceMemcpy works", "[memcpy]", vector_type_list)
{
  using vector_t                            = typename c2h::get<0, TestType>;
  constexpr std::uint32_t threads_per_block = 8;

  // Test the vectorized_copy for various aligned and misaligned input and output pointers.
  std::size_t in_offset  = GENERATE(0, 1, sizeof(uint32_t) - 1);
  std::size_t out_offset = GENERATE(0, 1, sizeof(vector_t) - 1);
  std::size_t copy_size =
    GENERATE_COPY(0, 1, sizeof(uint32_t), sizeof(vector_t), 2 * threads_per_block * sizeof(vector_t));
  CAPTURE(in_offset, out_offset, copy_size);

  // Prepare data
  const std::size_t alloc_size_in  = in_offset + copy_size;
  const std::size_t alloc_size_out = out_offset + copy_size;
  c2h::device_vector<uint8_t> data_input_buffer(alloc_size_in);
  c2h::device_vector<uint8_t> data_output_buffer(alloc_size_out);
  thrust::sequence(c2h::device_policy, data_input_buffer.begin(), data_input_buffer.end(), static_cast<uint8_t>(0));
  thrust::fill_n(c2h::device_policy, data_output_buffer.begin(), alloc_size_out, static_cast<uint8_t>(0x42));

  auto d_in  = thrust::raw_pointer_cast(data_input_buffer.data());
  auto d_out = thrust::raw_pointer_cast(data_output_buffer.data());

  test_vectorized_copy_kernel<threads_per_block, vector_t>
    <<<1, threads_per_block>>>(d_in + in_offset, d_out + out_offset, static_cast<int>(copy_size));

  // Verify the result
  c2h::device_vector<uint32_t> data_in(copy_size);
  c2h::device_vector<uint32_t> data_out(copy_size);
  thrust::copy(
    data_input_buffer.cbegin() + in_offset, data_input_buffer.cbegin() + in_offset + copy_size, data_in.begin());
  thrust::copy(
    data_output_buffer.cbegin() + out_offset, data_output_buffer.cbegin() + out_offset + copy_size, data_out.begin());
  REQUIRE(data_in == data_out);
}
