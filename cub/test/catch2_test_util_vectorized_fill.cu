// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/vectorized_fill.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/std/functional>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

template <typename T, int NumItems>
__global__ void vectorized_fill_kernel(T* in, T val, T* out, ::cuda::std::integral_constant<int, NumItems>)
{
  // Dummy data to test proper alignment of data array
  [[maybe_unused]] char x{42};

  // To avoid misaligned writes in vectorized_fill, we need to ensure that the array is aligned to at least four
  // bytes (e.g., see AgentSelectIf)
  alignas(::cuda::std::max(4, static_cast<int>(alignof(T)))) T data[NumItems];
  for (int i = 0; i < NumItems; ++i)
  {
    data[i] = in[i];
  }

  cub::detail::vectorized_fill(data, val);

  for (int i = 0; i < NumItems; ++i)
  {
    out[i] = data[i];
  }
}

template <typename T, int NumItems>
CUB_RUNTIME_FUNCTION static cudaError_t test_vectorized_fill_wrapper(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  T* in,
  T val,
  T* out,
  ::cuda::std::integral_constant<int, NumItems> num_items_val,
  cudaStream_t stream = 0)
{
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }
  vectorized_fill_kernel<<<1, 1, 0, stream>>>(in, val, out, num_items_val);
  return cudaSuccess;
}

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(test_vectorized_fill_wrapper, test_vectorized_fill_from_device);

using types = c2h::type_list<char, std::int16_t, float, uchar3, double2, c2h::custom_type_t<c2h::equal_comparable_t>>;

using num_items_list = c2h::enum_type_list<int, 1, 2, 3, 4, 5, 7, 8, 11, 16, 17>;

C2H_TEST("CUB's vectorized_fill", "[util][vectorized_fill]", types, num_items_list)
{
  using type              = typename c2h::get<0, TestType>;
  constexpr int num_items = c2h::get<1, TestType>::value;

  c2h::device_vector<type> in(num_items);
  c2h::device_vector<type> out(num_items);
  c2h::gen(C2H_SEED(1), in);
  c2h::gen(C2H_SEED(1), out);

  // Pick a random value to fill the array with
  c2h::device_vector<type> val(1);
  c2h::gen(C2H_SEED(2), val);
  type value = val[0];

  c2h::device_vector<type> original_in(in);
  c2h::device_vector<type> expected_vals(num_items, value);

  test_vectorized_fill_from_device(
    thrust::raw_pointer_cast(in.data()),
    value,
    thrust::raw_pointer_cast(out.data()),
    ::cuda::std::integral_constant<int, num_items>{});

  REQUIRE(in == original_in);
  REQUIRE(out == expected_vals);
}
