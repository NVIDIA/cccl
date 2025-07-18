// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
// #define CCCL_CHECK_SASS

#if defined(CCCL_CHECK_SASS)
#  include <cub/warp/warp_reduce.cuh>

#  include <test_util.h>

#  include <c2h/catch2_test_helper.h>
#  include <c2h/extended_types.h>

__device__ uint64_t input[256];
__device__ uint64_t output[256];

template <typename T, typename ReductionOp>
__global__ void warp_reduce_kernel(ReductionOp reduction_op)
{
  using storage_t = typename cub::WarpReduce<T>::TempStorage;
  storage_t storage{};
  auto value      = *reinterpret_cast<T*>(&input + threadIdx.x);
  auto output_ptr = reinterpret_cast<T*>(&output + threadIdx.x);
  *output_ptr     = cub::WarpReduce<T>{storage}.Reduce(value, reduction_op);
}

/***********************************************************************************************************************
 * Types
 **********************************************************************************************************************/
// clang-format off

using arithmetic_type_list = c2h::type_list<
  int32_t, int64_t,
  float, double,
  cuda::std::complex<float>,
  cuda::std::complex<double>,
  ushort2
#  if TEST_HALF_T()
   , __half
   , __half2
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
   , __nv_bfloat16
   , __nv_bfloat162
#  endif // TEST_BF_T()
>;


using bitwise_type_list = c2h::type_list<uint32_t, uint64_t>;

using min_max_type_list = c2h::type_list<
  int32_t, int64_t,
   ushort2,
  float, double
#  if TEST_HALF_T()
  , __half
   , __half2
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
  , __nv_bfloat16
   , __nv_bfloat162
#  endif // TEST_BF_T()
  >;

// clang-format on
/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("SASS WarpReduce::Sum", "[reduce][warp][predefined_op][full]", arithmetic_type_list)
{
  using T = c2h::get<0, TestType>;
  warp_reduce_kernel<T><<<1, 32>>>(cuda::std::plus<>{});
}

C2H_TEST("SASS WarpReduce::Bitwise", "[reduce][warp][predefined_op][full]", bitwise_type_list)
{
  using T = c2h::get<0, TestType>;
  warp_reduce_kernel<T><<<1, 32>>>(cuda::std::bit_and<>{});
}

C2H_TEST("SASS WarpReduce::Min/Max", "[reduce][warp][predefined_op][full]", min_max_type_list)
{
  using T = c2h::get<0, TestType>;
  warp_reduce_kernel<T><<<1, 32>>>(cuda::minimum<>{});
}

#else

#  include "c2h/catch2_test_helper.h"

C2H_TEST("WarpReduce Empty Test", "[reduce][thread][empty]") {}

#endif // defined(CCCL_CHECK_SASS)
