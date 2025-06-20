// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cub/detail/integer_utils.cuh>

#include "c2h/catch2_test_helper.h"

/***********************************************************************************************************************
 * TEST CASES
 **********************************************************************************************************************/

using integral_types =
  c2h::type_list<int16_t,
                 uint16_t,
                 int32_t,
                 uint32_t,
                 int64_t,
                 uint64_t
#if TEST_INT128()
                 ,
                 __int128_t,
                 __uint128_t
#endif
                 >;

using floating_point_types =
  c2h::type_list<float,
                 double
#if TEST_HALF_T()
                 ,
                 __half
#endif
#if TEST_BF_T()
                 ,
                 __nv_bfloat16
#endif
                 >;

using operator_types = c2h::type_list<cuda::minimum<>, cuda::maximum<>>;

template <typename T>
static __global__ void test_int_kernel(const T* input, int num_items)
{
  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_items)
  {
    auto value       = input[i];
    auto [high, low] = cub::detail::split_integers(value);
    auto result      = cub::detail::merge_integers(high, low);
    assert(value == result);
  }
}

template <typename Operation, typename T>
static __global__ void test_float_kernel(Operation op, const T* input, int num_items)
{
  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_items - 1)
  {
    auto valueA     = input[i];
    auto valueB     = input[i + 1];
    auto int_valueA = cub::detail::floating_point_to_comparable_int(op, valueA);
    auto int_valueB = cub::detail::floating_point_to_comparable_int(op, valueB);
    assert((valueA < valueB) == (int_valueA < int_valueB));
    auto result_valueA = cub::detail::comparable_int_to_floating_point<T>(int_valueA);
    auto result_valueB = cub::detail::comparable_int_to_floating_point<T>(int_valueB);
    assert(valueA == result_valueA);
    assert(valueB == result_valueB);
  }
}

C2H_TEST("Split/Merge Integers", "[Split/Merge][Random]", integral_types)
{
  using T        = c2h::get<0, TestType>;
  auto num_items = 1 << 16;
  c2h::device_vector<T> d_in(1 << 16);
  c2h::gen(C2H_SEED(1), d_in);
  test_int_kernel<<<cuda::ceil_div(num_items, 256), 256>>>(thrust::raw_pointer_cast(d_in.data()), num_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

C2H_TEST(
  "Compare Floating-point with Integers", "[CompareFloatingPointWithInt][Random]", operator_types, floating_point_types)
{
  using cub::detail::comparable_int_to_floating_point;
  using cub::detail::floating_point_to_comparable_int;
  using Op       = c2h::get<0, TestType>;
  using T        = c2h::get<1, TestType>;
  auto num_items = 1 << 16;
  c2h::device_vector<T> d_in(1 << 16);
  c2h::gen(C2H_SEED(1), d_in);
  test_float_kernel<<<cuda::ceil_div(num_items, 256), 256>>>(Op{}, thrust::raw_pointer_cast(d_in.data()), num_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}
