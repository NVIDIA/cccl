// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause
#include <cub/detail/integer_utils.cuh>

#include <c2h/catch2_test_helper.h>

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
    auto [high, low] = cub::detail::split_integer(value);
    static_assert(sizeof(high) == sizeof(T) / 2);
    static_assert(sizeof(low) == sizeof(T) / 2);
    static_assert(cuda::std::is_signed_v<T> == cuda::std::is_signed_v<decltype(high)>);
    static_assert(cuda::std::is_signed_v<T> == cuda::std::is_signed_v<decltype(low)>);
    auto result = cub::detail::merge_integers(high, low);
    static_assert(sizeof(result) == sizeof(T));
    static_assert(cuda::std::is_signed_v<T> == cuda::std::is_signed_v<decltype(result)>);
    assert(value == result);
  }
}

static __global__ void test_int_special_values_kernel()
{
  {
    auto [high, low] = cub::detail::split_integer(0xAABBCCDD);
    assert(high = 0xAABB);
    assert(low = 0xCCDD);
  }
  {
    auto [high, low] = cub::detail::split_integer(0xAABBCCDDEEFF1234);
    assert(high = 0xAABBCCDD);
    assert(low = 0xEEFF1234);
  }
}

template <typename Operation, typename T>
static __device__ void test_floating_point(Operation op, T valueA, T valueB)
{
  auto int_valueA = cub::detail::floating_point_to_comparable_int(op, valueA);
  auto int_valueB = cub::detail::floating_point_to_comparable_int(op, valueB);
  static_assert(sizeof(int_valueA) == sizeof(T));
  static_assert(sizeof(int_valueB) == sizeof(T));
  assert(op(valueA, valueB) == op(int_valueA, int_valueB));
  auto result_valueA = cub::detail::comparable_int_to_floating_point<T>(int_valueA);
  auto result_valueB = cub::detail::comparable_int_to_floating_point<T>(int_valueB);
  static_assert(cuda::std::is_same_v<T, decltype(result_valueA)>);
  static_assert(cuda::std::is_same_v<T, decltype(result_valueB)>);
  assert(valueA == result_valueA);
  assert(valueB == result_valueB);
}

template <typename Operation, typename T>
static __global__ void test_float_kernel(Operation op, const T* input, int num_items)
{
  auto i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i < num_items - 1)
  {
    test_floating_point(op, input[i], input[i + 1]);
  }
}

template <typename T, typename Operation>
static __global__ void test_float_special_values_kernel(Operation op)
{
  test_floating_point(op, T{-0.0f}, T{0.0f});
  test_floating_point(op, T{1.0f}, cuda::std::numeric_limits<T>::max());
  test_floating_point(op, T{1.0f}, cuda::std::numeric_limits<T>::min());
  test_floating_point(op, T{1.0f}, cuda::std::numeric_limits<T>::quiet_NaN());
  test_floating_point(op, cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::max());
  test_floating_point(op, cuda::std::numeric_limits<T>::max(), cuda::std::numeric_limits<T>::quiet_NaN());
  test_floating_point(op, cuda::std::numeric_limits<T>::min(), cuda::std::numeric_limits<T>::quiet_NaN());
  test_floating_point(op, cuda::std::numeric_limits<T>::quiet_NaN(), cuda::std::numeric_limits<T>::quiet_NaN());
}

C2H_TEST("Split/Merge Integers", "[Split/Merge][Random]", integral_types)
{
  using T              = c2h::get<0, TestType>;
  const auto num_items = 1 << 16;
  c2h::device_vector<T> d_in(num_items);
  c2h::gen(C2H_SEED(1), d_in);
  test_int_kernel<<<cuda::ceil_div(num_items, 256), 256>>>(thrust::raw_pointer_cast(d_in.data()), num_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  test_int_special_values_kernel<<<1, 1>>>();
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

C2H_TEST(
  "Compare Floating-point with Integers", "[CompareFloatingPointWithInt][Random]", operator_types, floating_point_types)
{
  using Op             = c2h::get<0, TestType>;
  using T              = c2h::get<1, TestType>;
  const auto num_items = 1 << 16;
  c2h::device_vector<T> d_in(num_items);
  c2h::gen(C2H_SEED(1), d_in);
  test_float_kernel<<<cuda::ceil_div(num_items, 256), 256>>>(Op{}, thrust::raw_pointer_cast(d_in.data()), num_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  test_float_special_values_kernel<T><<<1, 1>>>(Op{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}
