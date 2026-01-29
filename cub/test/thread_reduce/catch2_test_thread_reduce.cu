// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_macro.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstring>
#include <functional>
#include <limits>
#include <numeric>

#include "c2h/catch2_test_helper.h"
#include "c2h/extended_types.h"
#include "c2h/generators.h"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

/***********************************************************************************************************************
 * Thread Reduce Wrapper Kernels
 **********************************************************************************************************************/

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel(const T* __restrict__ d_in, T* __restrict__ d_out, ReduceOperator reduce_operator)
{
  T thread_data[NUM_ITEMS];
#pragma unroll
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::ThreadReduce(thread_data, reduce_operator);
}

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel_array(const T* d_in, T* d_out, ReduceOperator reduce_operator)
{
  cuda::std::array<T, NUM_ITEMS> thread_data;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::ThreadReduce(thread_data, reduce_operator);
}

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel_span(const T* d_in, T* d_out, ReduceOperator reduce_operator)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  cuda::std::span<T, NUM_ITEMS> span(thread_data);
  *d_out = cub::ThreadReduce(span, reduce_operator);
}

#if _CCCL_STD_VER >= 2023

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void thread_reduce_kernel_mdspan(const T* d_in, T* d_out, ReduceOperator reduce_operator)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  using Extent = cuda::std::extents<int, NUM_ITEMS>;
  cuda::std::mdspan<T, Extent> mdspan(thread_data, cuda::std::extents<int, NUM_ITEMS>{});
  *d_out = cub::ThreadReduce(mdspan, reduce_operator);
}

#endif // _CCCL_STD_VER >= 2023

/***********************************************************************************************************************
 * CUB operator to STD operator
 **********************************************************************************************************************/

template <typename T, typename>
struct cub_operator_to_std;

template <typename T>
struct cub_operator_to_std<T, cuda::std::plus<>>
{
  using type = ::std::plus<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, cuda::std::multiplies<>>
{
  using type = ::std::multiplies<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, cuda::std::bit_and<>>
{
  using type = ::std::bit_and<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, cuda::std::bit_or<>>
{
  using type = ::std::bit_or<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, cuda::std::bit_xor<>>
{
  using type = ::std::bit_xor<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, cuda::minimum<>>
{
  using type = cuda::minimum<>;
};

template <typename T>
struct cub_operator_to_std<T, cuda::maximum<>>
{
  using type = cuda::maximum<>;
};

template <typename T, typename Operator>
using cub_operator_to_std_t = typename cub_operator_to_std<T, Operator>::type;

/***********************************************************************************************************************
 * Type list definition
 **********************************************************************************************************************/

using narrow_precision_type_list = c2h::type_list<
#if TEST_HALF_T()
  __half,
#endif // TEST_HALF_T()
#if TEST_BF_T()
  __nv_bfloat16
#endif // TEST_BF_T()
  >;

using integral_type_list =
  c2h::type_list<cuda::std::int8_t, cuda::std::int16_t, cuda::std::uint16_t, cuda::std::int32_t, cuda::std::int64_t>;

using fp_type_list = c2h::type_list<float, double>;

using cub_operator_integral_list =
  c2h::type_list<cuda::std::plus<>,
                 cuda::std::multiplies<>,
                 cuda::std::bit_and<>,
                 cuda::std::bit_or<>,
                 cuda::std::bit_xor<>,
                 cuda::minimum<>,
                 cuda::maximum<>>;

using cub_operator_fp_list =
  c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::minimum<>, cuda::maximum<>>;

/***********************************************************************************************************************
 * Verify results and kernel launch
 **********************************************************************************************************************/

_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES((cuda::std::is_floating_point_v<T>) )
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE_THAT(expected_data, Catch::Matchers::WithinRel(test_results, T{0.05}));
}

_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES((!cuda::std::is_floating_point_v<T>) )
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE(expected_data == test_results);
}

template <typename T, typename ReduceOperator>
void run_thread_reduce_kernel(
  int num_items, const c2h::device_vector<T>& in, c2h::device_vector<T>& out, ReduceOperator reduce_operator)
{
  switch (num_items)
  {
    case 1:
      thread_reduce_kernel<1>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 2:
      thread_reduce_kernel<2>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 3:
      thread_reduce_kernel<3>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 4:
      thread_reduce_kernel<4>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 5:
      thread_reduce_kernel<5>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 6:
      thread_reduce_kernel<6>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 7:
      thread_reduce_kernel<7>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 8:
      thread_reduce_kernel<8>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 9:
      thread_reduce_kernel<9>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 10:
      thread_reduce_kernel<10>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 11:
      thread_reduce_kernel<11>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 12:
      thread_reduce_kernel<12>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 13:
      thread_reduce_kernel<13>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 14:
      thread_reduce_kernel<14>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 15:
      thread_reduce_kernel<15>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    case 16:
      thread_reduce_kernel<16>
        <<<1, 1>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), reduce_operator);
      break;
    default:
      FAIL("Unsupported number of items");
  }
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

constexpr int max_size  = 16;
constexpr int num_seeds = 10;

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("ThreadReduce Integral Type Tests", "[reduce][thread]", integral_type_list, cub_operator_integral_list)
{
  using value_t                    = c2h::get<0, TestType>;
  using op_t                       = c2h::get<1, TestType>;
  constexpr auto reduce_op         = op_t{};
  constexpr auto std_reduce_op     = cub_operator_to_std_t<value_t, op_t>{};
  constexpr auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  for (int num_items = 1; num_items <= max_size; ++num_items)
  {
    auto reference_result = std::accumulate(h_in.begin(), h_in.begin() + num_items, operator_identity, std_reduce_op);
    run_thread_reduce_kernel(num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

C2H_TEST("ThreadReduce Floating-Point Type Tests", "[reduce][thread]", fp_type_list, cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  constexpr auto reduce_op     = op_t{};
  constexpr auto std_reduce_op = cub_operator_to_std_t<value_t, op_t>{};
  const auto operator_identity = cuda::identity_element<op_t, value_t>();
  CAPTURE(c2h::type_name<value_t>(), max_size, c2h::type_name<decltype(reduce_op)>());
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, cuda::std::numeric_limits<value_t>::min());
  c2h::host_vector<value_t> h_in = d_in;
  for (int num_items = 1; num_items <= max_size; ++num_items)
  {
    auto reference_result = std::accumulate(h_in.begin(), h_in.begin() + num_items, operator_identity, std_reduce_op);
    run_thread_reduce_kernel(num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
  }
}

#if TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Narrow PrecisionType Tests",
         "[reduce][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  constexpr auto reduce_op     = op_t{};
  constexpr auto std_reduce_op = cub_operator_to_std_t<float, op_t>{};
  const auto operator_identity = cuda::identity_element<op_t, float>();
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, value_t{1.0f}, value_t{2.0f});
  c2h::host_vector<float> h_in_float = d_in;
  for (int num_items = 1; num_items <= max_size; ++num_items)
  {
    CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<decltype(reduce_op)>());
    auto reference_result =
      std::accumulate(h_in_float.begin(), h_in_float.begin() + num_items, operator_identity, std_reduce_op);
    run_thread_reduce_kernel(num_items, d_in, d_out, reduce_op);
    verify_results(reference_result, float{c2h::host_vector<value_t>(d_out)[0]});
  }
}

#endif // TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Container Tests", "[reduce][thread]")
{
  c2h::device_vector<int> d_in(max_size);
  c2h::device_vector<int> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in);
  c2h::host_vector<int> h_in = d_in;
  auto reference_result      = std::accumulate(h_in.begin(), h_in.end(), 0, std::plus<int>{});

  thread_reduce_kernel_array<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), cuda::std::plus<>{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  verify_results(reference_result, c2h::host_vector<int>(d_out)[0]);

  thread_reduce_kernel_span<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), cuda::std::plus<>{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  verify_results(reference_result, c2h::host_vector<int>(d_out)[0]);

#if _CCCL_STD_VER >= 2023
  thread_reduce_kernel_mdspan<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), cuda::std::plus<>{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  verify_results(reference_result, c2h::host_vector<int>(d_out)[0]);
#endif // _CCCL_STD_VER >= 2023
}
