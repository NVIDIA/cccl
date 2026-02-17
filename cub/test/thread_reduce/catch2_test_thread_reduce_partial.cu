// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_macro.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "catch2_test_device_reduce.cuh"
#include "thread_reduce/catch2_test_thread_reduce_helper.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <c2h/generators.h>
#include <c2h/operator.cuh>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

inline constexpr int max_size  = 16;
inline constexpr int num_seeds = 3;

/***********************************************************************************************************************
 * Thread Reduce Wrapper Kernels
 **********************************************************************************************************************/

template <int NumItems, typename In, typename Out, typename ReduceOperator>
__global__ void thread_reduce_partial_kernel(In d_in, Out d_out, ReduceOperator reduce_operator, int valid_items)
{
  using value_t = cuda::std::iter_value_t<In>;
  value_t thread_data[NumItems];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::detail::ThreadReducePartial(thread_data, reduce_operator, valid_items);
}

template <int NumItems, typename T, typename ReduceOperator>
__global__ void
thread_reduce_partial_kernel_array(const T* d_in, T* d_out, ReduceOperator reduce_operator, int valid_items)
{
  cuda::std::array<T, NumItems> thread_data;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::detail::ThreadReducePartial(thread_data, reduce_operator, valid_items);
}

template <int NumItems, typename T, typename ReduceOperator>
__global__ void
thread_reduce_partial_kernel_span(const T* d_in, T* d_out, ReduceOperator reduce_operator, int valid_items)
{
  T thread_data[NumItems];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  cuda::std::span<T, NumItems> span(thread_data);
  *d_out = cub::detail::ThreadReducePartial(span, reduce_operator, valid_items);
}

#if _CCCL_STD_VER >= 2023

template <int NumItems, typename T, typename ReduceOperator>
__global__ void
thread_reduce_partial_kernel_mdspan(const T* d_in, T* d_out, ReduceOperator reduce_operator, int valid_items)
{
  T thread_data[NumItems];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  using Extent = cuda::std::extents<int, NumItems>;
  cuda::std::mdspan<T, Extent> mdspan(thread_data, cuda::std::extents<int, NumItems>{});
  *d_out = cub::detail::ThreadReducePartial(mdspan, reduce_operator, valid_items);
}

#endif // _CCCL_STD_VER >= 2023

/***********************************************************************************************************************
 * Type list definition
 **********************************************************************************************************************/

using narrow_precision_type_list = c2h::type_list<
#if TEST_HALF_T()
  half_t,
#endif // TEST_HALF_T()
#if TEST_BF_T()
  bfloat16_t
#endif // TEST_BF_T()
  >;

using integral_type_list =
  c2h::type_list<cuda::std::int8_t, cuda::std::uint16_t, cuda::std::int32_t, cuda::std::uint64_t>;

using fp_type_list = c2h::type_list<float, double>;

using cub_operator_integral_list =
  c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::std::bit_or<>, cuda::maximum<>>;

using cub_operator_fp_list = c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::minimum<>>;

static_assert(max_size > 4);
using items_per_thread_list = c2h::enum_type_list<int, 1, 3, max_size - 1, max_size>;

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("ThreadReduce Integral Type Tests",
         "[reduce][thread]",
         integral_type_list,
         cub_operator_integral_list,
         items_per_thread_list)
{
  using value_t                    = c2h::get<0, TestType>;
  using op_t                       = c2h::get<1, TestType>;
  using accum_t                    = cuda::std::__accumulator_t<op_t, value_t>;
  constexpr int num_items          = c2h::get<2, TestType>::value;
  using dist_param                 = dist_interval<value_t, op_t, num_items>;
  constexpr auto reduce_op         = op_t{};
  constexpr auto operator_identity = cuda::identity_element<op_t, accum_t>();
  const int valid_items            = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<decltype(reduce_op)>(), valid_items);
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<accum_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  const int bounded_valid_items  = cuda::std::min(valid_items, num_items);
  auto reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reduce_op, operator_identity);
  thread_reduce_partial_kernel<num_items>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), reduce_op, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == c2h::host_vector<accum_t>(d_out)[0]);
}

C2H_TEST("ThreadReduce Floating-Point Type Tests",
         "[reduce][thread]",
         fp_type_list,
         cub_operator_fp_list,
         items_per_thread_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = cuda::std::__accumulator_t<op_t, value_t>;
  constexpr int num_items      = c2h::get<2, TestType>::value;
  using dist_param             = dist_interval<value_t, op_t, num_items>;
  constexpr auto reduce_op     = op_t{};
  const auto operator_identity = cuda::identity_element<op_t, accum_t>();
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<decltype(reduce_op)>(), valid_items);
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<accum_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  const int bounded_valid_items  = cuda::std::min(valid_items, num_items);
  auto reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reduce_op, operator_identity);
  thread_reduce_partial_kernel<num_items>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), reduce_op, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == c2h::host_vector<accum_t>(d_out)[0]);
}

#if TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Narrow PrecisionType Tests",
         "[reduce][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list,
         items_per_thread_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = cuda::std::__accumulator_t<op_t, value_t>;
  constexpr int num_items      = c2h::get<2, TestType>::value;
  using dist_param             = dist_interval<value_t, op_t, num_items>;
  constexpr auto reduce_op     = unwrap_op(std::true_type{}, op_t{});
  const auto operator_identity = identity_v<op_t, accum_t>;
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<accum_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  CAPTURE(h_in, dist_param::min(), dist_param::max());
  CAPTURE(c2h::type_name<value_t>(), c2h::type_name<decltype(reduce_op)>(), valid_items, num_items, operator_identity);
  const int bounded_valid_items = cuda::std::min(valid_items, num_items);
  const value_t reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reduce_op, operator_identity);

  thread_reduce_partial_kernel<num_items><<<1, 1>>>(
    unwrap_it(thrust::raw_pointer_cast(d_in.data())),
    unwrap_it(thrust::raw_pointer_cast(d_out.data())),
    reduce_op,
    valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == c2h::host_vector<accum_t>(d_out)[0]);
}

#endif // TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Container Tests", "[reduce][thread]")
{
  using op_t       = cuda::std::plus<>;
  using dist_param = dist_interval<int, op_t, max_size>;
  c2h::device_vector<int> d_in(max_size);
  c2h::device_vector<int> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<int> h_in = d_in;
  const int valid_items      = GENERATE_COPY(
    take(1, random(2, max_size - 2)),
    take(1, random(max_size + 2, cuda::std::numeric_limits<int>::max())),
    values({1, max_size - 1, max_size, max_size + 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, max_size);
  auto reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, op_t{}, 0);

  thread_reduce_partial_kernel_array<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), op_t{}, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == c2h::host_vector<int>(d_out)[0]);

  thread_reduce_partial_kernel_span<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), op_t{}, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == c2h::host_vector<int>(d_out)[0]);

#if _CCCL_STD_VER >= 2023
  thread_reduce_partial_kernel_mdspan<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), op_t{}, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == c2h::host_vector<int>(d_out)[0]);
#endif // _CCCL_STD_VER >= 2023
}

C2H_TEST("ThreadReducePartial does not invoke the reduction operator on invalid elements", "[reduce][thread]")
{
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  const int valid_items = GENERATE_COPY(
    take(3, random(2, max_size - 1)),
    take(1, random(max_size + 1, cuda::std::numeric_limits<int>::max())),
    values({-1, 0, 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, max_size);
  CAPTURE(valid_items);
  // First initialize with invalid segments than overwrite the first valid_items
  c2h::host_vector<segment> h_in(max_size);
  thrust::copy(in_it, in_it + bounded_valid_items, h_in.begin());
  c2h::device_vector<segment> d_in = h_in;
  auto reference_result            = compute_single_problem_reference(
    h_in.cbegin(), h_in.cbegin() + bounded_valid_items, merge_segments_op{nullptr}, segment{1, 1});

  c2h::device_vector<segment> d_out(max_size);
  c2h::device_vector<bool> error_flag(1, false);
  thread_reduce_partial_kernel<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    merge_segments_op{thrust::raw_pointer_cast(error_flag.data())},
    valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(error_flag.front() == false);
  if (valid_items > 0)
  {
    REQUIRE(reference_result == c2h::host_vector<segment>(d_out)[0]);
  }
}
