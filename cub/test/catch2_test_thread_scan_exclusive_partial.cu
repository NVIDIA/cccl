// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/thread/thread_scan.cuh>

#include <cuda/functional>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/functional>
#include <cuda/std/limits>

#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include "thread_reduce/catch2_test_thread_reduce_helper.cuh"
#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>
#include <c2h/generators.h>
#include <c2h/operator.cuh>

constexpr int max_size  = 16;
constexpr int num_seeds = 3;

/***********************************************************************************************************************
 * Thread Scan Wrapper Kernels
 **********************************************************************************************************************/

template <int NumItems, typename In, typename Out, typename Accum, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel(
  In d_in, Out d_out, ScanOperator scan_operator, int valid_items, Accum prefix, bool apply_prefix, Accum filler)
{
  using value_t  = cuda::std::iter_value_t<In>;
  using output_t = cuda::std::iter_value_t<Out>;
  value_t thread_input[NumItems];
  output_t thread_output[NumItems];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_input[i]  = d_in[i];
    thread_output[i] = static_cast<output_t>(filler);
  }
  cub::detail::ThreadScanExclusivePartial(thread_input, thread_output, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    d_out[i] = thread_output[i];
  }
}

// The following kernels are less general/complex with the added benefit that we can test doing the scan in-place

template <int NumItems, typename T, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel_array(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  cuda::std::array<T, NumItems> thread_data;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  cub::detail::ThreadScanExclusivePartial(thread_data, thread_data, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

template <int NumItems, typename T, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel_span(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  T thread_data[NumItems];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  cuda::std::span<T, NumItems> span(thread_data);
  cub::detail::ThreadScanExclusivePartial(span, span, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

#if _CCCL_STD_VER >= 2023

template <int NumItems, typename T, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel_mdspan(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  T thread_data[NumItems];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    thread_data[i] = d_in[i];
  }
  using Extent = cuda::std::extents<int, NumItems>;
  cuda::std::mdspan<T, Extent> mdspan(thread_data, cuda::std::extents<int, NumItems>{});
  cub::detail::ThreadScanExclusivePartial(mdspan, mdspan, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NumItems; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

#endif // _CCCL_STD_VER >= 2023

/***********************************************************************************************************************
 * Type list definition
 **********************************************************************************************************************/

using narrow_precision_type_list = c2h::type_list<
#if TEST_HALF_T()
  type_pair<half_t>,
#endif // TEST_HALF_T()
#if TEST_BF_T()
  type_pair<bfloat16_t>
#endif // TEST_BF_T()
  >;

using integral_type_list =
  c2h::type_list<type_pair<cuda::std::uint8_t, cuda::std::int32_t>,
                 type_pair<cuda::std::int16_t>,
                 type_pair<cuda::std::uint32_t>,
                 type_pair<cuda::std::int64_t>>;

using fp_type_list = c2h::type_list<type_pair<float>, type_pair<double>>;

using cub_operator_integral_list =
  c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::std::bit_and<>, cuda::minimum<>>;

using cub_operator_fp_list = c2h::type_list<cuda::std::plus<>, cuda::std::multiplies<>, cuda::maximum<>>;

static_assert(max_size > 4);
using items_per_thread_list = c2h::enum_type_list<int, 1, 3, max_size - 1, max_size>;

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("ThreadScanExclusive Integral Type Tests",
         "[scan][thread]",
         integral_type_list,
         cub_operator_integral_list,
         items_per_thread_list)
{
  using params                     = params_t<TestType>;
  using value_t                    = typename params::item_t;
  using output_t                   = typename params::output_t;
  using op_t                       = c2h::get<1, TestType>;
  using accum_t                    = cuda::std::__accumulator_t<op_t, value_t>;
  constexpr int num_items          = c2h::get<2, TestType>::value;
  using dist_param                 = dist_interval<value_t, op_t, num_items, accum_t, output_t>;
  using filler_dist_param          = dist_interval<accum_t, op_t, num_items, accum_t, output_t>;
  constexpr auto scan_op           = op_t{};
  constexpr auto operator_identity = cuda::identity_element<op_t, accum_t>();
  const int valid_items            = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  const int bounded_valid_items = std::min(valid_items, num_items);
  const accum_t prefix          = GENERATE_COPY(take(1, random(dist_param::min(), dist_param::max())));
  const bool apply_prefix       = GENERATE(true, false);
  const accum_t filler          = GENERATE(take(1, random(filler_dist_param::min(), filler_dist_param::max())));
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix);

  c2h::device_vector<value_t> d_in(num_items, thrust::no_init);
  c2h::device_vector<output_t> d_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(num_items, static_cast<output_t>(filler));

  compute_exclusive_scan_reference(
    h_in.cbegin(),
    h_in.cbegin() + bounded_valid_items,
    reference_result.begin(),
    apply_prefix ? prefix : operator_identity,
    scan_op);

  thread_scan_exclusive_partial_kernel<num_items><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    scan_op,
    valid_items,
    prefix,
    apply_prefix,
    filler);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  if (!apply_prefix)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

C2H_TEST("ThreadScanExclusive Floating-Point Type Tests",
         "[scan][thread]",
         fp_type_list,
         cub_operator_fp_list,
         items_per_thread_list)
{
  using params                 = params_t<TestType>;
  using value_t                = typename params::item_t;
  using output_t               = typename params::output_t;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = cuda::std::__accumulator_t<op_t, value_t>;
  constexpr int num_items      = c2h::get<2, TestType>::value;
  using dist_param             = dist_interval<value_t, op_t, num_items, accum_t, output_t>;
  using filler_dist_param      = dist_interval<accum_t, op_t, num_items, accum_t, output_t>;
  constexpr auto scan_op       = op_t{};
  const auto operator_identity = cuda::identity_element<op_t, accum_t>();
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  const int bounded_valid_items = std::min(valid_items, num_items);
  const accum_t prefix          = GENERATE_COPY(take(1, random(dist_param::min(), dist_param::max())));
  const bool apply_prefix       = GENERATE(true, false);
  const accum_t filler          = GENERATE(take(1, random(filler_dist_param::min(), filler_dist_param::max())));
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix);

  c2h::device_vector<value_t> d_in(num_items, thrust::no_init);
  c2h::device_vector<value_t> d_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<value_t> reference_result(num_items, filler);

  compute_exclusive_scan_reference(
    h_in.cbegin(),
    h_in.cbegin() + bounded_valid_items,
    reference_result.begin(),
    apply_prefix ? prefix : operator_identity,
    scan_op);

  thread_scan_exclusive_partial_kernel<num_items><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    scan_op,
    valid_items,
    prefix,
    apply_prefix,
    filler);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  if (!apply_prefix)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

#if TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadScanExclusive Narrow PrecisionType Tests",
         "[scan][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list,
         items_per_thread_list)
{
  using params                 = params_t<TestType>;
  using value_t                = typename params::item_t;
  using output_t               = typename params::output_t;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = cuda::std::__accumulator_t<op_t, value_t>;
  constexpr int num_items      = c2h::get<2, TestType>::value;
  using dist_param             = dist_interval<value_t, op_t, num_items, accum_t, output_t>;
  using filler_dist_param      = dist_interval<accum_t, op_t, num_items, accum_t, output_t>;
  constexpr auto scan_op       = unwrap_op(std::true_type{}, op_t{});
  const auto operator_identity = identity_v<op_t, accum_t>;
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  const int bounded_valid_items = std::min(valid_items, num_items);
  auto prefix =
    static_cast<accum_t>(GENERATE_COPY(take(1, random(float{dist_param::min()}, float{dist_param::max()}))));
  const bool apply_prefix = GENERATE(true, false);
  auto filler =
    static_cast<accum_t>(GENERATE(take(1, random(float{filler_dist_param::min()}, float{filler_dist_param::max()}))));
  CAPTURE(
    c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix, operator_identity);

  c2h::device_vector<value_t> d_in(num_items, thrust::no_init);
  c2h::device_vector<output_t> d_out(num_items, thrust::no_init);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(num_items, filler);

  compute_exclusive_scan_reference(
    h_in.cbegin(),
    h_in.cbegin() + bounded_valid_items,
    reference_result.begin(),
    apply_prefix ? prefix : operator_identity,
    scan_op);

  thread_scan_exclusive_partial_kernel<num_items><<<1, 1>>>(
    unwrap_it(thrust::raw_pointer_cast(d_in.data())),
    unwrap_it(thrust::raw_pointer_cast(d_out.data())),
    scan_op,
    valid_items,
    *unwrap_it(&prefix),
    apply_prefix,
    *unwrap_it(&filler));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  if (!apply_prefix)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

#endif // TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadScanExclusive Container Tests", "[scan][thread]")
{
  c2h::device_vector<int> d_in(max_size, thrust::no_init);
  c2h::device_vector<int> d_out(max_size, thrust::no_init);
  using dist_param = dist_interval<int, cuda::std::plus<>, max_size>;
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<int> h_in = d_in;
  const int valid_items      = GENERATE_COPY(
    take(1, random(2, max_size - 2)),
    take(1, random(max_size + 2, cuda::std::numeric_limits<int>::max())),
    values({1, max_size - 1, max_size, max_size + 1}));
  const int bounded_valid_items          = cuda::std::min(valid_items, max_size);
  c2h::host_vector<int> reference_result = h_in;
  compute_exclusive_scan_reference(
    h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reference_result.begin(), 0, cuda::std::plus<>{});

  thread_scan_exclusive_partial_kernel_array<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    cuda::std::plus<>{},
    valid_items,
    0,
    true);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == d_out);

  thrust::fill(d_out.begin(), d_out.end(), 0);
  thread_scan_exclusive_partial_kernel_span<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    cuda::std::plus<>{},
    valid_items,
    0,
    true);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == d_out);

#if _CCCL_STD_VER >= 2023
  thrust::fill(d_out.begin(), d_out.end(), 0);
  thread_scan_exclusive_partial_kernel_mdspan<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    cuda::std::plus<>{},
    valid_items,
    0,
    true);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(reference_result == d_out);
#endif // _CCCL_STD_VER >= 2023
}

C2H_TEST("ThreadScanExclusive Invalid Test", "[scan][thread]")
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
  const bool apply_prefix       = GENERATE(true, false);
  // Invalid prefix when !apply_prefix
  const segment prefix{0, apply_prefix ? 1 : 0};
  CAPTURE(valid_items, prefix, apply_prefix);
  // First initialize with invalid segments than overwrite the first valid_items
  c2h::host_vector<segment> h_in(max_size);
  thrust::copy(in_it, in_it + bounded_valid_items, h_in.begin());
  c2h::device_vector<segment> d_in = h_in;
  c2h::host_vector<segment> reference_result(max_size);
  compute_exclusive_scan_reference(
    h_in.cbegin(),
    h_in.cbegin() + bounded_valid_items,
    reference_result.begin(),
    apply_prefix ? prefix : segment{1, 1},
    merge_segments_op{nullptr});

  c2h::device_vector<segment> d_out(max_size);
  c2h::device_vector<bool> error_flag(1, false);
  thread_scan_exclusive_partial_kernel<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    merge_segments_op{thrust::raw_pointer_cast(error_flag.data())},
    valid_items,
    prefix,
    apply_prefix,
    segment{});
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(error_flag.front() == false);
  if (!apply_prefix && valid_items > 0)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}
