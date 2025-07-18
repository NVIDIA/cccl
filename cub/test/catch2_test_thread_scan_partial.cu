

// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/util_macro.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <cstring>
#include <functional>
#include <numeric>

#include "c2h/catch2_test_helper.h"
#include "c2h/extended_types.h"
#include "c2h/generators.h"
#include "catch2_test_device_reduce.cuh"
#include "catch2_test_device_scan.cuh"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

constexpr int max_size  = 16;
constexpr int num_seeds = 3;

/***********************************************************************************************************************
 * Thread Scan Wrapper Kernels
 **********************************************************************************************************************/

template <int NUM_ITEMS, typename In, typename Out, typename Accum, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel(
  In d_in, Out d_out, ScanOperator scan_operator, int valid_items, Accum prefix, bool apply_prefix)
{
  using value_t  = ::cuda::std::iter_value_t<In>;
  using output_t = ::cuda::std::iter_value_t<Out>;
  value_t thread_input[NUM_ITEMS];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_input[i] = d_in[i];
  }
  output_t thread_output[NUM_ITEMS];
  cub::detail::ThreadScanExclusivePartial(thread_input, thread_output, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_output[i];
  }
}

template <int NUM_ITEMS, typename In, typename Out, typename Accum, typename ScanOperator>
__global__ void thread_scan_inclusive_partial_kernel(
  In d_in, Out d_out, ScanOperator scan_operator, int valid_items, Accum prefix, bool apply_prefix)
{
  using value_t  = ::cuda::std::iter_value_t<In>;
  using output_t = ::cuda::std::iter_value_t<Out>;
  value_t thread_input[NUM_ITEMS];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_input[i] = d_in[i];
  }
  output_t thread_output[NUM_ITEMS];
  cub::detail::ThreadScanInclusivePartial(thread_input, thread_output, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_output[i];
  }
}

// The following kernels are less general/complex wuith the added benefit that we can test doing the scan in-place

template <int NUM_ITEMS, typename T, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel_array(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  ::cuda::std::array<T, NUM_ITEMS> thread_data;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  cub::detail::ThreadScanExclusivePartial(thread_data, thread_data, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

template <int NUM_ITEMS, typename T, typename ScanOperator>
__global__ void thread_scan_inclusive_partial_kernel_array(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  ::cuda::std::array<T, NUM_ITEMS> thread_data;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  cub::detail::ThreadScanInclusivePartial(thread_data, thread_data, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

template <int NUM_ITEMS, typename T, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel_span(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  ::cuda::std::span<T, NUM_ITEMS> span(thread_data);
  cub::detail::ThreadScanExclusivePartial(span, span, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

template <int NUM_ITEMS, typename T, typename ScanOperator>
__global__ void thread_scan_inclusive_partial_kernel_span(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  ::cuda::std::span<T, NUM_ITEMS> span(thread_data);
  cub::detail::ThreadScanInclusivePartial(span, span, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

#if _CCCL_STD_VER >= 2023

template <int NUM_ITEMS, typename T, typename ScanOperator>
__global__ void thread_scan_exclusive_partial_kernel_mdspan(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  using Extent = ::cuda::std::extents<int, NUM_ITEMS>;
  ::cuda::std::mdspan<T, Extent> mdspan(thread_data, ::cuda::std::extents<int, NUM_ITEMS>{});
  cub::detail::ThreadScanExclusivePartial(mdspan, mdspan, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

template <int NUM_ITEMS, typename T, typename ScanOperator>
__global__ void thread_scan_inclusive_partial_kernel_mdspan(
  const T* d_in, T* d_out, ScanOperator scan_operator, int valid_items, T prefix, bool apply_prefix)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  using Extent = ::cuda::std::extents<int, NUM_ITEMS>;
  ::cuda::std::mdspan<T, Extent> mdspan(thread_data, ::cuda::std::extents<int, NUM_ITEMS>{});
  cub::detail::ThreadScanInclusivePartial(mdspan, mdspan, scan_operator, valid_items, prefix, apply_prefix);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    d_out[i] = thread_data[i];
  }
}

#endif // _CCCL_STD_VER >= 2023

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename T, typename Operator, typename = void>
struct cub_operator_to_identity;

template <typename T>
struct cub_operator_to_identity<T, cuda::std::plus<>>
{
  static constexpr T value()
  {
    return T{};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::std::multiplies<>>
{
  static constexpr T value()
  {
    return T{1};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::std::bit_and<>>
{
  static constexpr T value()
  {
    return static_cast<T>(~T{0});
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::std::bit_or<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::std::bit_xor<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::minimum<>>
{
  static constexpr T value()
  {
    return ::cuda::std::numeric_limits<T>::max();
  }
};

template <typename T>
struct cub_operator_to_identity<T, cuda::maximum<>>
{
  static constexpr T value()
  {
    return ::cuda::std::numeric_limits<T>::lowest();
  }
};

namespace detail
{

template <typename T, typename Operator, typename = void>
struct dist_interval
{
  static constexpr T min()
  {
    return ::cuda::std::numeric_limits<T>::lowest();
  }
  static constexpr T max()
  {
    return ::cuda::std::numeric_limits<T>::max();
  }
};

template <typename T>
struct dist_interval<T, cuda::std::plus<>, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_signed_integer_v<T>>>
{
  // Avoid possibility of over-/underflow causing UB
  static constexpr T min()
  {
    return ::cuda::std::numeric_limits<T>::min() / max_size;
  }
  static constexpr T max()
  {
    return ::cuda::std::numeric_limits<T>::max() / max_size;
  }
};

template <typename T>
struct dist_interval<T, cuda::std::multiplies<>, ::cuda::std::enable_if_t<::cuda::std::__cccl_is_signed_integer_v<T>>>
{
  // Avoid possibility of over-/underflow causing UB.
  // Use floating point arithmetic to avoid unnecessarily small interval.
  static constexpr T min()
  {
    const double log2_abs_min = ::cuda::std::log2(::cuda::std::abs(::cuda::std::numeric_limits<T>::min()));
    return -::cuda::std::exp2(log2_abs_min / max_size);
  }
  static constexpr T max()
  {
    const double log2_max = ::cuda::std::log2(::cuda::std::numeric_limits<T>::max());
    return ::cuda::std::exp2(log2_max / max_size);
  }
};

} // namespace detail

template <typename Input,
          typename Output,
          typename Operator,
          typename Accum = ::cuda::std::__accumulator_t<Operator, Input>>
struct dist_interval
{
  // Values in the interval need to be representable in Input and if either Output or Accum are signed integers we want
  // to avoid UB.
  static constexpr Input min()
  {
    auto res = ::cuda::std::numeric_limits<Input>::lowest();
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Output>)
    {
      res = ::cuda::std::max(res, static_cast<Input>(detail::dist_interval<Output, Operator>::min()));
    }
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Accum>)
    {
      res = ::cuda::std::max(res, static_cast<Input>(detail::dist_interval<Accum, Operator>::min()));
    }
    return res;
  }
  static constexpr Input max()
  {
    auto res = ::cuda::std::numeric_limits<Input>::max();
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Output>)
    {
      res = ::cuda::std::min(res, static_cast<Input>(detail::dist_interval<Output, Operator>::max()));
    }
    if constexpr (cuda::std::__cccl_is_signed_integer_v<Accum>)
    {
      res = ::cuda::std::min(res, static_cast<Input>(detail::dist_interval<Accum, Operator>::max()));
    }
    return res;
  }
};

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
  c2h::type_list<type_pair<::cuda::std::int8_t>,
                 type_pair<::cuda::std::uint8_t, ::cuda::std::int32_t>,
                 type_pair<::cuda::std::int16_t>,
                 type_pair<::cuda::std::uint16_t>,
                 type_pair<::cuda::std::int32_t>,
                 type_pair<::cuda::std::int64_t>>;

using fp_type_list = c2h::type_list<type_pair<float>, type_pair<double>>;

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
 * Kernel launch
 **********************************************************************************************************************/

template <typename ValueT,
          typename OutputT,
          typename ScanOperator,
          typename AccumT = ::cuda::std::__accumulator_t<ScanOperator, ValueT>>
void run_thread_scan_exclusive_partial_kernel(
  int num_items,
  c2h::device_vector<ValueT>& in,
  c2h::device_vector<OutputT>& out,
  ScanOperator scan_operator,
  int valid_items,
  AccumT prefix,
  bool apply_prefix)
{
  const auto d_in  = unwrap_it(thrust::raw_pointer_cast(in.data()));
  const auto d_out = unwrap_it(thrust::raw_pointer_cast(out.data()));
  switch (num_items)
  {
    case 1:
      thread_scan_exclusive_partial_kernel<1><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 2:
      thread_scan_exclusive_partial_kernel<2><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 3:
      thread_scan_exclusive_partial_kernel<3><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 4:
      thread_scan_exclusive_partial_kernel<4><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 5:
      thread_scan_exclusive_partial_kernel<5><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 6:
      thread_scan_exclusive_partial_kernel<6><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 7:
      thread_scan_exclusive_partial_kernel<7><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 8:
      thread_scan_exclusive_partial_kernel<8><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 9:
      thread_scan_exclusive_partial_kernel<9><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 10:
      thread_scan_exclusive_partial_kernel<10><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 11:
      thread_scan_exclusive_partial_kernel<11><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 12:
      thread_scan_exclusive_partial_kernel<12><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 13:
      thread_scan_exclusive_partial_kernel<13><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 14:
      thread_scan_exclusive_partial_kernel<14><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 15:
      thread_scan_exclusive_partial_kernel<15><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 16:
      thread_scan_exclusive_partial_kernel<16><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    default:
      FAIL("Unsupported number of items");
  }
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <typename ValueT,
          typename OutputT,
          typename ScanOperator,
          typename AccumT = ::cuda::std::__accumulator_t<ScanOperator, ValueT>>
void run_thread_scan_inclusive_partial_kernel(
  int num_items,
  c2h::device_vector<ValueT>& in,
  c2h::device_vector<OutputT>& out,
  ScanOperator scan_operator,
  int valid_items,
  AccumT prefix,
  bool apply_prefix)
{
  const auto d_in  = unwrap_it(thrust::raw_pointer_cast(in.data()));
  const auto d_out = unwrap_it(thrust::raw_pointer_cast(out.data()));
  switch (num_items)
  {
    case 1:
      thread_scan_inclusive_partial_kernel<1><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 2:
      thread_scan_inclusive_partial_kernel<2><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 3:
      thread_scan_inclusive_partial_kernel<3><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 4:
      thread_scan_inclusive_partial_kernel<4><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 5:
      thread_scan_inclusive_partial_kernel<5><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 6:
      thread_scan_inclusive_partial_kernel<6><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 7:
      thread_scan_inclusive_partial_kernel<7><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 8:
      thread_scan_inclusive_partial_kernel<8><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 9:
      thread_scan_inclusive_partial_kernel<9><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 10:
      thread_scan_inclusive_partial_kernel<10><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 11:
      thread_scan_inclusive_partial_kernel<11><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 12:
      thread_scan_inclusive_partial_kernel<12><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 13:
      thread_scan_inclusive_partial_kernel<13><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 14:
      thread_scan_inclusive_partial_kernel<14><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 15:
      thread_scan_inclusive_partial_kernel<15><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    case 16:
      thread_scan_inclusive_partial_kernel<16><<<1, 1>>>(d_in, d_out, scan_operator, valid_items, prefix, apply_prefix);
      break;
    default:
      FAIL("Unsupported number of items");
  }
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("ThreadScanExclusive Integral Type Tests", "[scan][thread]", integral_type_list, cub_operator_integral_list)
{
  using params                     = params_t<TestType>;
  using value_t                    = typename params::item_t;
  using output_t                   = typename params::output_t;
  using op_t                       = c2h::get<1, TestType>;
  using accum_t                    = ::cuda::std::__accumulator_t<op_t, value_t>;
  using dist_param                 = dist_interval<value_t, output_t, op_t>;
  constexpr auto scan_op           = op_t{};
  constexpr auto operator_identity = cub_operator_to_identity<accum_t, op_t>::value();
  const int num_items              = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items            = GENERATE_COPY(
    take(1, random(2, num_items - 2)),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items - 1, num_items, num_items + 1}));
  const accum_t prefix    = GENERATE_COPY(take(1, random(dist_param::min(), dist_param::max())));
  const bool apply_prefix = GENERATE(true, false);
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix);
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<output_t> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(max_size);
  compute_exclusive_scan_reference(
    h_in.cbegin(), h_in.cend(), reference_result.begin(), apply_prefix ? prefix : operator_identity, scan_op);
  run_thread_scan_exclusive_partial_kernel(num_items, d_in, d_out, scan_op, valid_items, prefix, apply_prefix);
  // Resize to allow using REQUIRE() directly
  const int bounded_valid_items = std::min(valid_items, num_items);
  d_out.resize(bounded_valid_items);
  reference_result.resize(bounded_valid_items);
  if (!apply_prefix)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

C2H_TEST("ThreadScanInclusive Integral Type Tests", "[scan][thread]", integral_type_list, cub_operator_integral_list)
{
  using params                     = params_t<TestType>;
  using value_t                    = typename params::item_t;
  using output_t                   = typename params::output_t;
  using op_t                       = c2h::get<1, TestType>;
  using accum_t                    = ::cuda::std::__accumulator_t<op_t, value_t>;
  using dist_param                 = dist_interval<value_t, output_t, op_t>;
  constexpr auto scan_op           = op_t{};
  constexpr auto operator_identity = cub_operator_to_identity<accum_t, op_t>::value();
  const int num_items              = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items            = GENERATE_COPY(
    take(1, random(2, num_items - 2)),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items - 1, num_items, num_items + 1}));
  const accum_t prefix    = GENERATE_COPY(take(1, random(dist_param::min(), dist_param::max())));
  const bool apply_prefix = GENERATE(true, false);
  CAPTURE(c2h::type_name<value_t>(),
          c2h::type_name<output_t>(),
          c2h::type_name<accum_t>(),
          num_items,
          c2h::type_name<op_t>(),
          valid_items,
          operator_identity,
          prefix,
          apply_prefix);
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<output_t> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(max_size);
  compute_inclusive_scan_reference(
    h_in.cbegin(), h_in.cend(), reference_result.begin(), scan_op, apply_prefix ? prefix : operator_identity);
  run_thread_scan_inclusive_partial_kernel(num_items, d_in, d_out, scan_op, valid_items, prefix, apply_prefix);
  // Resize to allow using REQUIRE() directly
  const int bounded_valid_items = std::min(valid_items, num_items);
  d_out.resize(bounded_valid_items);
  reference_result.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
}

C2H_TEST("ThreadScanExclusive Floating-Point Type Tests", "[scan][thread]", fp_type_list, cub_operator_fp_list)
{
  using params                 = params_t<TestType>;
  using value_t                = typename params::item_t;
  using output_t               = typename params::output_t;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = ::cuda::std::__accumulator_t<op_t, value_t>;
  constexpr auto scan_op       = op_t{};
  const auto operator_identity = cub_operator_to_identity<accum_t, op_t>::value();
  const int num_items          = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, num_items - 2)),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items - 1, num_items, num_items + 1}));
  const accum_t prefix = GENERATE_COPY(
    take(1, random(::cuda::std::numeric_limits<value_t>::lowest(), ::cuda::std::numeric_limits<value_t>::max())));
  const bool apply_prefix = GENERATE(true, false);
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix);
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<value_t> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in);
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<value_t> reference_result(max_size);
  compute_exclusive_scan_reference(
    h_in.cbegin(), h_in.cend(), reference_result.begin(), apply_prefix ? prefix : operator_identity, scan_op);
  run_thread_scan_exclusive_partial_kernel(num_items, d_in, d_out, scan_op, valid_items, prefix, apply_prefix);
  // Resize to allow using REQUIRE() directly
  const int bounded_valid_items = std::min(valid_items, num_items);
  d_out.resize(bounded_valid_items);
  reference_result.resize(bounded_valid_items);
  if (!apply_prefix)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

C2H_TEST("ThreadScanInclusive Floating-Point Type Tests", "[scan][thread]", fp_type_list, cub_operator_fp_list)
{
  using params                 = params_t<TestType>;
  using value_t                = typename params::item_t;
  using output_t               = typename params::output_t;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = ::cuda::std::__accumulator_t<op_t, value_t>;
  constexpr auto scan_op       = op_t{};
  const auto operator_identity = cub_operator_to_identity<accum_t, op_t>::value();
  const int num_items          = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, num_items - 2)),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items - 1, num_items, num_items + 1}));
  const accum_t prefix = GENERATE_COPY(
    take(1, random(::cuda::std::numeric_limits<value_t>::lowest(), ::cuda::std::numeric_limits<value_t>::max())));
  const bool apply_prefix = GENERATE(true, false);
  CAPTURE(c2h::type_name<value_t>(),
          c2h::type_name<output_t>(),
          c2h::type_name<accum_t>(),
          num_items,
          c2h::type_name<op_t>(),
          valid_items,
          operator_identity,
          prefix,
          apply_prefix);
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<output_t> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in);
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(max_size);
  compute_inclusive_scan_reference(
    h_in.cbegin(), h_in.cend(), reference_result.begin(), scan_op, apply_prefix ? prefix : operator_identity);
  run_thread_scan_inclusive_partial_kernel(num_items, d_in, d_out, scan_op, valid_items, prefix, apply_prefix);
  // Resize to allow using REQUIRE() directly
  const int bounded_valid_items = std::min(valid_items, num_items);
  d_out.resize(bounded_valid_items);
  reference_result.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
}

#if TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadScanExclusive Narrow PrecisionType Tests",
         "[scan][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list)
{
  using params                 = params_t<TestType>;
  using value_t                = typename params::item_t;
  using output_t               = typename params::output_t;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = ::cuda::std::__accumulator_t<op_t, value_t>;
  constexpr auto scan_op       = unwrap_op(std::true_type{}, op_t{});
  const auto operator_identity = cub_operator_to_identity<accum_t, op_t>::value();
  const int num_items          = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, num_items - 2)),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items - 1, num_items, num_items + 1}));
  auto prefix             = static_cast<accum_t>(GENERATE_COPY(take(1, random(1.0f, 2.0f))));
  const bool apply_prefix = GENERATE(true, false);
  CAPTURE(
    c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix, operator_identity);
  c2h::device_vector<value_t> d_in(max_size);
  c2h::device_vector<output_t> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in, value_t{1.0f}, value_t{2.0f});
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(max_size);
  compute_exclusive_scan_reference(
    h_in.cbegin(), h_in.cend(), reference_result.begin(), apply_prefix ? prefix : operator_identity, scan_op);
  run_thread_scan_exclusive_partial_kernel(
    num_items, d_in, d_out, scan_op, valid_items, *unwrap_it(&prefix), apply_prefix);
  // Resize to allow using REQUIRE() directly
  const int bounded_valid_items = std::min(valid_items, num_items);
  d_out.resize(bounded_valid_items);
  reference_result.resize(bounded_valid_items);
  if (!apply_prefix)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

C2H_TEST("ThreadScanInclusive Narrow PrecisionType Tests",
         "[scan][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list)
{
  using params                 = params_t<TestType>;
  using value_t                = typename params::item_t;
  using output_t               = typename params::output_t;
  using op_t                   = c2h::get<1, TestType>;
  using accum_t                = ::cuda::std::__accumulator_t<op_t, value_t>;
  constexpr auto scan_op       = unwrap_op(std::true_type{}, op_t{});
  const auto operator_identity = cub_operator_to_identity<accum_t, op_t>::value();
  const int num_items          = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, num_items - 2)),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items - 1, num_items, num_items + 1}));
  auto prefix             = static_cast<accum_t>(GENERATE_COPY(take(1, random(1.0f, 2.0f))));
  const bool apply_prefix = GENERATE(true, false);
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<op_t>(), valid_items, prefix, apply_prefix);
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<output_t> d_out(num_items);
  c2h::gen(C2H_SEED(num_seeds), d_in, value_t{1.0f}, value_t{2.0f});
  c2h::host_vector<value_t> h_in = d_in;
  c2h::host_vector<output_t> reference_result(num_items);
  compute_inclusive_scan_reference(
    h_in.cbegin(), h_in.cend(), reference_result.begin(), scan_op, apply_prefix ? prefix : operator_identity);
  run_thread_scan_inclusive_partial_kernel(
    num_items, d_in, d_out, scan_op, valid_items, *unwrap_it(&prefix), apply_prefix);
  // Resize to allow using REQUIRE() directly
  const int bounded_valid_items = std::min(valid_items, num_items);
  d_out.resize(bounded_valid_items);
  reference_result.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
}

#endif // TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadScamExclusive Container Tests", "[scan][thread]")
{
  c2h::device_vector<int> d_in(max_size);
  c2h::device_vector<int> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in);
  c2h::host_vector<int> h_in = d_in;
  const int valid_items      = GENERATE_COPY(
    take(1, random(2, max_size - 2)),
    take(1, random(max_size + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, max_size - 1, max_size, max_size + 1}));
  const int bounded_valid_items = cuda::std::min(valid_items, max_size);
  c2h::host_vector<int> reference_result(bounded_valid_items);
  std::exclusive_scan(h_in.begin(), h_in.begin() + bounded_valid_items, reference_result.begin(), 0, std::plus<int>{});
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
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
  thrust::fill(d_out.begin(), d_out.end(), 0);

  d_out.resize(max_size);
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
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);

#if _CCCL_STD_VER >= 2023
  d_out.resize(max_size);
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
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
#endif // _CCCL_STD_VER >= 2023
}

C2H_TEST("ThreadScamInclusive Container Tests", "[scan][thread]")
{
  c2h::device_vector<int> d_in(max_size);
  c2h::device_vector<int> d_out(max_size);
  c2h::gen(C2H_SEED(num_seeds), d_in);
  c2h::host_vector<int> h_in = d_in;
  const int valid_items      = GENERATE_COPY(
    take(1, random(2, max_size - 2)),
    take(1, random(max_size + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, max_size - 1, max_size, max_size + 1}));
  const int bounded_valid_items = cuda::std::min(valid_items, max_size);
  c2h::host_vector<int> reference_result(bounded_valid_items);
  compute_inclusive_scan_reference(
    h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reference_result.begin(), cuda::std::plus<>{}, 0);

  thread_scan_inclusive_partial_kernel_array<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    cuda::std::plus<>{},
    valid_items,
    0,
    true);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
  thrust::fill(d_out.begin(), d_out.end(), 0);

  d_out.resize(max_size);
  thrust::fill(d_out.begin(), d_out.end(), 0);
  thread_scan_inclusive_partial_kernel_span<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    cuda::std::plus<>{},
    valid_items,
    0,
    true);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);

#if _CCCL_STD_VER >= 2023
  d_out.resize(max_size);
  thrust::fill(d_out.begin(), d_out.end(), 0);
  thread_scan_inclusive_partial_kernel_mdspan<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    cuda::std::plus<>{},
    valid_items,
    0,
    true);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
#endif // _CCCL_STD_VER >= 2023
}

struct segment
{
  using offset_t = int32_t;
  // Make sure that default constructed segments can not be merged
  offset_t begin = cuda::std::numeric_limits<offset_t>::min();
  offset_t end   = cuda::std::numeric_limits<offset_t>::max();

  __host__ __device__ friend bool operator==(segment left, segment right)
  {
    return left.begin == right.begin && left.end == right.end;
  }

  // Needed for final comparison with reference
  friend std::ostream& operator<<(std::ostream& os, const segment& seg)
  {
    return os << "[ " << seg.begin << ", " << seg.end << " )";
  }
};

// Needed for data input using fancy iterators
struct tuple_to_segment_op
{
  __host__ __device__ segment operator()(cuda::std::tuple<segment::offset_t, segment::offset_t> interval)
  {
    const auto [begin, end] = interval;
    return {begin, end};
  }
};

// Actual scan operator doing the core test when run on device
struct merge_segments_op
{
  __host__ merge_segments_op(bool* error_flag_ptr)
      : error_flag_ptr_{error_flag_ptr}
  {}

  __device__ void check_inputs(segment left, segment right)
  {
    if (left.end != right.begin || left == right)
    {
      *error_flag_ptr_ = true;
    }
  }

  __host__ __device__ segment operator()(segment left, segment right)
  {
    NV_IF_TARGET(NV_IS_DEVICE, check_inputs(left, right););
    return {left.begin, right.end};
  }

  bool* error_flag_ptr_;
};

C2H_TEST("ThreadScamExclusive Invalid Test", "[scan][thread]")
{
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  const int valid_items = GENERATE_COPY(
    take(3, random(2, max_size - 1)),
    take(1, random(max_size + 1, ::cuda::std::numeric_limits<int>::max())),
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
  c2h::host_vector<segment> reference_result(bounded_valid_items);
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
    apply_prefix);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(error_flag.front() == false);
  d_out.resize(bounded_valid_items);
  if (!apply_prefix && valid_items > 0)
  {
    // Undefined for exclusive scan
    reference_result.front() = d_out.front();
  }
  REQUIRE(reference_result == d_out);
}

C2H_TEST("ThreadScamInclusive Invalid Test", "[scan][thread]")
{
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  const int valid_items = GENERATE_COPY(
    take(3, random(2, max_size - 1)),
    take(1, random(max_size + 1, ::cuda::std::numeric_limits<int>::max())),
    values({-1, 0, 1}));
  const int bounded_valid_items = cuda::std::clamp(valid_items, 0, max_size);
  const bool apply_prefix       = GENERATE(true, false);
  // Invalid prefix when not apply_prefix
  const segment prefix{0, apply_prefix ? 1 : 0};
  CAPTURE(valid_items, prefix, apply_prefix);
  // First initialize with invalid segments than overwrite the first valid_items
  c2h::host_vector<segment> h_in(max_size);
  thrust::copy(in_it, in_it + bounded_valid_items, h_in.begin());
  c2h::device_vector<segment> d_in = h_in;
  c2h::host_vector<segment> reference_result(bounded_valid_items);
  compute_inclusive_scan_reference(
    h_in.cbegin(),
    h_in.cbegin() + bounded_valid_items,
    reference_result.begin(),
    merge_segments_op{nullptr},
    apply_prefix ? prefix : segment{1, 1});

  c2h::device_vector<segment> d_out(max_size);
  c2h::device_vector<bool> error_flag(1, false);
  thread_scan_inclusive_partial_kernel<max_size><<<1, 1>>>(
    thrust::raw_pointer_cast(d_in.data()),
    thrust::raw_pointer_cast(d_out.data()),
    merge_segments_op{thrust::raw_pointer_cast(error_flag.data())},
    valid_items,
    prefix,
    apply_prefix);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  REQUIRE(error_flag.front() == false);
  d_out.resize(bounded_valid_items);
  REQUIRE(reference_result == d_out);
}
