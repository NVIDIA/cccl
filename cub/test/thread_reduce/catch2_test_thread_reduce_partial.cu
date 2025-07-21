
// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/type_traits.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/util_macro.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/std/__algorithm/clamp.h>
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
#include "catch2_test_device_reduce.cuh"
#include <catch2/matchers/catch_matchers_floating_point.hpp>

constexpr int max_size  = 16;
constexpr int num_seeds = 3;

/***********************************************************************************************************************
 * Thread Reduce Wrapper Kernels
 **********************************************************************************************************************/

template <int NUM_ITEMS, typename In, typename Out, typename ReduceOperator>
__global__ void thread_reduce_partial_kernel(In d_in, Out d_out, ReduceOperator reduce_operator, int valid_items)
{
  using value_t = ::cuda::std::iter_value_t<In>;
  value_t thread_data[NUM_ITEMS];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::detail::ThreadReducePartial(thread_data, reduce_operator, valid_items);
}

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void
thread_reduce_partial_kernel_array(const T* d_in, T* d_out, ReduceOperator reduce_operator, int valid_items)
{
  ::cuda::std::array<T, NUM_ITEMS> thread_data;

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  *d_out = cub::detail::ThreadReducePartial(thread_data, reduce_operator, valid_items);
}

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void
thread_reduce_partial_kernel_span(const T* d_in, T* d_out, ReduceOperator reduce_operator, int valid_items)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  ::cuda::std::span<T, NUM_ITEMS> span(thread_data);
  *d_out = cub::detail::ThreadReducePartial(span, reduce_operator, valid_items);
}

#if _CCCL_STD_VER >= 2023

template <int NUM_ITEMS, typename T, typename ReduceOperator>
__global__ void
thread_reduce_partial_kernel_mdspan(const T* d_in, T* d_out, ReduceOperator reduce_operator, int valid_items)
{
  T thread_data[NUM_ITEMS];

  _CCCL_PRAGMA_UNROLL_FULL()
  for (int i = 0; i < NUM_ITEMS; ++i)
  {
    thread_data[i] = d_in[i];
  }
  using Extent = ::cuda::std::extents<int, NUM_ITEMS>;
  ::cuda::std::mdspan<T, Extent> mdspan(thread_data, ::cuda::std::extents<int, NUM_ITEMS>{});
  *d_out = cub::detail::ThreadReducePartial(mdspan, reduce_operator, valid_items);
}

#endif // _CCCL_STD_VER >= 2023

/***********************************************************************************************************************
 * CUB operator to STD operator
 **********************************************************************************************************************/

template <typename T, typename>
struct cub_operator_to_std;

template <typename T>
struct cub_operator_to_std<T, ::cuda::std::plus<>>
{
  using type = ::std::plus<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, ::cuda::std::multiplies<>>
{
  using type = ::std::multiplies<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, ::cuda::std::bit_and<>>
{
  using type = ::std::bit_and<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, ::cuda::std::bit_or<>>
{
  using type = ::std::bit_or<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, ::cuda::std::bit_xor<>>
{
  using type = ::std::bit_xor<T>; // T: MSVC complains about possible loss of data
};

template <typename T>
struct cub_operator_to_std<T, ::cuda::minimum<>>
{
  using type = ::cuda::minimum<>;
};

template <typename T>
struct cub_operator_to_std<T, ::cuda::maximum<>>
{
  using type = ::cuda::maximum<>;
};

template <typename T, typename Operator>
using cub_operator_to_std_t = typename cub_operator_to_std<T, Operator>::type;

/***********************************************************************************************************************
 * CUB operator to identity
 **********************************************************************************************************************/

template <typename T, typename Operator, typename = void>
struct cub_operator_to_identity;

template <typename T>
struct cub_operator_to_identity<T, ::cuda::std::plus<>>
{
  static constexpr T value()
  {
    return T{};
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::std::multiplies<>>
{
  static constexpr T value()
  {
    return T{1};
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::std::bit_and<>>
{
  static constexpr T value()
  {
    return static_cast<T>(~T{0});
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::std::bit_or<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::std::bit_xor<>>
{
  static constexpr T value()
  {
    return T{0};
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::minimum<>>
{
  static constexpr T value()
  {
    return ::cuda::std::numeric_limits<T>::max();
  }
};

template <typename T>
struct cub_operator_to_identity<T, ::cuda::maximum<>>
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
struct dist_interval<
  T,
  ::cuda::std::plus<>,
  ::cuda::std::enable_if_t<::cuda::std::__cccl_is_signed_integer_v<T> || ::cuda::std::is_floating_point_v<T>>>
{
  // signed_integer: Avoid possibility of over-/underflow causing UB
  // floating_point: Avoid possibility of over-/underflow causing inf breaking approximate associativity
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
struct dist_interval<
  T,
  ::cuda::std::multiplies<>,
  ::cuda::std::enable_if_t<::cuda::std::__cccl_is_signed_integer_v<T> || ::cuda::std::is_floating_point_v<T>>>
{
  // signed_integer: Avoid possibility of over-/underflow causing UB
  // floating_point: Avoid possibility of over-/underflow causing inf breaking approximate associativity
  // Use floating point arithmetic to avoid unnecessarily small interval.
  static constexpr T min()
  {
    const double log2_abs_min = ::cuda::std::log2(::cuda::std::fabs(::cuda::std::numeric_limits<T>::min()));
    return static_cast<T>(-::cuda::std::exp2(log2_abs_min / max_size));
  }
  static constexpr T max()
  {
    const double log2_max = ::cuda::std::log2(::cuda::std::numeric_limits<T>::max());
    return static_cast<T>(::cuda::std::exp2(log2_max / max_size));
  }
};

} // namespace detail

template <typename Input, typename Operator, typename Accum = ::cuda::std::__accumulator_t<Operator, Input>>
struct dist_interval
{
  // Values in the interval need to be representable in Input and if either Output or Accum are signed integers we want
  // to avoid UB.
  static constexpr Input min()
  {
    auto res = ::cuda::std::numeric_limits<Input>::lowest();
    if constexpr (::cuda::std::__cccl_is_signed_integer_v<Accum> || ::cuda::std::is_floating_point_v<Accum>)
    {
      res = ::cuda::std::max(res, static_cast<Input>(detail::dist_interval<Accum, Operator>::min()));
    }
    return res;
  }
  static constexpr Input max()
  {
    auto res = ::cuda::std::numeric_limits<Input>::max();
    if constexpr (::cuda::std::__cccl_is_signed_integer_v<Accum> || ::cuda::std::is_floating_point_v<Accum>)
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
  half_t,
#endif // TEST_HALF_T()
#if TEST_BF_T()
  bfloat16_t
#endif // TEST_BF_T()
  >;

using integral_type_list = c2h::
  type_list<::cuda::std::int8_t, ::cuda::std::int16_t, ::cuda::std::uint16_t, ::cuda::std::int32_t, ::cuda::std::int64_t>;

using fp_type_list = c2h::type_list<float, double>;

using cub_operator_integral_list =
  c2h::type_list<::cuda::std::plus<>,
                 ::cuda::std::multiplies<>,
                 ::cuda::std::bit_and<>,
                 ::cuda::std::bit_or<>,
                 ::cuda::std::bit_xor<>,
                 ::cuda::minimum<>,
                 ::cuda::maximum<>>;

using cub_operator_fp_list =
  c2h::type_list<::cuda::std::plus<>, ::cuda::std::multiplies<>, ::cuda::minimum<>, ::cuda::maximum<>>;

/***********************************************************************************************************************
 * Verify results and kernel launch
 **********************************************************************************************************************/

_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES((::cuda::std::is_floating_point_v<T>) )
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE_THAT(expected_data, Catch::Matchers::WithinRel(test_results, T{0.05}));
}

_CCCL_TEMPLATE(typename T)
_CCCL_REQUIRES((!::cuda::std::is_floating_point_v<T>) )
void verify_results(const T& expected_data, const T& test_results)
{
  REQUIRE(expected_data == test_results);
}

template <typename T, typename ReduceOperator>
void run_thread_reduce_partial_kernel(
  int num_items, c2h::device_vector<T>& in, c2h::device_vector<T>& out, ReduceOperator reduce_operator, int valid_items)
{
  auto const in_it  = unwrap_it(thrust::raw_pointer_cast(in.data()));
  auto const out_it = unwrap_it(thrust::raw_pointer_cast(out.data()));
  switch (num_items)
  {
    case 1:
      thread_reduce_partial_kernel<1><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 2:
      thread_reduce_partial_kernel<2><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 3:
      thread_reduce_partial_kernel<3><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 4:
      thread_reduce_partial_kernel<4><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 5:
      thread_reduce_partial_kernel<5><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 6:
      thread_reduce_partial_kernel<6><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 7:
      thread_reduce_partial_kernel<7><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 8:
      thread_reduce_partial_kernel<8><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 9:
      thread_reduce_partial_kernel<9><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 10:
      thread_reduce_partial_kernel<10><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 11:
      thread_reduce_partial_kernel<11><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 12:
      thread_reduce_partial_kernel<12><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 13:
      thread_reduce_partial_kernel<13><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 14:
      thread_reduce_partial_kernel<14><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 15:
      thread_reduce_partial_kernel<15><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
      break;
    case 16:
      thread_reduce_partial_kernel<16><<<1, 1>>>(in_it, out_it, reduce_operator, valid_items);
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

C2H_TEST("ThreadReduce Integral Type Tests", "[reduce][thread]", integral_type_list, cub_operator_integral_list)
{
  using value_t                    = c2h::get<0, TestType>;
  using op_t                       = c2h::get<1, TestType>;
  using dist_param                 = dist_interval<value_t, op_t>;
  constexpr auto reduce_op         = op_t{};
  constexpr auto operator_identity = cub_operator_to_identity<value_t, op_t>::value();
  const int num_items              = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items            = GENERATE_COPY(
    take(1, random(2, ::cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<decltype(reduce_op)>(), valid_items);
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  const int bounded_valid_items  = ::cuda::std::min(valid_items, num_items);
  auto reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reduce_op, operator_identity);
  run_thread_reduce_partial_kernel(num_items, d_in, d_out, reduce_op, valid_items);
  verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
}

C2H_TEST("ThreadReduce Floating-Point Type Tests", "[reduce][thread]", fp_type_list, cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  using dist_param             = dist_interval<value_t, op_t>;
  constexpr auto reduce_op     = op_t{};
  const auto operator_identity = cub_operator_to_identity<value_t, op_t>::value();
  const int num_items          = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, ::cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  CAPTURE(c2h::type_name<value_t>(), num_items, c2h::type_name<decltype(reduce_op)>(), valid_items);
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  const int bounded_valid_items  = ::cuda::std::min(valid_items, num_items);
  auto reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reduce_op, operator_identity);
  run_thread_reduce_partial_kernel(num_items, d_in, d_out, reduce_op, valid_items);
  verify_results(reference_result, c2h::host_vector<value_t>(d_out)[0]);
}

#if TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Narrow PrecisionType Tests",
         "[reduce][thread][narrow]",
         narrow_precision_type_list,
         cub_operator_fp_list)
{
  using value_t                = c2h::get<0, TestType>;
  using op_t                   = c2h::get<1, TestType>;
  using dist_param             = dist_interval<value_t, op_t>;
  constexpr auto reduce_op     = unwrap_op(std::true_type{}, op_t{});
  const auto operator_identity = cub_operator_to_identity<value_t, op_t>::value();
  const int num_items          = GENERATE_COPY(take(3, random(1, max_size)));
  const int valid_items        = GENERATE_COPY(
    take(1, random(2, ::cuda::std::max(2, num_items - 1))),
    take(1, random(num_items + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, num_items, num_items + 1}));
  c2h::device_vector<value_t> d_in(num_items);
  c2h::device_vector<value_t> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<value_t> h_in = d_in;
  CAPTURE(h_in, dist_param::min(), dist_param::max());
  CAPTURE(c2h::type_name<value_t>(), c2h::type_name<decltype(reduce_op)>(), valid_items, num_items, operator_identity);
  const int bounded_valid_items = ::cuda::std::min(valid_items, num_items);
  const value_t reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, reduce_op, operator_identity);

  run_thread_reduce_partial_kernel(num_items, d_in, d_out, reduce_op, valid_items);
  REQUIRE(reference_result == c2h::host_vector<value_t>(d_out)[0]);
}

#endif // TEST_HALF_T() || TEST_BF_T()

C2H_TEST("ThreadReduce Container Tests", "[reduce][thread]")
{
  using op_t       = ::cuda::std::plus<>;
  using dist_param = dist_interval<int, op_t>;
  c2h::device_vector<int> d_in(max_size);
  c2h::device_vector<int> d_out(1);
  c2h::gen(C2H_SEED(num_seeds), d_in, dist_param::min(), dist_param::max());
  c2h::host_vector<int> h_in = d_in;
  const int valid_items      = GENERATE_COPY(
    take(1, random(2, max_size - 2)),
    take(1, random(max_size + 2, ::cuda::std::numeric_limits<int>::max())),
    values({1, max_size - 1, max_size, max_size + 1}));
  const int bounded_valid_items = ::cuda::std::clamp(valid_items, 0, max_size);
  auto reference_result =
    compute_single_problem_reference(h_in.cbegin(), h_in.cbegin() + bounded_valid_items, op_t{}, 0);

  thread_reduce_partial_kernel_array<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), op_t{}, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  verify_results(reference_result, c2h::host_vector<int>(d_out)[0]);

  thread_reduce_partial_kernel_span<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), op_t{}, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  verify_results(reference_result, c2h::host_vector<int>(d_out)[0]);

#if _CCCL_STD_VER >= 2023
  thread_reduce_partial_kernel_mdspan<max_size>
    <<<1, 1>>>(thrust::raw_pointer_cast(d_in.data()), thrust::raw_pointer_cast(d_out.data()), op_t{}, valid_items);
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
  verify_results(reference_result, c2h::host_vector<int>(d_out)[0]);
#endif // _CCCL_STD_VER >= 2023
}

struct segment
{
  using offset_t = int32_t;
  // Make sure that default constructed segments can not be merged
  offset_t begin = ::cuda::std::numeric_limits<offset_t>::min();
  offset_t end   = ::cuda::std::numeric_limits<offset_t>::max();

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
  __host__ __device__ segment operator()(::cuda::std::tuple<segment::offset_t, segment::offset_t> interval)
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

C2H_TEST("ThreadReducePartial Invalid Test", "[reduce][thread]")
{
  const auto in_it = cuda::make_transform_iterator(
    thrust::make_zip_iterator(cuda::counting_iterator<segment::offset_t>{1},
                              cuda::counting_iterator<segment::offset_t>{2}),
    tuple_to_segment_op{});
  const int valid_items = GENERATE_COPY(
    take(3, random(2, max_size - 1)),
    take(1, random(max_size + 1, ::cuda::std::numeric_limits<int>::max())),
    values({-1, 0, 1}));
  const int bounded_valid_items = ::cuda::std::clamp(valid_items, 0, max_size);
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
    verify_results(reference_result, c2h::host_vector<segment>(d_out)[0]);
  }
}
