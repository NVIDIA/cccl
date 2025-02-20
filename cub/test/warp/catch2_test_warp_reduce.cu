/***********************************************************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use input source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions input binary form must reproduce the above copyright notice, this list of conditions and the
 *       following disclaimer input the documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the names of its contributors may be used to endorse or promote
 *       products derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/ptx>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <numeric>

#include <c2h/catch2_test_helper.h>
#include <c2h/custom_type.h>
#include <test_util.h>
#include <utils/check_results.cuh>
#include <utils/operator.cuh>

constexpr int warp_size = 32;

template <unsigned LogicalWarpThreads, int TotalWarps, typename T, typename ReductionOp>
__global__ void warp_reduce_kernel(T* input, T* output, ReductionOp reduction_op)
{
  using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;
  using storage_t     = typename warp_reduce_t::TempStorage;
  __shared__ storage_t storage[TotalWarps];
  constexpr bool is_power_of_two = cuda::std::has_single_bit(LogicalWarpThreads);
  auto lane                      = cuda::ptx::get_sreg_laneid();
  auto logical_warp              = is_power_of_two ? threadIdx.x / LogicalWarpThreads : threadIdx.x / warp_size;
  auto logical_lane              = is_power_of_two ? threadIdx.x % LogicalWarpThreads : lane;
  if (!is_power_of_two && lane >= LogicalWarpThreads)
  {
    return;
  }
  auto thread_data = input[threadIdx.x];
  warp_reduce_t warp_reduce{storage[logical_warp]};
  auto result = reduction_op(warp_reduce, thread_data);
  if (logical_lane == 0)
  {
    output[logical_warp] = result;
  }
}

template <unsigned LogicalWarpThreads, int TotalWarps, typename T, typename ReductionOp>
__global__ void warp_reduce_kernel(T* input, T* output, ReductionOp reduction_op, int num_items)
{
  using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;
  using storage_t     = typename warp_reduce_t::TempStorage;
  __shared__ storage_t storage[TotalWarps];
  constexpr bool is_power_of_two = cuda::std::has_single_bit(LogicalWarpThreads);
  auto lane                      = cuda::ptx::get_sreg_laneid();
  auto logical_warp              = is_power_of_two ? threadIdx.x / LogicalWarpThreads : threadIdx.x / warp_size;
  auto logical_lane              = is_power_of_two ? threadIdx.x % LogicalWarpThreads : lane;
  if (!is_power_of_two && lane >= num_items)
  {
    return;
  }
  auto thread_data = input[threadIdx.x];
  warp_reduce_t warp_reduce{storage[logical_warp]};
  auto result = reduction_op(warp_reduce, thread_data, num_items);
  if (logical_lane == 0)
  {
    output[logical_warp] = result;
  }
}
inline constexpr int items_per_thread = 4;

template <unsigned LogicalWarpThreads, int TotalWarps, typename T, typename ReductionOp>
__global__ void warp_reduce_multiple_items_kernel(T* input, T* output, ReductionOp reduction_op)
{
  using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;
  using storage_t     = typename warp_reduce_t::TempStorage;
  __shared__ storage_t storage[TotalWarps];
  constexpr bool is_power_of_two = cuda::std::has_single_bit(LogicalWarpThreads);
  auto lane                      = cuda::ptx::get_sreg_laneid();
  auto logical_warp              = is_power_of_two ? threadIdx.x / LogicalWarpThreads : threadIdx.x / warp_size;
  auto logical_lane              = is_power_of_two ? threadIdx.x % LogicalWarpThreads : lane;
  if (!is_power_of_two && lane >= LogicalWarpThreads)
  {
    return;
  }
  T thread_data[items_per_thread];
  for (int i = 0; i < items_per_thread; ++i)
  {
    thread_data[i] = input[threadIdx.x * items_per_thread + i];
  }
  warp_reduce_t warp_reduce{storage[logical_warp]};
  static_assert(cub::detail::is_fixed_size_random_access_range_t<decltype(thread_data)>{});
  auto result = reduction_op(warp_reduce, thread_data);
  if (logical_lane == 0)
  {
    output[logical_warp] = result;
  }
}

template <typename Op, typename T>
struct warp_reduce_t
{
  template <int LogicalWarpThreads>
  __device__ auto operator()(cub::WarpReduce<T, LogicalWarpThreads> warp_reduce, T& data) const
  {
    return warp_reduce.Reduce(data, Op{});
  }

  template <int LogicalWarpThreads>
  __device__ auto operator()(cub::WarpReduce<T, LogicalWarpThreads> warp_reduce, T& data, int num_items) const
  {
    return warp_reduce.Reduce(data, Op{}, num_items);
  }
};

template <typename T>
struct warp_reduce_t<cuda::std::plus<>, T>
{
  template <int LogicalWarpThreads, typename... TArgs>
  __device__ auto operator()(cub::WarpReduce<T, LogicalWarpThreads> warp_reduce, TArgs&&... args) const
  {
    return warp_reduce.Sum(args...);
  }
};

template <typename T>
struct warp_reduce_t<cuda::maximum<>, T>
{
  template <int LogicalWarpThreads, typename... TArgs>
  __device__ auto operator()(cub::WarpReduce<T, LogicalWarpThreads> warp_reduce, TArgs&&... args) const
  {
    return warp_reduce.Max(args...);
  }
};

template <typename T>
struct warp_reduce_t<cuda::minimum<>, T>
{
  template <int LogicalWarpThreads, typename... TArgs>
  __device__ auto operator()(cub::WarpReduce<T, LogicalWarpThreads> warp_reduce, TArgs&&... args) const
  {
    return warp_reduce.Min(args...);
  }
};

/**
 * @brief Delegate wrapper for WarpReduce::Sum
 */
// template <typename T>
// struct warp_sum_t
//{
//   template <int LogicalWarpThreads>
//   __device__ auto operator()(cub::WarpReduce<T, LogicalWarpThreads> warp_reduce, const T& thread_data) const
//   {
//     return warp_reduce.Sum(thread_data);
//   }
// };
//
///**
// * @brief Delegate wrapper for partial WarpReduce::Sum
// */
// template <typename T>
// struct warp_sum_partial_t
//{
//  int num_valid;
//  template <int LogicalWarpThreads>
//  __device__ __forceinline__ T
//  operator()(int linear_tid, cub::WarpReduce<T, LogicalWarpThreads>& warp_reduce, T& thread_data) const
//  {
//    auto result = warp_reduce.Sum(thread_data, num_valid);
//    return ((linear_tid % LogicalWarpThreads) == 0) ? result : thread_data;
//  }
//};

/**
 * @brief Delegate wrapper for WarpReduce::Reduce
 */
// template <typename T, typename ReductionOpT>
// struct warp_reduce_t
//{
//   ReductionOpT reduction_op;
//   template <int LogicalWarpThreads>
//   __device__ __forceinline__ T
//   operator()(int linear_tid, cub::WarpReduce<T, LogicalWarpThreads>& warp_reduce, T& thread_data) const
//   {
//     auto result = warp_reduce.Reduce(thread_data, reduction_op);
//     return ((linear_tid % LogicalWarpThreads) == 0) ? result : thread_data;
//   }
// };
//
///**
// * @brief Delegate wrapper for partial WarpReduce::Reduce
// */
// template <typename T, typename ReductionOpT>
// struct warp_reduce_partial_t
//{
//  int num_valid;
//  ReductionOpT reduction_op;
//  template <int LogicalWarpThreads>
//  __device__ T operator()(int linear_tid, cub::WarpReduce<T, LogicalWarpThreads>& warp_reduce, T& thread_data) const
//  {
//    auto result = warp_reduce.Reduce(thread_data, reduction_op, num_valid);
//    return ((linear_tid % LogicalWarpThreads) == 0) ? result : thread_data;
//  }
//};

template <int LogicalWarpThreads, int TotalWarps, typename T, typename... TArgs>
void warp_reduce_launch(c2h::device_vector<T>& input, c2h::device_vector<T>& output, TArgs... args)
{
  warp_reduce_kernel<LogicalWarpThreads, TotalWarps><<<1, TotalWarps * warp_size>>>(
    thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), args...);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int LogicalWarpThreads, int TotalWarps, typename T, typename... TArgs>
void warp_reduce_multiple_items_launch(c2h::device_vector<T>& input, c2h::device_vector<T>& output, TArgs... args)
{
  warp_reduce_multiple_items_kernel<LogicalWarpThreads, TotalWarps><<<1, TotalWarps * warp_size>>>(
    thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), args...);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/***********************************************************************************************************************
 * Types
 **********************************************************************************************************************/
#if 0
// List of types to test
using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

using full_type_list = c2h::type_list<uint8_t, uint16_t, int32_t, int64_t, custom_t, ulonglong4, uchar3, short2>;

using builtin_type_list = c2h::type_list<uint8_t, uint16_t, int32_t, int64_t>;

using predefined_op_list = c2h::type_list<::cuda::std::plus<>, ::cuda::maximum<>, ::cuda::minimum<>>;

using logical_warp_threads = c2h::enum_type_list<unsigned, 32, 16, 9, 7, 1>;

#else
using builtin_type_list = c2h::type_list<int32_t>;

using predefined_op_list = c2h::type_list<::cuda::std::plus<>>;

using logical_warp_threads = c2h::enum_type_list<unsigned, 32>;
#endif

/***********************************************************************************************************************
 * Reference
 **********************************************************************************************************************/

template <typename predefined_op, typename T>
void compute_host_reference(
  const c2h::host_vector<T>& h_in,
  c2h::host_vector<T>& h_out,
  int total_warps,
  int items_per_warp,
  int logical_warps,
  int logical_warp_stride,
  int items_per_logical_warp = 0)
{
  constexpr auto identity = operator_identity_v<T, predefined_op>;
  items_per_logical_warp  = items_per_logical_warp == 0 ? logical_warp_stride : items_per_logical_warp;
  for (int i = 0; i < total_warps; ++i)
  {
    for (int j = 0; j < logical_warps; ++j)
    {
      auto start                   = h_in.begin() + i * items_per_warp + j * logical_warp_stride;
      auto end                     = start + items_per_logical_warp;
      h_out[i * logical_warps + j] = std::accumulate(start, end, identity, predefined_op{});
    }
  }
}

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/
/*
C2H_TEST("WarpReduce::Sum", "[reduce][warp][predefined][full]", full_type_list, logical_warp_threads)
{
  using type                          = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads = c2h::get<1, TestType>::value;
  constexpr auto total_warps          = 4u;
  constexpr bool is_power_of_two      = cuda::std::has_single_bit(logical_warp_threads);
  constexpr auto logical_warps        = is_power_of_two ? warp_size / logical_warp_threads : 1;
  constexpr auto input_size           = total_warps * warp_size;
  constexpr auto output_size          = total_warps * logical_warps;
  CAPTURE(c2h::type_name<type>(), logical_warp_threads);
  c2h::device_vector<type> d_in(input_size);
  c2h::device_vector<type> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  // Run test
  warp_reduce_launch<logical_warp_threads, total_warps>(d_in, d_out, warp_reduce_t<cuda::std::plus<>, type>{});

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, total_warps, warp_size, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Sum/Max/Min",
         "[reduce][warp][predefined][full]",
         builtin_type_list,
         predefined_op_list,
         logical_warp_threads)
{
  using type                          = c2h::get<0, TestType>;
  using predefined_op                 = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads = c2h::get<2, TestType>::value;
  constexpr auto total_warps          = 4u;
  constexpr bool is_power_of_two      = cuda::std::has_single_bit(logical_warp_threads);
  constexpr auto logical_warps        = is_power_of_two ? warp_size / logical_warp_threads : 1;
  constexpr auto input_size           = total_warps * warp_size;
  constexpr auto output_size          = total_warps * logical_warps;
  CAPTURE(c2h::type_name<type>(), c2h::type_name<predefined_op>(), logical_warp_threads);
  c2h::device_vector<type> d_in(input_size);
  c2h::device_vector<type> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  // Run test
  warp_reduce_launch<logical_warp_threads, total_warps>(d_in, d_out, warp_reduce_t<predefined_op, type>{});

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, total_warps, warp_size, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Sum", "[reduce][warp][generic][full]", full_type_list, logical_warp_threads)
{
  using type                          = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads = c2h::get<1, TestType>::value;
  constexpr auto total_warps          = 4u;
  constexpr bool is_power_of_two      = cuda::std::has_single_bit(logical_warp_threads);
  constexpr auto logical_warps        = is_power_of_two ? warp_size / logical_warp_threads : 1;
  constexpr auto input_size           = total_warps * warp_size;
  constexpr auto output_size          = total_warps * logical_warps;
  CAPTURE(c2h::type_name<type>(), logical_warp_threads);
  c2h::device_vector<type> d_in(input_size);
  c2h::device_vector<type> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  // Run test
  warp_reduce_launch<logical_warp_threads, total_warps>(d_in, d_out, warp_reduce_t<custom_plus, type>{});

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, total_warps, warp_size, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

//----------------------------------------------------------------------------------------------------------------------
// partial

C2H_TEST("WarpReduce::Sum/Max/Min Partial",
         "[reduce][warp][predefined][partial]",
         builtin_type_list,
         predefined_op_list,
         logical_warp_threads)
{
  using type                          = c2h::get<0, TestType>;
  using predefined_op                 = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads = c2h::get<2, TestType>::value;
  constexpr auto total_warps          = 4u;
  constexpr bool is_power_of_two      = cuda::std::has_single_bit(logical_warp_threads);
  constexpr auto logical_warps        = is_power_of_two ? warp_size / logical_warp_threads : 1;
  constexpr auto input_size           = total_warps * warp_size;
  constexpr auto output_size          = total_warps * logical_warps;
  const int valid_items               = GENERATE_COPY(take(2, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<type>(), c2h::type_name<predefined_op>(), logical_warp_threads, valid_items);
  c2h::device_vector<type> d_in(input_size);
  c2h::device_vector<type> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  // Run test
  warp_reduce_launch<logical_warp_threads, total_warps>(d_in, d_out, warp_reduce_t<predefined_op, type>{}, valid_items);

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(output_size);
  compute_host_reference<predefined_op>(
    h_in, h_out, total_warps, warp_size, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Sum", "[reduce][warp][generic][partial]", full_type_list, logical_warp_threads)
{
  using type                          = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads = c2h::get<1, TestType>::value;
  constexpr auto total_warps          = 4u;
  constexpr bool is_power_of_two      = cuda::std::has_single_bit(logical_warp_threads);
  constexpr auto logical_warps        = is_power_of_two ? warp_size / logical_warp_threads : 1;
  constexpr auto input_size           = total_warps * warp_size;
  constexpr auto output_size          = total_warps * logical_warps;
  const int valid_items               = GENERATE_COPY(take(2, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<type>(), logical_warp_threads);
  c2h::device_vector<type> d_in(input_size);
  c2h::device_vector<type> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  // Run test
  warp_reduce_launch<logical_warp_threads, total_warps>(d_in, d_out, warp_reduce_t<custom_plus, type>{}, valid_items);

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(
    h_in, h_out, total_warps, warp_size, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}
*/
//----------------------------------------------------------------------------------------------------------------------
// multiple items per thread

C2H_TEST("WarpReduce::Sum/Max/Min Multiple Items Per Thread",
         "[reduce][warp][predefined][full]",
         builtin_type_list,
         predefined_op_list,
         logical_warp_threads)
{
  using type                          = c2h::get<0, TestType>;
  using predefined_op                 = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads = c2h::get<2, TestType>::value;
  constexpr auto total_warps          = 4u;
  constexpr bool is_power_of_two      = cuda::std::has_single_bit(logical_warp_threads);
  constexpr auto logical_warps        = is_power_of_two ? warp_size / logical_warp_threads : 1;
  constexpr auto input_size           = total_warps * warp_size * items_per_thread;
  constexpr auto output_size          = total_warps * logical_warps;
  CAPTURE(c2h::type_name<type>(), c2h::type_name<predefined_op>(), logical_warp_threads);
  c2h::device_vector<type> d_in(input_size, 1);
  c2h::device_vector<type> d_out(output_size);
  // c2h::gen(C2H_SEED(1), d_in);
  //  Run test
  warp_reduce_multiple_items_launch<logical_warp_threads, total_warps>(
    d_in, d_out, warp_reduce_t<predefined_op, type>{});

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(output_size);
  compute_host_reference<predefined_op>(
    h_in,
    h_out,
    total_warps,
    warp_size * items_per_thread,
    logical_warps,
    logical_warp_threads,
    logical_warp_threads * items_per_thread);
  verify_results(h_out, d_out);
}

#if 0
C2H_TEST("Warp sum works", "[reduce][warp][predefined]", builtin_type_list, logical_warp_threads, predefined_op_list)
{
  using params                        = params_t<TestType>;
  constexpr auto logical_warp_threads = params::logical_warp_threads;
  constexpr auto total_warps          = params::total_warps;
  using type                          = typename params::type;
  // Prepare test data
  c2h::device_vector<type> d_in(params::input_size);
  c2h::device_vector<type> d_out(total_warps);
  constexpr auto valid_items = logical_warp_threads;
  c2h::gen(C2H_SEED(10), d_in);
  // Run test
  warp_reduce<logical_warp_threads, total_warps>(d_in, d_out, warp_sum_t<type>{});

  c2h::host_vector<type> h_in = d_in;
  c2h::host_vector<type> h_out(total_warps);
  for (int i = 0; i < total_warps; ++i)
  {
    auto start = h_in.begin() + i * logical_warp_threads;
    auto end   = h_in.begin() + (i + 1) * logical_warp_threads;
    h_out[i]   = std::accumulate(start, end, type{});
  }
  verify_results(h_out, d_out);
}

C2H_TEST("Warp reduce works", "[reduce][warp]", builtin_type_list, logical_warp_threads)
{
  using params   = params_t<TestType>;
  using type     = typename params::type;
  using red_op_t = ::cuda::minimum<>;

  // Prepare test data
  c2h::device_vector<type> d_in(params::input_size);
  c2h::device_vector<type> d_out(params::input_size);
  constexpr auto valid_items = params::logical_warp_threads;
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(d_in, d_out, warp_reduce_t<type, red_op_t>{red_op_t{}});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    red_op_t{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp sum on partial warp works", "[reduce][warp]", full_type_list, logical_warp_threads)
{
  using params = params_t<TestType>;
  using type   = typename params::type;

  // Prepare test data
  c2h::device_vector<type> d_in(params::input_size);
  c2h::device_vector<type> d_out(params::input_size);
  const int valid_items = GENERATE_COPY(take(2, random(1, params::logical_warp_threads)));
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(d_in, d_out, warp_sum_partial_t<type>{valid_items});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    ::cuda::std::plus<type>{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}

C2H_TEST("Warp reduce on partial warp works", "[reduce][warp]", builtin_type_list, logical_warp_threads)
{
  using params   = params_t<TestType>;
  using type     = typename params::type;
  using red_op_t = ::cuda::minimum<>;

  // Prepare test data
  c2h::device_vector<type> d_in(params::input_size);
  c2h::device_vector<type> d_out(params::input_size);
  const int valid_items = GENERATE_COPY(take(2, random(1, params::logical_warp_threads)));
  c2h::gen(C2H_SEED(10), d_in);

  // Run test
  warp_reduce<params::logical_warp_threads, params::total_warps>(
    d_in, d_out, warp_reduce_partial_t<type, red_op_t>{valid_items, red_op_t{}});

  // Prepare verification data
  c2h::host_vector<type> h_in  = d_in;
  c2h::host_vector<type> h_out = h_in;
  auto h_flags                 = thrust::make_constant_iterator(false);
  compute_host_reference(
    reduce_mode::all,
    h_in,
    h_flags,
    params::total_warps,
    params::logical_warp_threads,
    valid_items,
    red_op_t{},
    h_out.begin());

  // Verify results
  verify_results(h_out, d_out);
}
#endif
