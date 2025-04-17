/***********************************************************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright notice, this list of conditions and the
 *       following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the
 *       following disclaimer in the documentation and/or other materials provided with the distribution.
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
#include <cuda/std/complex>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

// #include <cstdio>
#include <numeric>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/custom_type.h>
#include <c2h/operator.cuh>
#include <test_util.h>

/***********************************************************************************************************************
 * Constants
 **********************************************************************************************************************/

inline constexpr int warp_size        = 32;
inline constexpr auto total_warps     = 4u;
inline constexpr int items_per_thread = 4;

/***********************************************************************************************************************
 * Kernel
 **********************************************************************************************************************/

template <typename T>
inline constexpr bool is_device_supported_type_v = true;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800

template <>
inline constexpr bool is_device_supported_type_v<__nv_bfloat16> = false;

template <>
inline constexpr bool is_device_supported_type_v<__nv_bfloat162> = false;

template <>
inline constexpr bool is_device_supported_type_v<cuda::std::complex<__nv_bfloat16>> = false;

#endif

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 530

template <>
inline constexpr bool is_device_supported_type_v<__half> = false;

template <>
inline constexpr bool is_device_supported_type_v<__half2> = false;

template <>
inline constexpr bool is_device_supported_type_v<cuda::std::complex<__half>> = false;

#endif

template <unsigned LogicalWarpThreads, bool EnableNumItems = false, typename Input, typename T, typename ReductionOp>
__device__ void warp_reduce_function(Input& thread_data, T* output, ReductionOp reduction_op, int num_items = 0)
{
  if constexpr (is_device_supported_type_v<T>)
  {
    using warp_reduce_t = cub::WarpReduce<T, LogicalWarpThreads>;
    using storage_t     = typename warp_reduce_t::TempStorage;
    __shared__ storage_t storage[total_warps];
    constexpr bool is_power_of_two = cuda::std::has_single_bit(LogicalWarpThreads);
    auto lane                      = cuda::ptx::get_sreg_laneid();
    auto logical_warp              = is_power_of_two ? threadIdx.x / LogicalWarpThreads : threadIdx.x / warp_size;
    auto logical_lane              = is_power_of_two ? threadIdx.x % LogicalWarpThreads : lane;
    auto limit                     = EnableNumItems ? num_items : LogicalWarpThreads;
    if (!is_power_of_two && lane >= limit)
    {
      return;
    }
    warp_reduce_t warp_reduce{storage[logical_warp]};
    using result_t = decltype(reduction_op(warp_reduce, thread_data));
    result_t result;
    if constexpr (EnableNumItems)
    {
      result = reduction_op(warp_reduce, thread_data, num_items);
    }
    else
    {
      result = reduction_op(warp_reduce, thread_data);
    }
    if (logical_lane == 0)
    {
      output[logical_warp] = result;
    }
  }
}

template <unsigned LogicalWarpThreads, bool EnableNumItems, typename T, typename ReductionOp>
__global__ void warp_reduce_kernel(T* input, T* output, ReductionOp reduction_op, int num_items = 0)
{
  auto thread_data = input[threadIdx.x];
  warp_reduce_function<LogicalWarpThreads, EnableNumItems>(thread_data, output, reduction_op, num_items);
}

template <unsigned LogicalWarpThreads, typename T, typename ReductionOp>
__global__ void warp_reduce_multiple_items_kernel(T* input, T* output, ReductionOp reduction_op)
{
  T thread_data[items_per_thread];
  for (int i = 0; i < items_per_thread; ++i)
  {
    thread_data[i] = input[threadIdx.x * items_per_thread + i];
  }
  warp_reduce_function<LogicalWarpThreads>(thread_data, output, reduction_op);
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

template <int LogicalWarpThreads, bool EnableNumItems = false, typename T, typename... TArgs>
void warp_reduce_launch(c2h::device_vector<T>& input, c2h::device_vector<T>& output, TArgs... args)
{
  warp_reduce_kernel<LogicalWarpThreads, EnableNumItems><<<1, total_warps * warp_size>>>(
    thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), args...);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int LogicalWarpThreads, typename T, typename... TArgs>
void warp_reduce_multiple_items_launch(c2h::device_vector<T>& input, c2h::device_vector<T>& output, TArgs... args)
{
  warp_reduce_multiple_items_kernel<LogicalWarpThreads><<<1, total_warps * warp_size>>>(
    thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(output.data()), args...);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

/***********************************************************************************************************************
 * Types
 **********************************************************************************************************************/
// clang-format off

using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

using arithmetic_type_list = c2h::type_list<
  int8_t, uint16_t, int32_t, int64_t,
  float, double,
  cuda::std::complex<float>, cuda::std::complex<double>,
  short2, ushort2, float2,
  ulonglong4, custom_t
#  if _CCCL_HAS_INT128()
  , __int128_t
#  endif
#  if TEST_HALF_T()
   , __half
   , __half2
   , cuda::std::complex<__half>
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
   , __nv_bfloat16
   , __nv_bfloat162
   , cuda::std::complex<__nv_bfloat16>
#  endif // TEST_BF_T()
>;

using bitwise_type_list = c2h::type_list<uint8_t, uint16_t, uint32_t, uint64_t
#  if _CCCL_HAS_INT128()
    , __uint128_t
#  endif
>;

using bitwise_op_list = c2h::type_list<cuda::std::bit_and<>, cuda::std::bit_or<>, cuda::std::bit_xor<>>;

using min_max_type_list = c2h::type_list<
  int8_t, uint16_t, int32_t, int64_t,
  short2, ushort2,
#  if _CCCL_HAS_INT128()
  __int128_t,
#  endif
  float, double, custom_t
#  if TEST_HALF_T()
  , __half
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
  , __nv_bfloat16
#  endif // TEST_BF_T()
  >;

using min_max_op_list = c2h::type_list<cuda::minimum<>, cuda::maximum<>>;

using builtin_type_list = c2h::type_list<int8_t, uint16_t, int32_t, int64_t, float, double>;

using logical_warp_threads = c2h::enum_type_list<unsigned, 32, 16, 7, 1>;

// clang-format on
/***********************************************************************************************************************
 * Reference
 **********************************************************************************************************************/

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // numeric(33): C: '=': conversion from 'int' to '_Ty', possible loss of data

template <typename predefined_op, typename T>
void compute_host_reference(
  const c2h::host_vector<T>& h_in,
  c2h::host_vector<T>& h_out,
  int logical_warps,
  int logical_warp_threads,
  int items_per_logical_warp = 0,
  int items_per_thread1      = 1)
{
  auto identity          = operator_identity_v<T, predefined_op>;
  items_per_logical_warp = items_per_logical_warp == 0 ? logical_warp_threads : items_per_logical_warp;
  for (unsigned i = 0; i < total_warps; ++i)
  {
    for (int j = 0; j < logical_warps; ++j)
    {
      auto start                   = h_in.begin() + (i * warp_size + j * logical_warp_threads) * items_per_thread1;
      auto end                     = start + items_per_logical_warp * items_per_thread1;
      h_out[i * logical_warps + j] = std::accumulate(start, end, identity, predefined_op{});
    }
  }
}

_CCCL_DIAG_POP

std::array<unsigned, 3> get_test_config(unsigned logical_warp_threads, unsigned items_per_thread1 = 1)
{
  bool is_power_of_two = cuda::std::has_single_bit(logical_warp_threads);
  auto logical_warps   = is_power_of_two ? warp_size / logical_warp_threads : 1;
  auto input_size      = total_warps * warp_size * items_per_thread1;
  auto output_size     = total_warps * logical_warps;
  return {input_size, output_size, logical_warps};
}

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("WarpReduce::Sum", "[reduce][warp][predefined_op][full]", arithmetic_type_list, logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<1, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  CAPTURE(c2h::type_name<T>(), c2h::type_name<T>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  if constexpr (cuda::std::__is_any_floating_point_v<T>)
  {
    c2h::gen(C2H_SEED(1), d_in, T{-1.0}, T{2.0});
  }
  else if constexpr (cuda::std::is_same_v<T, float2>)
  {
    c2h::gen(C2H_SEED(1), d_in, T{-2.0, -1.0}, T{1.0, 2.0});
  }
  else
  {
    c2h::gen(C2H_SEED(1), d_in);
  }
  warp_reduce_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<cuda::std::plus<>, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Bitwise",
         "[reduce][warp][predefined_op][full]",
         bitwise_type_list,
         bitwise_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  warp_reduce_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<predefined_op, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Min/Max",
         "[reduce][warp][predefined_op][full]",
         min_max_type_list,
         min_max_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  warp_reduce_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<predefined_op, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::CustomSum", "[reduce][warp][generic][full]", logical_warp_threads)
{
  using T                                       = int;
  constexpr auto logical_warp_threads           = c2h::get<0, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  CAPTURE(c2h::type_name<T>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  warp_reduce_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<custom_plus, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

//----------------------------------------------------------------------------------------------------------------------
// partial

C2H_TEST("WarpReduce::Sum Partial", "[reduce][warp][predefined_op][partial]", arithmetic_type_list, logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<1, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  const int valid_items                         = GENERATE_COPY(take(2, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<T>(), logical_warp_threads, valid_items);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  if constexpr (cuda::std::__is_any_floating_point_v<T>)
  {
    c2h::gen(C2H_SEED(1), d_in, T{-1.0}, T{2.0});
  }
  else if constexpr (cuda::std::is_same_v<T, float2>)
  {
    c2h::gen(C2H_SEED(1), d_in, T{-2.0, -1.0}, T{1.0, 2.0});
  }
  else
  {
    c2h::gen(C2H_SEED(1), d_in);
  }
  warp_reduce_launch<logical_warp_threads, true>(d_in, d_out, warp_reduce_t<cuda::std::plus<>, T>{}, valid_items);

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Bitwise Partial",
         "[reduce][warp][predefined_op][partial]",
         bitwise_type_list,
         bitwise_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  const int valid_items                         = GENERATE_COPY(take(1, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads, valid_items);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  warp_reduce_launch<logical_warp_threads, true>(d_in, d_out, warp_reduce_t<predefined_op, T>{}, valid_items);

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Min/Max Partial",
         "[reduce][warp][predefined_op][partial]",
         min_max_type_list,
         min_max_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  const int valid_items                         = GENERATE_COPY(take(1, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads, valid_items);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  warp_reduce_launch<logical_warp_threads, true>(d_in, d_out, warp_reduce_t<predefined_op, T>{}, valid_items);

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::CustomSum Partial", "[reduce][warp][predefined_op][partial]", logical_warp_threads)
{
  using T                                       = int;
  constexpr auto logical_warp_threads           = c2h::get<0, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  const int valid_items                         = GENERATE_COPY(take(1, random(1u, logical_warp_threads)));
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(1), d_in);
  warp_reduce_launch<logical_warp_threads, true>(d_in, d_out, warp_reduce_t<custom_plus, T>{}, valid_items);

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

//----------------------------------------------------------------------------------------------------------------------
// multiple items per thread

C2H_TEST("WarpReduce::Sum Multiple Items Per Thread",
         "[reduce][warp][predefined_op][full]",
         builtin_type_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<1, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads, items_per_thread);
  CAPTURE(c2h::type_name<T>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    c2h::gen(C2H_SEED(1), d_in, T{-1.0}, T{2.0});
  }
  else
  {
    c2h::gen(C2H_SEED(1), d_in);
  }
  warp_reduce_multiple_items_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<cuda::std::plus<>, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads, 0, items_per_thread);
  verify_results(h_out, d_out);
}
