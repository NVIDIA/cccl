// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cub/util_macro.cuh>
#include <cub/warp/warp_reduce.cuh>

#include <thrust/iterator/constant_iterator.h>

#include <cuda/functional>
#include <cuda/ptx>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/bit>
#include <cuda/std/functional>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include <array>
#include <numeric>

#include <test_util.h>

#include <c2h/catch2_test_helper.h>
#include <c2h/check_results.cuh>
#include <c2h/custom_type.h>
#include <c2h/operator.cuh>

/***********************************************************************************************************************
 * Constants
 **********************************************************************************************************************/

inline constexpr int warp_size            = 32;
inline constexpr auto total_warps         = 4u;
inline constexpr int num_items_per_thread = 4;

/***********************************************************************************************************************
 * Kernel
 **********************************************************************************************************************/

template <unsigned LogicalWarpThreads, bool EnableNumItems = false, typename T, typename Output, typename ReductionOp>
__device__ void warp_reduce_function(T& thread_data, Output* output, ReductionOp reduction_op, int num_items = 0)
{
  using warp_reduce_t = cub::WarpReduce<Output, LogicalWarpThreads>;
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

template <unsigned LogicalWarpThreads, bool EnableNumItems, typename T, typename ReductionOp>
__global__ void warp_reduce_kernel(T* input, T* output, ReductionOp reduction_op, int num_items = 0)
{
  auto thread_data = input[threadIdx.x];
  warp_reduce_function<LogicalWarpThreads, EnableNumItems>(thread_data, output, reduction_op, num_items);
}

template <unsigned LogicalWarpThreads, typename T, typename ReductionOp>
__global__ void warp_reduce_multiple_items_kernel(T* input, T* output, ReductionOp reduction_op)
{
  T thread_data[num_items_per_thread];
  for (int i = 0; i < num_items_per_thread; ++i)
  {
    thread_data[i] = input[threadIdx.x * num_items_per_thread + i];
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

using custom_t =
  c2h::custom_type_t<c2h::accumulateable_t, c2h::equal_comparable_t, c2h::lexicographical_less_comparable_t>;

using full_type_list =
  c2h::type_list<uint8_t,
                 uint16_t,
                 int32_t,
                 int64_t,
                 custom_t,
#if _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4_16a,
#else // _CCCL_CTK_AT_LEAST(13, 0)
                 ulonglong4,
#endif // _CCCL_CTK_AT_LEAST(13, 0)
                 uchar3,
                 short2>;

using builtin_type_list = c2h::type_list<uint8_t, uint16_t, int32_t, int64_t>;

using predefined_op_list = c2h::type_list<cuda::std::plus<>, cuda::maximum<>, cuda::minimum<>>;

using logical_warp_threads = c2h::enum_type_list<unsigned, 32, 16, 9, 7, 1>;

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
  int items_per_thread       = 1)
{
  const auto identity    = identity_v<predefined_op, T>;
  items_per_logical_warp = items_per_logical_warp == 0 ? logical_warp_threads : items_per_logical_warp;
  for (unsigned i = 0; i < total_warps; ++i)
  {
    for (int j = 0; j < logical_warps; ++j)
    {
      auto start                   = h_in.begin() + (i * warp_size + j * logical_warp_threads) * items_per_thread;
      auto end                     = start + items_per_logical_warp * items_per_thread;
      h_out[i * logical_warps + j] = static_cast<T>(std::accumulate(start, end, identity, predefined_op{}));
    }
  }
}

_CCCL_DIAG_POP

std::array<unsigned, 3> get_test_config(unsigned logical_warp_threads, unsigned items_per_thread = 1)
{
  bool is_power_of_two = cuda::std::has_single_bit(logical_warp_threads);
  auto logical_warps   = is_power_of_two ? warp_size / logical_warp_threads : 1;
  auto input_size      = total_warps * warp_size * items_per_thread;
  auto output_size     = total_warps * logical_warps;
  return {input_size, output_size, logical_warps};
}

/***********************************************************************************************************************
 * Test cases
 **********************************************************************************************************************/

C2H_TEST("WarpReduce::Sum, full_type_list", "[reduce][warp][predefined_op][full]", full_type_list, logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<1, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  CAPTURE(c2h::type_name<T>(), c2h::type_name<T>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  warp_reduce_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<cuda::std::plus<>, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Sum/Max/Min, builtin types",
         "[reduce][warp][predefined_op][full]",
         builtin_type_list,
         predefined_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  warp_reduce_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<predefined_op, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::CustomSum", "[reduce][warp][generic][full]", full_type_list, logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<1, TestType>::value;
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

C2H_TEST("WarpReduce::Sum/Max/Min Partial",
         "[reduce][warp][predefined_op][partial]",
         builtin_type_list,
         predefined_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  const int valid_items                         = GENERATE_COPY(take(2, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads, valid_items);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  warp_reduce_launch<logical_warp_threads, true>(d_in, d_out, warp_reduce_t<predefined_op, T>{}, valid_items);

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

C2H_TEST("WarpReduce::Sum", "[reduce][warp][generic][partial]", full_type_list, logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<1, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads);
  const int valid_items                         = GENERATE_COPY(take(2, random(1u, logical_warp_threads)));
  CAPTURE(c2h::type_name<T>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  warp_reduce_launch<logical_warp_threads, true>(d_in, d_out, warp_reduce_t<custom_plus, T>{}, valid_items);

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<cuda::std::plus<>>(h_in, h_out, logical_warps, logical_warp_threads, valid_items);
  verify_results(h_out, d_out);
}

//----------------------------------------------------------------------------------------------------------------------
// multiple items per thread

C2H_TEST("WarpReduce::Sum/Max/Min Multiple Items Per Thread",
         "[reduce][warp][predefined_op][full]",
         builtin_type_list,
         predefined_op_list,
         logical_warp_threads)
{
  using T                                       = c2h::get<0, TestType>;
  using predefined_op                           = c2h::get<1, TestType>;
  constexpr auto logical_warp_threads           = c2h::get<2, TestType>::value;
  auto [input_size, output_size, logical_warps] = get_test_config(logical_warp_threads, num_items_per_thread);
  CAPTURE(c2h::type_name<T>(), c2h::type_name<predefined_op>(), logical_warp_threads);
  c2h::device_vector<T> d_in(input_size);
  c2h::device_vector<T> d_out(output_size);
  c2h::gen(C2H_SEED(10), d_in);
  warp_reduce_multiple_items_launch<logical_warp_threads>(d_in, d_out, warp_reduce_t<predefined_op, T>{});

  c2h::host_vector<T> h_in = d_in;
  c2h::host_vector<T> h_out(output_size);
  compute_host_reference<predefined_op>(h_in, h_out, logical_warps, logical_warp_threads, 0, num_items_per_thread);
  verify_results(h_out, d_out);
}
