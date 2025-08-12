// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/util_arch.cuh>
#include <cub/warp/warp_scan.cuh>

#include <cuda/cmath>

#include <c2h/catch2_test_helper.h>

template <int LogicalWarpThreads, int TotalWarps, class T, class ActionT>
__global__ void
warp_combine_scan_kernel(T* in, T* inclusive_out, T* exclusive_out, ActionT action, int valid_items, T filler)
{
  using warp_scan_t = cub::WarpScan<T, LogicalWarpThreads>;
  using storage_t   = typename warp_scan_t::TempStorage;

  __shared__ storage_t storage[TotalWarps];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  // Get warp index
  int warp_id = tid / LogicalWarpThreads;

  T inc_out     = filler;
  T exc_out     = filler;
  T thread_data = in[tid];

  warp_scan_t scan(storage[warp_id]);

  action(scan, thread_data, inc_out, exc_out, valid_items);

  inclusive_out[tid] = inc_out;
  exclusive_out[tid] = exc_out;
}

template <int LogicalWarpThreads, int TotalWarps, class T, class ActionT>
void warp_combine_scan(
  c2h::device_vector<T>& in,
  c2h::device_vector<T>& inclusive_out,
  c2h::device_vector<T>& exclusive_out,
  ActionT action,
  int valid_items,
  T filler)
{
  warp_combine_scan_kernel<LogicalWarpThreads, TotalWarps, T, ActionT><<<1, LogicalWarpThreads * TotalWarps>>>(
    thrust::raw_pointer_cast(in.data()),
    thrust::raw_pointer_cast(inclusive_out.data()),
    thrust::raw_pointer_cast(exclusive_out.data()),
    action,
    valid_items,
    filler);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <int LogicalWarpThreads, int TotalWarps, class T, class ActionT>
__global__ void warp_scan_kernel(T* in, T* out, ActionT action, int valid_items)
{
  using warp_scan_t = cub::WarpScan<T, LogicalWarpThreads>;
  using storage_t   = typename warp_scan_t::TempStorage;

  __shared__ storage_t storage[TotalWarps];

  const int tid = cub::RowMajorTid(blockDim.x, blockDim.y, blockDim.z);

  // Get warp index
  int warp_id = tid / LogicalWarpThreads;

  T thread_data = in[tid];

  warp_scan_t scan(storage[warp_id]);

  action(scan, thread_data, valid_items);

  out[tid] = thread_data;
}

template <int LogicalWarpThreads, int TotalWarps, class T, class ActionT>
void warp_scan(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action, int valid_items)
{
  warp_scan_kernel<LogicalWarpThreads, TotalWarps, T, ActionT><<<1, LogicalWarpThreads * TotalWarps>>>(
    thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action, valid_items);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

enum class scan_mode
{
  exclusive,
  inclusive
};

template <class T, class ScanOpT>
c2h::host_vector<T> compute_host_reference(
  scan_mode mode,
  c2h::host_vector<T>& result,
  int logical_warp_threads,
  ScanOpT scan_op,
  int valid_items,
  T initial_value = T{})
{
  if (result.empty())
  {
    return c2h::host_vector<T>{};
  }
  assert(result.size() % logical_warp_threads == 0ul);

  // The accumulator variable is used to calculate warp_aggregate without
  // taking initial_value into consideration in both exclusive and inclusive scan.
  int num_warps = static_cast<int>(result.size()) / logical_warp_threads;
  c2h::host_vector<T> warp_accumulator(num_warps);
  if (mode == scan_mode::exclusive)
  {
    for (int w = 0; w < num_warps; ++w)
    {
      T* output     = result.data() + w * logical_warp_threads;
      T accumulator = T{};
      T current     = static_cast<T>(scan_op(initial_value, output[0]));
      if (valid_items > 0)
      {
        accumulator = output[0];
        output[0]   = initial_value;
      }
      for (int i = 1; i < cuda::std::clamp(valid_items, 0, logical_warp_threads); i++)
      {
        accumulator = static_cast<T>(scan_op(accumulator, output[i]));
        T tmp       = output[i];
        output[i]   = current;
        current     = static_cast<T>(scan_op(current, tmp));
      }
      warp_accumulator[w] = accumulator;
    }
  }
  else
  {
    for (int w = 0; w < num_warps; ++w)
    {
      T* output     = result.data() + w * logical_warp_threads;
      T accumulator = T{};
      T current     = static_cast<T>(scan_op(initial_value, output[0]));
      if (valid_items > 0)
      {
        accumulator = output[0];
        output[0]   = current;
      }
      for (int i = 1; i < cuda::std::clamp(valid_items, 0, logical_warp_threads); i++)
      {
        T tmp       = output[i];
        current     = static_cast<T>(scan_op(current, tmp));
        accumulator = static_cast<T>(scan_op(accumulator, tmp));
        output[i]   = current;
      }
      warp_accumulator[w] = accumulator;
    }
  }

  return warp_accumulator;
}

template <unsigned LogicalWarpThreads>
struct total_warps_t
{
private:
  static constexpr int max_warps      = 2;
  static constexpr bool is_arch_warp  = (LogicalWarpThreads == cub::detail::warp_threads);
  static constexpr bool is_pow_of_two = cuda::std::has_single_bit(LogicalWarpThreads);
  static constexpr int total_warps    = (is_arch_warp || is_pow_of_two) ? max_warps : 1;

public:
  static constexpr int value()
  {
    return total_warps;
  }
};

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int logical_warp_threads = c2h::get<1, TestType>::value;
  static constexpr scan_mode mode           = c2h::get<2, TestType>::value;
  static constexpr int total_warps          = total_warps_t<logical_warp_threads>::value();
  static constexpr int tile_size            = total_warps * logical_warp_threads;
};
