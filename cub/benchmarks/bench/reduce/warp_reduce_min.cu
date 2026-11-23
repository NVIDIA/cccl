// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/functional>
#include <cuda/ptx>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

// complex types cannot be compared with operator<
using value_types = nvbench::type_list<
  int8_t,
  int16_t,
  int32_t,
  int64_t,
#if _CCCL_HAS_INT128()
  int128_t,
#endif
#if _CCCL_HAS_NVFP16() && _CCCL_CTK_AT_LEAST(12, 2)
  __half,
#endif
#if _CCCL_HAS_NVBF16() && _CCCL_CTK_AT_LEAST(12, 2)
  __nv_bfloat16,
#endif
  float,
  double
#if _CCCL_HAS_FLOAT128()
  ,
  __float128
#endif
  >;

template <int Rounds, typename T, typename Action>
_CCCL_DEVICE_API __forceinline__ T warp_reduce_min_rounds(T seed, Action action)
{
  const auto lane = threadIdx.x % 32;
  T data          = seed;
  T contribution  = seed;
  for (int i = 0; i < Rounds; ++i)
  {
    const T minimum = action(data);
    // Only lane zero owns the minimum. Other lanes reuse their seed for round two, then alternate its sign, so every
    // round reduces inputs that differ between lanes.
    data         = lane == 0 ? minimum : contribution;
    contribution = static_cast<T>(-contribution);
  }
  return data;
}

template <typename T>
_CCCL_DEVICE_API __forceinline__ T warp_reduce_min_seed()
{
  const auto lane = threadIdx.x % 32;
  return static_cast<T>(2 * static_cast<int>((cuda::ptx::get_sreg_clock() >> lane) & 1u) - 1);
}

template <int BlockSize, int Rounds, typename Action, typename T>
__launch_bounds__(BlockSize) _CCCL_KERNEL_ATTRIBUTES static void warp_reduce_min_kernel(const Action action)
{
  // Other lanes only alternate the sign of their seed, so no arithmetic besides the reduction remains in the loop.
  sink(warp_reduce_min_rounds<Rounds>(warp_reduce_min_seed<T>(), action));
}

template <int BlockSize, int Rounds, typename Action, typename T>
_CCCL_HOST_API constexpr auto& get_warp_reduce_kernel(::cuda::minimum<>)
{
  return warp_reduce_min_kernel<BlockSize, Rounds, Action, T>;
}

using op_t = ::cuda::minimum<>;
#include "warp_reduce_base.cuh"
