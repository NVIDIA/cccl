// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/ptx>
#include <cuda/std/complex>
#include <cuda/std/type_traits>

#include <device_side_benchmark.cuh>
#include <nvbench_helper.cuh>

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
  double,
#if _CCCL_HAS_FLOAT128()
  __float128,
#endif
  cuda::std::complex<float>,
  cuda::std::complex<double>>;

template <int Rounds, typename T, typename Action>
_CCCL_DEVICE_API __forceinline__ T warp_reduce_sum_rounds(T seed, Action action)
{
  const auto lane = threadIdx.x % 32;
  T data          = seed;
  T contribution  = seed;
  for (int i = 0; i < Rounds; ++i)
  {
    const T sum = action(data);
    // Only lane zero owns the sum. Other lanes reuse their seed for round two, then alternate its sign.
    data         = lane == 0 ? sum : contribution;
    contribution = static_cast<T>(-contribution);
  }
  return data;
}

template <typename T>
_CCCL_DEVICE_API __forceinline__ T warp_reduce_sum_seed()
{
  const auto lane   = threadIdx.x % 32;
  const auto sample = [lane]() {
    return 2 * static_cast<int>((cuda::ptx::get_sreg_clock() >> lane) & 1u) - 1;
  };
  if constexpr (::cuda::std::is_same_v<T, complex32> || ::cuda::std::is_same_v<T, complex64>)
  {
    using component = typename T::value_type;
    return T{static_cast<component>(sample()), static_cast<component>(sample())};
  }
  else
  {
    return static_cast<T>(sample());
  }
}

template <int BlockSize, int Rounds, typename Action, typename T>
__launch_bounds__(BlockSize) _CCCL_KERNEL_ATTRIBUTES static void warp_reduce_sum_kernel(const Action action)
{
  // Unit-magnitude components keep every partial sum within [-63, 63], even for int8_t.
  sink(warp_reduce_sum_rounds<Rounds>(warp_reduce_sum_seed<T>(), action));
}

template <int BlockSize, int Rounds, typename Action, typename T>
_CCCL_HOST_API constexpr auto& get_warp_reduce_kernel(::cuda::std::plus<>)
{
  return warp_reduce_sum_kernel<BlockSize, Rounds, Action, T>;
}

using op_t = ::cuda::std::plus<>;
#include "warp_reduce_base.cuh"
