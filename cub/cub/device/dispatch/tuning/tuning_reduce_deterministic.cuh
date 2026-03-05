// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_arch.cuh>

#include <cuda/__device/arch_id.h>

CUB_NAMESPACE_BEGIN

namespace detail::rfa
{
struct reduce_policy
{
  int block_threads;
  int items_per_thread;
  BlockReduceAlgorithm block_algorithm;
};

struct single_tile_policy
{
  int block_threads;
  int items_per_thread;
  BlockReduceAlgorithm block_algorithm;
};

struct rfa_policy
{
  reduce_policy reduce;
  single_tile_policy single_tile;
};

struct policy_selector
{
  type_t accum_t;
  int accum_size;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> rfa_policy
  {
    if (arch >= ::cuda::arch_id::sm_90)
    {
      // only tuned for float, fall through for other types
      if (accum_t == type_t::float32)
      {
        // ipt_13.tpb_224  1.107188  1.009709  1.097114  1.316820
        const auto scaled = scale_mem_bound(224, 13, accum_size);
        return {{scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
                {scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
      }
    }

    if (arch >= ::cuda::arch_id::sm_86)
    {
      // only tuned for float and double, fall through for other types
      if (accum_t == type_t::float32)
      {
        // ipt_6.tpb_224  1.034383  1.000000  1.032097  1.090909
        const auto scaled = scale_mem_bound(224, 6, accum_size);
        return {{scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
                {scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
      }
      if (accum_t == type_t::float64)
      {
        // ipt_11.tpb_128 ()  1.232089  1.002124  1.245336  1.582279
        const auto scaled = scale_mem_bound(128, 11, accum_size);
        return {{scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
                {scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
      }
    }

    if (arch >= ::cuda::arch_id::sm_60)
    {
      const auto scaled = scale_mem_bound(256, 16, accum_size);
      return {{scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
              {scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
    }

    const auto scaled = scale_mem_bound(256, 20, accum_size);
    return {{scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING},
            {scaled.block_threads, scaled.items_per_thread, BLOCK_REDUCE_RAKING}};
  }
};

// stateless version which can be passed to kernels
template <typename AccumT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> rfa_policy
  {
    return policy_selector{classify_type<AccumT>, int{sizeof(AccumT)}}(arch);
  }
};
} // namespace detail::rfa

CUB_NAMESPACE_END
