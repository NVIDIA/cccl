/***********************************************************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_operators.cuh> // is_cuda_minimum_maximum_v
#include <cub/warp/specializations/warp_reduce_config.cuh>
#include <cub/warp/warp_utils.cuh> // logical_warp_base_id

#include <cuda/bit> // cuda::bitmask
#include <cuda/functional>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN
namespace detail
{

//----------------------------------------------------------------------------------------------------------------------
// SM100 Min/Max Reduction

#if __cccl_ptx_isa >= 860

#  define _CUB_REDUX_FLOAT_OP(OPERATOR, PTX_OP)                                                               \
                                                                                                              \
    template <typename = void>                                                                                \
    [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE float redux_sm100a_ptx(OPERATOR, float value, uint32_t mask) \
    {                                                                                                         \
      float result;                                                                                           \
      asm volatile("{"                                                                                        \
                   "redux.sync." #PTX_OP ".f32 %0, %1, %2;"                                                   \
                   "}"                                                                                        \
                   : "=f"(result)                                                                             \
                   : "f"(value), "r"(mask));                                                                  \
      return result;                                                                                          \
    }

_CUB_REDUX_FLOAT_OP(::cuda::minimum<>, min)
_CUB_REDUX_FLOAT_OP(::cuda::maximum<>, max)

#endif // __cccl_ptx_isa >= 860

//----------------------------------------------------------------------------------------------------------------------

template <typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE uint32_t redux_lane_mask(Config config)
{
  auto [logical_mode, _, logical_size, valid_items, is_segmented, first_pos] = config;
  constexpr bool is_single_reduction                                         = logical_mode == single_reduction;
  [[maybe_unused]] auto shift = is_single_reduction ? 0 : cub::detail::logical_warp_base_id(logical_size);
  if constexpr (is_segmented)
  {
    return ::cuda::bitmask(first_pos, valid_items.extent(0) - first_pos + 1);
  }
  else if constexpr (valid_items.rank_dynamic() == 1)
  {
    auto base_mask = ::cuda::bitmask(0, valid_items.extent(0));
    return base_mask << shift;
  }
  else
  {
    constexpr auto base_mask = ::cuda::bitmask(0, logical_size); // must be constexpr
    return base_mask << shift;
  }
}

//----------------------------------------------------------------------------------------------------------------------
// PTX Redux SM100

extern "C" _CCCL_DEVICE float redux_min_max_sync_is_not_supported_before_sm100a();

template <typename T, typename ReductionOp, typename Config>
[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE T redux_sm100a(ReductionOp, T value, Config config)
{
  using namespace _CUDA_VSTD;
#if __cccl_ptx_isa >= 860
  static_assert(is_same_v<T, float> && is_cuda_minimum_maximum_v<ReductionOp, T>);
  const auto mask = cub::detail::redux_lane_mask(config);
  NV_IF_TARGET(NV_PROVIDES_SM_100,
               (return cub::detail::redux_sm100a_ptx(ReductionOp1{}, value, mask);),
               (return cub::detail::redux_min_max_sync_is_not_supported_before_sm100a();))
#else
  static_assert(__always_false_v<T>, "redux.sync.min/max.f32  requires PTX ISA >= 860");
#endif // __cccl_ptx_isa >= 860
}

} // namespace detail
CUB_NAMESPACE_END
