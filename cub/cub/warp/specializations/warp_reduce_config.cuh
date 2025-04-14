/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
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

#include <cuda/__bit/bitmask.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/bit>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

/***********************************************************************************************************************
 * WarpReduce Configuration
 **********************************************************************************************************************/

CUB_NAMESPACE_BEGIN
namespace detail
{

enum class ReduceLogicalMode
{
  SingleReduction,
  MultipleReductions
};

enum class ReduceResultMode
{
  AllLanes,
  SingleLane
};

template <ReduceLogicalMode LogicalMode>
using reduce_logical_mode_t = _CUDA_VSTD::integral_constant<ReduceLogicalMode, LogicalMode>;

template <ReduceResultMode Kind>
using reduce_result_mode_t = _CUDA_VSTD::integral_constant<ReduceResultMode, Kind>;

template <size_t LastPos = _CUDA_VSTD::dynamic_extent>
using last_pos_t = _CUDA_VSTD::extents<int, LastPos>;

template <size_t LogicalSize>
using logical_warp_size_t = _CUDA_VSTD::integral_constant<int, LogicalSize>;

template <bool IsSegmented = true>
using is_segmented_t = _CUDA_VSTD::bool_constant<IsSegmented>;

//----------------------------------------------------------------------------------------------------------------------
// WarpReduceConfig

template <ReduceLogicalMode LogicalMode,
          ReduceResultMode ResultMode,
          int LogicalWarpSize,
          size_t LastPos   = LogicalWarpSize - 1,
          bool IsSegmented = false>
struct WarpReduceConfig
{
  reduce_logical_mode_t<LogicalMode> logical_mode;
  reduce_result_mode_t<ResultMode> result_mode;
  logical_warp_size_t<LogicalWarpSize> logical_size;
  last_pos_t<LastPos> last_pos;
  is_segmented_t<IsSegmented> is_segmented;
};

template <ReduceLogicalMode LogicalMode,
          ReduceResultMode ResultMode,
          int LogicalWarpSize,
          size_t LastPos   = LogicalWarpSize - 1,
          bool IsSegmented = false>
WarpReduceConfig(reduce_logical_mode_t<LogicalMode>,
                 reduce_result_mode_t<ResultMode>,
                 logical_warp_size_t<LogicalWarpSize>,
                 last_pos_t<LastPos>         = {},
                 is_segmented_t<IsSegmented> = {})
  -> WarpReduceConfig<LogicalMode, ResultMode, LogicalWarpSize, LastPos, IsSegmented>;

} // namespace detail

/***********************************************************************************************************************
 * WarpReduce Configuration Interface
 **********************************************************************************************************************/

inline constexpr auto single_reduction = detail::reduce_logical_mode_t<detail::ReduceLogicalMode::SingleReduction>{};
inline constexpr auto multiple_reductions =
  detail::reduce_logical_mode_t<detail::ReduceLogicalMode::MultipleReductions>{};

inline constexpr auto all_lanes_result  = detail::reduce_result_mode_t<detail::ReduceResultMode::AllLanes>{};
inline constexpr auto first_lane_result = detail::reduce_result_mode_t<detail::ReduceResultMode::SingleLane>{};

/***********************************************************************************************************************
 * WarpReduce Check Configuration
 **********************************************************************************************************************/

namespace detail
{

template <typename WarpReduceConfig>
_CCCL_DEVICE _CCCL_FORCEINLINE void check_warp_reduce_config(WarpReduceConfig config)
{
  auto [logical_mode, result_mode, logical_size, last_pos, is_segmented] = config;
  // Check logical_size
  static_assert(logical_size > 0 && logical_size <= warp_threads, "invalid logical warp size");
  // Check logical mode
  if constexpr (logical_mode == multiple_reductions)
  {
    static constexpr bool is_power_of_two = _CUDA_VSTD::has_single_bit(uint32_t{logical_size()});
    static_assert(is_power_of_two, "Logical size must be a power of two with multiple reductions");
  }
  // Check segmented reduction with last position / result mode
  if constexpr (is_segmented)
  {
    static_assert(last_pos.rank_dynamic() == 1, "last_pos must be dynamic with segmented reductions");
    static_assert(result_mode == first_lane_result, "result_mode must be first_lane_result with segmented reductions");
  }
  // Check last position
  _CCCL_ASSERT(last_pos.extent(0) <= logical_size, "invalid last valid position");
  // Check lane mask
  auto logical_mask               = ::cuda::bitmask<uint32_t>(0, last_pos.extent(0));
  [[maybe_unused]] auto lane_mask = (logical_mode == single_reduction) ? logical_mask : 0xFFFFFFFF;
  _CCCL_ASSERT((::__activemask() & lane_mask) == lane_mask, "Invalid lane mask");
}

} // namespace detail
CUB_NAMESPACE_END
