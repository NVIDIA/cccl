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

#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

/***********************************************************************************************************************
 * WarpReduce Configuration
 **********************************************************************************************************************/

namespace detail
{

enum class ReduceLogicalMode
{
  SingleReduction,
  MultipleReductions
};

enum class WarpReduceResultMode
{
  AllLanes,
  SingleLane
};

template <ReduceLogicalMode LogicalMode>
using reduce_logical_mode_t = _CUDA_VSTD::integral_constant<ReduceLogicalMode, LogicalMode>;

template <WarpReduceResultMode Kind>
using reduce_result_mode_t = _CUDA_VSTD::integral_constant<WarpReduceResultMode, Kind>;

template <size_t ValidItems = _CUDA_VSTD::dynamic_extent>
using valid_items_t = _CUDA_VSTD::extents<int, ValidItems>;

template <size_t LogicalSize>
using logical_warp_size_t = _CUDA_VSTD::integral_constant<int, LogicalSize>;

} // namespace detail

inline constexpr auto single_reduction = detail::reduce_logical_mode_t<detail::ReduceLogicalMode::SingleReduction>{};
inline constexpr auto multiple_reductions =
  detail::reduce_logical_mode_t<detail::ReduceLogicalMode::MultipleReductions>{};

inline constexpr auto all_lanes_result  = detail::reduce_result_mode_t<detail::WarpReduceResultMode::AllLanes>{};
inline constexpr auto first_lane_result = detail::reduce_result_mode_t<detail::WarpReduceResultMode::SingleLane>{};

CUB_NAMESPACE_END
