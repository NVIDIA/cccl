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

#include "cuda/std/__bit/has_single_bit.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN
namespace detail
{

[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE int logical_lane_id(uint32_t LogicalWarpSize)
{
  _CCCL_ASSERT(LogicalWarpSize <= warp_threads, "invalid logical warp size");
  auto lane                   = _CUDA_VPTX::get_sreg_laneid();
  auto is_single_logical_warp = LogicalWarpSize == warp_threads || !_CUDA_VSTD::has_single_bit(LogicalWarpSize);
  auto logical_lane           = static_cast<int>(is_single_logical_warp ? lane : lane % LogicalWarpSize);
  _CCCL_ASSUME(logical_lane >= 0 && logical_lane < LogicalWarpSize);
  return logical_lane;
}

[[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE int logical_warp_id(uint32_t LogicalWarpSize)
{
  _CCCL_ASSERT(LogicalWarpSize <= warp_threads, "invalid logical warp size");
  auto lane                   = _CUDA_VPTX::get_sreg_laneid();
  auto is_single_logical_warp = LogicalWarpSize == warp_threads || !_CUDA_VSTD::has_single_bit(LogicalWarpSize);
  auto logical_warp_id        = static_cast<int>(is_single_logical_warp ? 0 : lane / LogicalWarpSize);
  _CCCL_ASSUME(logical_warp_id >= 0 && logical_warp_id < warp_threads / LogicalWarpSize);
  return logical_warp_id;
}

} // namespace detail
CUB_NAMESPACE_END
