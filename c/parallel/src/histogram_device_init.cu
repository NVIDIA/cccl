//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This is a concept implementation of moving ScaleTransform initialization to device side
// to achieve true type erasure in c.parallel

#include <cub/device/dispatch/kernels/histogram.cuh>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN
namespace detail::histogram {

// Extended Transforms that supports device-side initialization
template <typename LevelT, typename OffsetT, typename SampleT>
struct TransformsDeviceInit : Transforms<LevelT, OffsetT, SampleT>
{
  // Parameters needed for ScaleTransform initialization
  struct ScaleTransformParams {
    int num_levels;
    cccl_value_t lower_level;
    cccl_value_t upper_level;
  };
  
  // Device-side ScaleTransform that can be initialized on device
  struct ScaleTransformDeviceInit : Transforms<LevelT, OffsetT, SampleT>::ScaleTransform
  {
    using Base = typename Transforms<LevelT, OffsetT, SampleT>::ScaleTransform;
    
    // Device-side initialization method
    _CCCL_DEVICE _CCCL_FORCEINLINE cudaError_t InitDevice(
      int num_levels, 
      const cccl_value_t& lower_level_val,
      const cccl_value_t& upper_level_val)
    {
      // Extract values from cccl_value_t based on LevelT
      LevelT lower_level = *static_cast<const LevelT*>(lower_level_val.state);
      LevelT upper_level = *static_cast<const LevelT*>(upper_level_val.state);
      
      this->m_max = static_cast<typename Base::CommonT>(upper_level);
      this->m_min = static_cast<typename Base::CommonT>(lower_level);

      // Check whether accurate bin computation for an integral sample type may overflow
      if (this->MayOverflow(static_cast<typename Base::CommonT>(num_levels - 1), 
                           ::cuda::std::is_integral<typename Base::CommonT>{}))
      {
        return cudaErrorInvalidValue;
      }

      this->m_scale = this->ComputeScale(num_levels, this->m_max, this->m_min);
      return cudaSuccess;
    }
  };
  
  // Pass-through bin transform operator (unchanged)
  using PassThruTransform = typename Transforms<LevelT, OffsetT, SampleT>::PassThruTransform;
  
  // Searches for bin given a list of bin-boundary levels (unchanged)
  template <typename LevelIteratorT>
  using SearchTransform = typename Transforms<LevelT, OffsetT, SampleT>::template SearchTransform<LevelIteratorT>;
};

} // namespace detail::histogram
CUB_NAMESPACE_END