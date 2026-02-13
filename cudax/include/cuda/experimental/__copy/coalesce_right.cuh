//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_COALESCE_RIGHT_H
#define __CUDAX_COPY_COALESCE_RIGHT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/fill.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Runtime coalesce-right for dynamic shape/stride layouts.
 *
 * Merges right-adjacent contiguous modes and returns a compacted layout descriptor.
 *
 * @par Algorithm
 * 1. Traverse modes right-to-left, skipping size-1 modes.
 * 2. Merge into the previous output mode when `stride[i] == curr_stride`,
 *    where `curr_stride = out_stride * out_shape` of the most recent mode.
 * 3. Compact valid modes to the beginning of the arrays.
 */
template <typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_Tp, _MaxRank>
__coalesce_right(const __raw_tensor<_Tp, _MaxRank>& __tensor_input) noexcept
{
  using ::cuda::std::int64_t;
  __raw_tensor<_Tp, _MaxRank> __result{__tensor_input.__data};
  auto& __shapes  = __result.__shapes;
  auto& __strides = __result.__strides;
  ::cuda::std::fill(__shapes.begin(), __shapes.end(), ::cuda::std::size_t{1});
  ::cuda::std::fill(__strides.begin(), __strides.end(), int64_t{0});
  const auto __rank  = static_cast<int>(__tensor_input.__rank);
  int __out          = __rank - 1;
  auto __curr_stride = int64_t{0};
  for (int __i = __rank - 1; __i >= 0; --__i)
  {
    if (__tensor_input.__shapes[__i] == 1)
    {
      continue;
    }
    // Merge contiguous modes
    if (__out < __rank - 1 && __tensor_input.__strides[__i] == __curr_stride)
    {
      __result.__shapes[__out + 1] *= __tensor_input.__shapes[__i];
    }
    else
    {
      __result.__shapes[__out]  = __tensor_input.__shapes[__i];
      __result.__strides[__out] = __tensor_input.__strides[__i];
      --__out;
    }
    __curr_stride = __result.__strides[__out + 1] * static_cast<int64_t>(__result.__shapes[__out + 1]);
  }
  // shift/compact the result at the beginning of the array
  const auto __first_valid = __out + 1;
  const auto __rank_out    = __rank - __first_valid;
  for (int __j = 0; __j < __rank_out; ++__j)
  {
    __result.__shapes[__j]  = __result.__shapes[__first_valid + __j];
    __result.__strides[__j] = __result.__strides[__first_valid + __j];
  }
  __result.__rank = ::cuda::std::max(__rank_out, 1);
  return __result;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_COALESCE_RIGHT_H
