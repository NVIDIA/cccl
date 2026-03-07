//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_COALESCE_LEFT_H
#define __CUDAX_COPY_COALESCE_LEFT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>

#  include <cuda/experimental/__copy/types.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Runtime coalesce-left for dynamic shape/stride layouts.
 *
 * Merges left-adjacent contiguous modes and returns a compacted layout descriptor.
 *
 * @par Algorithm
 * 1. Traverse modes left-to-right, skipping size-1 modes.
 * 2. Merge into the previous output mode when `stride[i] == curr_stride`,
 *    where `curr_stride = out_stride * out_shape` of the most recent mode.
 * 3. Append non-contiguous modes without reordering.
 */
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>
__coalesce_left(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor_input) noexcept
{
  using __raw_tensor_t = __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>;
  using __extent_t     = typename __raw_tensor_t::__unsigned_extent_t;
  if (__tensor_input.__rank <= 1)
  {
    return __tensor_input;
  }

  __raw_tensor_t __result{__tensor_input.__data, 0, {}, {}};
  bool __have_output         = false;
  ::cuda::std::size_t __out  = 0;

  for (::cuda::std::size_t __i = 0; __i < __tensor_input.__rank; ++__i)
  {
    if (__tensor_input.__extents[__i] == 1)
    {
      continue;
    }

    if (!__have_output)
    {
      __result.__extents[0] = __tensor_input.__extents[__i];
      __result.__strides[0] = __tensor_input.__strides[__i];
      __have_output         = true;
      continue;
    }

    const auto __prev_extent = static_cast<_Sp>(__result.__extents[__out]);
    const auto __curr_stride = __result.__strides[__out] * __prev_extent;
    if (__tensor_input.__strides[__i] == __curr_stride)
    {
      __result.__extents[__out] *= __tensor_input.__extents[__i];
    }
    else
    {
      ++__out;
      __result.__extents[__out] = __tensor_input.__extents[__i];
      __result.__strides[__out] = __tensor_input.__strides[__i];
    }
  }

  if (__have_output)
  {
    __result.__rank = __out + 1;
  }
  else
  {
    __result.__extents[0] = __extent_t{1};
    __result.__strides[0] = _Sp{0};
    __result.__rank       = 1;
  }
  return __result;
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_COALESCE_LEFT_H
