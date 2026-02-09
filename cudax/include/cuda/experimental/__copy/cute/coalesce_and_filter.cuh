//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_FILTER_H
#define __CUDAX_COPY_CUTE_FILTER_H

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
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/cute/utils.cuh>

#  include <cute/layout.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _Shape, class _Stride>
[[nodiscard]] _CCCL_HOST_API auto
__simplify_dynamic(const ::cute::Layout<_Shape, _Stride>& __layout, bool __filter_zeros) noexcept
{
  constexpr auto __rank = __rank_v<_Shape>;
  if constexpr (__rank <= 1)
  {
    return __layout;
  }
  else
  {
    using ::cuda::std::int64_t;
    constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};
    ::cuda::std::array<int64_t, __rank> __shapes;
    ::cuda::std::array<int64_t, __rank> __strides;
    ::cuda::std::array<int64_t, __rank> __result_shapes;
    ::cuda::std::array<int64_t, __rank> __result_strides{};
    ::cuda::experimental::__init_layout(__layout.shape(), __layout.stride(), __shapes, __strides, __rank_seq);
    ::cuda::std::fill(__result_shapes.begin(), __result_shapes.end(), int64_t{1});
    ::cuda::std::size_t __out = 0;
    for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
    {
      if (__shapes[__i] == 1 || (__filter_zeros && __strides[__i] == 0))
      {
        continue;
      }
      if (__out > 0 && __result_strides[__out - 1] * __result_shapes[__out - 1] == __strides[__i])
      {
        __result_shapes[__out - 1] *= __shapes[__i];
      }
      else
      {
        __result_shapes[__out]  = __shapes[__i];
        __result_strides[__out] = __strides[__i];
        ++__out;
      }
    }
    const auto __s = ::cuda::experimental::__to_cute_tuple(__result_shapes, __rank_seq);
    const auto __d = ::cuda::experimental::__to_cute_tuple(__result_strides, __rank_seq);
    return ::cute::make_layout(__s, __d);
  }
}

/**
 * @brief Runtime version of CuTe's `coalesce` for layouts with dynamic shapes/strides.
 *
 * Produces a layout with the same number of modes as the input layout, but with contiguous modes merged.
 *
 * @par Algorithm
 * 1. Iterate over modes, skipping size-1 modes.
 * 2. Merge forward-contiguous modes where `stride[i] * shape[i] == stride[i+1]`.
 * 3. Pad remaining output modes with `(1, 0)` (identity).
 */
template <class _Shape, class _Stride>
[[nodiscard]] _CCCL_HOST_API auto __coalesce_dynamic(const ::cute::Layout<_Shape, _Stride>& __layout) noexcept
{
  return ::cuda::experimental::__simplify_dynamic(__layout, false);
}

/**
 * @brief Runtime version of CuTe's `filter` for layouts with dynamic shapes/strides.
 *
 * Produces a layout with the same number of modes as the input layout, but with contiguous modes merged and no size-0
 * modes. The algorithm is similar to `coalesce`.
 */
template <class _Shape, class _Stride>
[[nodiscard]] _CCCL_HOST_API auto __filter_dynamic(const ::cute::Layout<_Shape, _Stride>& __layout) noexcept
{
  return ::cuda::experimental::__simplify_dynamic(__layout, true);
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_FILTER_H
