//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_MDSPAN_D2D_H
#define __CUDAX_COPY_MDSPAN_D2D_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cub/device/device_transform.cuh>

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__functional/address_stability.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/traits.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__algorithm/max.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__host_stdlib/stdexcept>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/__type_traits/common_type.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>

#  include <cuda/experimental/__copy_bytes/memcpy_batch_tiles.cuh>
#  include <cuda/experimental/__copy_bytes/simplify_paired.cuh>
#  include <cuda/experimental/__copy_bytes/tensor_query.cuh>

#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Compute the maximum vectorization width in bytes for a raw tensor.
//!
//! Expects mode 0 to be the contiguous mode (stride == 1), as established by
//! @ref __sort_by_stride_paired. Computes the largest power-of-two vector width such that:
//! - The pointer is aligned to that width.
//! - All non-contiguous strides (in bytes) are divisible by it.
//! - The contiguous mode's shape is divisible by the element count.
//! The result is capped at 16 bytes. If mode 0 is not contiguous, returns sizeof(_Tp).
//!
//! @pre `__tensor.__rank` is in [1, _MaxRank].
//! @pre All shapes must be > 1 (no degenerate modes).
//! @pre Strides are sorted by @ref __sort_by_stride_paired (mode 0 has the smallest absolute stride).
//!
//! @param[in] __tensor Raw tensor with strides sorted by @ref __sort_by_stride_paired
//! @return Maximum safe vectorization width in bytes, in [sizeof(_Tp), 16]
template <typename _Ep, typename _Sp, typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__max_alignment(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  using ::cuda::std::size_t;
  namespace cudax = ::cuda::experimental;
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, size_t{1}, _MaxRank), "Invalid tensor rank");
  const auto& __extents = __tensor.__extents;
  const auto& __strides = __tensor.__strides;
  if (__strides[0] != 1)
  {
    return sizeof(_Tp);
  }
  // (1) pointer alignment
  size_t __alignment = ::cuda::__ptr_alignment(__tensor.__data);
  // (2) alignment over all strides
  for (size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    const auto __stride = cudax::__abs_integer(__strides[__i]);
    if (__stride != 1)
    {
      const auto __stride_bytes = static_cast<size_t>(__stride) * sizeof(_Tp);
      __alignment               = ::cuda::std::gcd(__alignment, __stride_bytes);
    }
  }
  _CCCL_ASSERT(__alignment % sizeof(_Tp) == 0, "Maximum vector size is not a multiple of the element size");
  // (3) Compute the number of items per vector over the contiguous mode
  size_t __items_per_vector = __alignment / sizeof(_Tp);
  __items_per_vector        = ::cuda::std::gcd(__items_per_vector, static_cast<size_t>(__extents[0]));
  return __items_per_vector * sizeof(_Tp);
}

} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_MDSPAN_D2D_H
