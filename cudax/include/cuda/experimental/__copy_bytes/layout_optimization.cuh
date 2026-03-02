//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_LAYOUT_OPTIMIZATION
#define _CUDAX__COPY_BYTES_LAYOUT_OPTIMIZATION

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/in_range.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/stable_sort.h>
#include <cuda/std/__cstdlib/abs.h>
#include <cuda/std/__numeric/gcd_lcm.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/__copy/raw_tensor_utils.cuh>
#include <cuda/experimental/__copy/types.cuh>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
#if !_CCCL_COMPILER(NVRTC)

//! @brief Compute the alignment of a pointer in bytes.
//!
//! Returns the largest power-of-two that divides the pointer address.
//!
//! @pre `__ptr` must not be null.
//!
//! @param[in] __ptr Pointer to check
//! @return Alignment in bytes (always a power of two)
[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t __ptr_alignment(const void* __ptr) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "Pointer is null");
  const auto __addr = reinterpret_cast<::cuda::std::uintptr_t>(__ptr);
  return static_cast<::cuda::std::size_t>(__addr & (~__addr + 1));
}

//! @brief Sort modes of a src/dst tensor pair by dst's ascending absolute stride.
//!
//! The same permutation is applied to both tensors, so shapes remain identical after sorting.
//! After sorting, the destination's stride-1 mode (if any) is at index 0.
//!
//! @pre `__src.__rank == __dst.__rank`, in [1, _MaxRank].
//! @pre Both tensors must have the same shapes (mode-by-mode).
//!
//! @param[in,out] __src Source tensor (shapes and strides reordered)
//! @param[in,out] __dst Destination tensor (shapes and strides reordered by same permutation)
template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __sort_by_stride_paired(__raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src,
                                            __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst) noexcept
{
  using ::cuda::std::size_t;
  _CCCL_ASSERT(::cuda::in_range(__src.__rank, size_t{1}, _MaxRank), "Invalid tensor rank");
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__extents == __dst.__extents, "Source and destination extents must be identical");
  struct __mode
  {
    _Ep __extent;
    _Sp __src_stride;
    _Sp __dst_stride;
  };
  ::cuda::std::array<__mode, _MaxRank> __modes{};
  for (size_t __i = 0; __i < __src.__rank; ++__i)
  {
    __modes[__i] = {__src.__extents[__i], __src.__strides[__i], __dst.__strides[__i]};
  }
  ::cuda::std::stable_sort(__modes.begin(), __modes.begin() + __src.__rank, [](auto __a, auto __b) {
    return ::cuda::std::abs(__a.__dst_stride) < ::cuda::std::abs(__b.__dst_stride);
  });
  for (size_t __i = 0; __i < __src.__rank; ++__i)
  {
    __src.__extents[__i] = __modes[__i].__extent;
    __src.__strides[__i] = __modes[__i].__src_stride;
    __dst.__strides[__i] = __modes[__i].__dst_stride;
  }
  __dst.__extents = __src.__extents;
}

//! @brief Flip modes where both tensors have negative strides to positive.
//!
//! For each mode where both src and dst strides are negative, reverses the iteration direction:
//! negates both strides and advances both data pointers by `(shape - 1) * stride`
//! (a negative offset, moving pointers backward).
//!
//! A negative stride means the mode is iterated in reverse (high address → low).
//! Flipping a stride (negating it + adjusting the pointer) reverses the iteration direction for that mode
//!
//! @pre `__src.__rank == __dst.__rank`, in [1, _MaxRank].
//! @pre Both tensors must have the same shapes (mode-by-mode).
//!
//! @param[in,out] __src Source tensor (data pointer and strides updated)
//! @param[in,out] __dst Destination tensor (data pointer and strides updated)
template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __flip_negative_strides_paired(__raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src,
                                                   __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__extents == __dst.__extents, "Source and destination extents must be identical");
  for (::cuda::std::size_t __i = 0; __i < __src.__rank; ++__i)
  {
    if (__src.__strides[__i] < 0 && __dst.__strides[__i] < 0)
    {
      const auto __adj_src = static_cast<_Sp>(__src.__extents[__i] - 1) * __src.__strides[__i];
      const auto __adj_dst = static_cast<_Sp>(__dst.__extents[__i] - 1) * __dst.__strides[__i];
      __src.__data += __adj_src;
      __dst.__data += __adj_dst;
      __src.__strides[__i] = -__src.__strides[__i];
      __dst.__strides[__i] = -__dst.__strides[__i];
    }
  }
}

//! @brief Coalesce adjacent contiguous modes in a src/dst tensor pair.
//!
//! Two adjacent modes (i, i+1) are merged when shape[i] * stride[i] == stride[i+1]
//! holds for BOTH tensors. After merging, consumed modes are compacted out and
//! both tensors' ranks are reduced accordingly.
//!
//! @note Coalescing can lose per-mode stride information, e.g.
//! (2,4,16):(80,20,3) --> (8,16):(20,3)
//!
//! @pre `__src.__rank == __dst.__rank`, in [1, _MaxRank].
//! @pre Both tensors must have the same shapes (mode-by-mode).
//! @pre All shapes must be > 1 (no degenerate modes).
//! @pre Destination strides must be sorted in non-descending order.
//!
//! @param[in,out] __src Source tensor (shapes, strides, and rank updated)
//! @param[in,out] __dst Destination tensor (shapes, strides, and rank updated)
template <typename _Ep, typename _Sp, typename _TpSrc, typename _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST_API void __coalesce_paired(__raw_tensor<_Ep, _Sp, _TpSrc, _MaxRank>& __src,
                                      __raw_tensor<_Ep, _Sp, _TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(::cuda::in_range(__src.__rank, ::cuda::std::size_t{1}, _MaxRank), "Invalid tensor rank");
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__extents == __dst.__extents, "Source and destination extents must be identical");
  _CCCL_ASSERT(::cuda::experimental::__has_no_extent1_modes(__src), "All extents must be greater than 1");
  _CCCL_ASSERT(::cuda::experimental::__has_sorted_strides(__dst), "Destination strides must be sorted by dst strides");
  const auto __rank         = __src.__rank;
  auto& __extents           = __src.__extents;
  ::cuda::std::size_t __out = 1;
  for (::cuda::std::size_t __i = 1; __i < __rank; ++__i)
  {
    const auto __prev_extent    = static_cast<_Sp>(__extents[__out - 1]);
    const bool __src_contiguous = (__prev_extent * __src.__strides[__out - 1] == __src.__strides[__i]);
    const bool __dst_contiguous = (__prev_extent * __dst.__strides[__out - 1] == __dst.__strides[__i]);
    if (__src_contiguous && __dst_contiguous)
    {
      __extents[__out - 1] *= __extents[__i];
      continue;
    }
    __extents[__out]       = __extents[__i];
    __src.__strides[__out] = __src.__strides[__i];
    __dst.__strides[__out] = __dst.__strides[__i];
    ++__out;
  }
  __src.__rank    = __out;
  __dst.__rank    = __out;
  __dst.__extents = __extents;
}

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
__max_vector_size_bytes(const __raw_tensor<_Ep, _Sp, _Tp, _MaxRank>& __tensor) noexcept
{
  using ::cuda::std::size_t;
  _CCCL_ASSERT(::cuda::in_range(__tensor.__rank, size_t{1}, _MaxRank), "Invalid tensor rank");
  _CCCL_ASSERT(::cuda::experimental::__has_no_extent1_modes(__tensor), "All extents must be different from 1");
  const auto& __extents = __tensor.__extents;
  const auto& __strides = __tensor.__strides;
  if (__strides[0] != 1)
  {
    return sizeof(_Tp);
  }
  // (1) pointer alignment
  size_t __alignment = ::cuda::experimental::__ptr_alignment(__tensor.__data);
  // (2) alignment over all strides
  for (size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    const auto __stride = ::cuda::std::abs(__strides[__i]);
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

#endif // !_CCCL_COMPILER(NVRTC)
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_LAYOUT_OPTIMIZATION
