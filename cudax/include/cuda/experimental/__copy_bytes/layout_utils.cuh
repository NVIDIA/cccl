//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__COPY_BYTES_LAYOUT_UTILS
#define _CUDAX__COPY_BYTES_LAYOUT_UTILS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__cmath/pow2.h>
#include <cuda/__fwd/hierarchy.h>
#include <cuda/__utility/static_for.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__cstdlib/abs.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__numeric/gcd_lcm.h>
#include <cuda/std/array>

#include <cuda/std/__cccl/prologue.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>

namespace cuda::experimental
{
//! @brief Create a CuTe tensor backed by global memory.
template <typename _Tp, typename Layout>
__host__ __device__ auto make_gmem_tensor(_Tp* __ptr, const Layout& layout)
{
  return ::cute::make_tensor(::cute::make_gmem_ptr(__ptr), layout);
}

//! @brief Create a compact row-major CuTe layout with all-static sizes.
template <int... Sizes>
__host__ __device__ constexpr auto make_static_layout()
{
  return ::cute::make_layout(::cute::make_shape(::cute::Int<Sizes>{}...));
}

//! @brief Compute the product of static extents from a cuda::std::extents type.
template <typename>
struct __extents_product;

template <typename IndexType, ::cuda::std::size_t... Es>
struct __extents_product<::cuda::std::extents<IndexType, Es...>>
{
  static constexpr int value = static_cast<int>((Es * ...));
};

//! @brief Extract the compile-time block thread count from a kernel_config type.
//!
//! Uses the hierarchy type embedded in Config to find the block-level descriptor,
//! then computes the product of its static extents.
template <typename Config>
inline constexpr int __config_block_threads =
  __extents_product<typename Config::hierarchy_type::template level_desc_type<::cuda::block_level>::extents_type>::value;

//! @brief Merge adjacent contiguous modes in-place.
//!
//! When shape[__i] * stride[__i] == stride[__i+1], modes __i and __i+1 are contiguous
//! and can be merged into a single mode. Consumed modes are set to shape=1, stride=0.
//!
//! @param shape  Array of mode __shapes (modified in-place)
//! @param stride Array of mode __strides (modified in-place)
//! @param rank   Number of modes
//!
//! @pre Modes must be sorted by ascending absolute stride before calling.
template <size_t _Np>
_CCCL_HOST void runtime_coalesce(::cuda::std::array<int, _Np>& shape, ::cuda::std::array<int, _Np>& stride, int rank)
{
  for (int __i = 0; __i < rank - 1; ++__i)
  {
    if (shape[__i] == 1)
    {
      continue;
    }
    if (shape[__i] * stride[__i] == stride[__i + 1])
    {
      shape[__i + 1]  = shape[__i] * shape[__i + 1];
      stride[__i + 1] = stride[__i];
      shape[__i]      = 1;
      stride[__i]     = 0;
    }
  }
}

//! @brief Sort modes by ascending absolute stride (insertion sort).
//!
//! After sorting, the stride-1 (fastest-changing) mode is first,
//! ensuring threads accessing consecutive linear indices hit consecutive addresses.
//!
//! @param shape  Array of mode __shapes (reordered in-place)
//! @param stride Array of mode __strides (reordered in-place)
//! @param rank   Number of modes
template <size_t _Np>
_CCCL_HOST void sort_modes_by_stride(::cuda::std::array<int, _Np>& shape, ::cuda::std::array<int, _Np>& stride, int rank)
{
  for (int __i = 1; __i < rank; ++__i)
  {
    int s  = shape[__i];
    int st = stride[__i];
    int j  = __i - 1;
    while (j >= 0 && (stride[j] < 0 ? -stride[j] : stride[j]) > (st < 0 ? -st : st))
    {
      shape[j + 1]  = shape[j];
      stride[j + 1] = stride[j];
      --j;
    }
    shape[j + 1]  = s;
    stride[j + 1] = st;
  }
}

//! @brief Check whether two preprocessed layouts are effectively identical.
//!
//! Both layouts must have been sorted before comparison.
//! Modes with shape==1 are skipped (they carry no data).
//!
//! @return true if both layouts have the same effective __shapes and __strides
template <size_t _Np>
_CCCL_HOST bool layouts_match(
  const ::cuda::std::array<int, _Np>& src_shape,
  const ::cuda::std::array<int, _Np>& src_stride,
  const ::cuda::std::array<int, _Np>& dst_shape,
  const ::cuda::std::array<int, _Np>& dst_stride,
  int rank)
{
  int si = 0;
  int di = 0;
  while (si < rank && di < rank)
  {
    while (si < rank && src_shape[si] == 1)
    {
      ++si;
    }
    while (di < rank && dst_shape[di] == 1)
    {
      ++di;
    }
    if (si >= rank && di >= rank)
    {
      return true;
    }
    if (si >= rank || di >= rank)
    {
      return false;
    }
    if (src_shape[si] != dst_shape[di] || src_stride[si] != dst_stride[di])
    {
      return false;
    }
    ++si;
    ++di;
  }
  while (si < rank && src_shape[si] == 1)
  {
    ++si;
  }
  while (di < rank && dst_shape[di] == 1)
  {
    ++di;
  }
  return si >= rank && di >= rank;
}

//! @brief Compute the alignment of a pointer in bytes.
//!
//! Returns the largest power-of-two that divides the pointer address.
//! For a null pointer, returns 16 (maximum vectorization width).
//!
//! @param p Pointer to check
//! @return Alignment in bytes (always a power of two, at most the true alignment)
_CCCL_HOST inline int ptr_alignment(const void* p)
{
  auto addr = reinterpret_cast<uintptr_t>(p);
  if (addr == 0)
  {
    return 16;
  }
  return static_cast<int>(addr & (~addr + 1));
}

//! @brief Compute the maximum vectorization width in bytes.
//!
//! Considers pointer alignment, all stride alignments, and shape divisibility.
//! The result is the largest power-of-two byte width (up to 16) that is safe
//! for recast<VecType> across all modes.
//!
//! @param src_ptr    Source pointer (used for alignment check)
//! @param dst_ptr    Destination pointer (used for alignment check)
//! @param shape      Array of mode __shapes (after coalescing/sorting)
//! @param stride     Array of mode __strides (after coalescing/sorting)
//! @param rank       Number of modes
//! @param elem_bytes sizeof(_Tp) for the element type
//! @return Maximum safe vectorization width in bytes
template <typename _Tp, size_t _Np>
_CCCL_HOST int max_vector_size(
  const _Tp* src_ptr,
  const _Tp* dst_ptr,
  const ::cuda::std::array<int, _Np>& shape,
  const ::cuda::std::array<int, _Np>& stride,
  int rank,
  int elem_bytes)
{
  int __vector_bytes = ::cuda::std::gcd(ptr_alignment(src_ptr), ptr_alignment(dst_ptr));

  for (int __i = 0; __i < rank; ++__i)
  {
    if (shape[__i] <= 1)
    {
      continue;
    }
    int __stride_bytes = (stride[__i] < 0 ? -stride[__i] : stride[__i]) * elem_bytes;
    __vector_bytes     = ::cuda::std::gcd(__vector_bytes, __stride_bytes);
  }

  if (__vector_bytes > 16)
  {
    __vector_bytes = 16;
  }

  int __items_per_vector = __vector_bytes / elem_bytes;
  for (int __i = 0; __i < rank; ++__i)
  {
    if (shape[__i] <= 1)
    {
      continue;
    }
    __items_per_vector = ::cuda::std::gcd(__items_per_vector, shape[__i]);
  }
  __vector_bytes = __items_per_vector * elem_bytes;

  return __vector_bytes;
}

//! @brief Compute the maximum vectorization width in bytes for a single tensor.
//!
//! Considers pointer alignment, all stride alignments, and shape divisibility.
//! The result is the largest power-of-two byte width (up to 16) that is safe
//! for recast<VecType>.
//!
//! @tparam _Tp Element type (sizeof(_Tp) determines the element width)
//! @tparam _Np Array capacity (rank is deduced as _Np)
//! @param __ptr       Pointer to tensor data (used for alignment check)
//! @param shape     Array of mode __shapes (after coalescing/sorting)
//! @param stride    Array of mode __strides (after coalescing/sorting)
//! @return Maximum safe vectorization width in bytes
template <typename _Tp, typename _ShapeIndex, typename _StrideIndex, ::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST ::cuda::std::size_t __max_vector_size(
  const _Tp* __ptr,
  const ::cuda::std::array<_ShapeIndex, _Np>& __shapes,
  const ::cuda::std::array<_StrideIndex, _Np>& __strides) noexcept
{
  using ::cuda::std::size_t;
  size_t __vector_bytes = ptr_alignment(__ptr);
  for (size_t __i = 0; __i < _Np; ++__i)
  {
    if (__shapes[__i] > 1)
    {
      const auto __stride_bytes = static_cast<size_t>(::cuda::std::abs(__strides[__i])) * sizeof(_Tp);
      __vector_bytes            = ::cuda::std::gcd(__vector_bytes, __stride_bytes);
    }
  }
  _CCCL_ASSERT(__vector_bytes % sizeof(_Tp) == 0, "Maximum vector size is not a multiple of the element size");
  size_t __items_per_vector = __vector_bytes / sizeof(_Tp);
  for (const auto __shape : __shapes)
  {
    if (__shape > 1)
    {
      __items_per_vector = ::cuda::std::gcd(__items_per_vector, __shape);
    }
  }
  const auto __result_bytes = __items_per_vector * sizeof(_Tp);
  _CCCL_ASSERT(::cuda::is_power_of_two(__result_bytes), "Maximum vector size is 16 bytes");
  constexpr size_t __max_vector_bytes = 16;
  return ::cuda::std::min(__result_bytes, __max_vector_bytes);
}

//! @brief Extract __shapes and __strides from a CuTe layout into plain arrays.
//!
//! For dynamic layouts, this extracts runtime values. For static layouts,
//! it converts compile-time values to runtime integers.
//!
//! @tparam Layout CuTe layout type
//! @param layout  The CuTe layout to extract from
//! @param shape   Output array for __shapes (must have at least rank elements)
//! @param stride  Output array for __strides (must have at least rank elements)
template <typename Layout, typename _ShapeIndex, typename _StrideIndex, ::cuda::std::size_t _Np>
_CCCL_HOST void __extract_layout(const Layout& layout,
                                 ::cuda::std::array<_ShapeIndex, _Np>& __out_shape,
                                 ::cuda::std::array<_StrideIndex, _Np>& __out_stride) noexcept
{
  constexpr auto __rank = decltype(::cute::rank(layout))::value;
  ::cuda::static_for<__rank>([&](auto __i) {
    __out_shape[__i]  = ::cute::shape<__i>(layout);
    __out_stride[__i] = ::cute::stride<__i>(layout);
  });
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_LAYOUT_UTILS
