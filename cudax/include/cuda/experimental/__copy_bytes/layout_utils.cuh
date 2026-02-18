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

#include <cuda/__fwd/hierarchy.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__numeric/gcd_lcm.h>
#include <cuda/std/array>

#include <cute/layout.hpp>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{

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
inline constexpr int __config_block_threads = __extents_product<
  typename Config::hierarchy_type::template level_desc_type<::cuda::block_level>::extents_type>::value;

//! @brief Merge adjacent contiguous modes in-place.
//!
//! When shape[i] * stride[i] == stride[i+1], modes i and i+1 are contiguous
//! and can be merged into a single mode. Consumed modes are set to shape=1, stride=0.
//!
//! @param shape  Array of mode shapes (modified in-place)
//! @param stride Array of mode strides (modified in-place)
//! @param rank   Number of modes
//!
//! @pre Modes must be sorted by ascending absolute stride before calling.
template <size_t N>
_CCCL_HOST void runtime_coalesce(::cuda::std::array<int, N>& shape, ::cuda::std::array<int, N>& stride, int rank)
{
  for (int i = 0; i < rank - 1; ++i)
  {
    if (shape[i] == 1)
    {
      continue;
    }
    if (shape[i] * stride[i] == stride[i + 1])
    {
      shape[i + 1]  = shape[i] * shape[i + 1];
      stride[i + 1] = stride[i];
      shape[i]      = 1;
      stride[i]     = 0;
    }
  }
}

//! @brief Sort modes by ascending absolute stride (insertion sort).
//!
//! After sorting, the stride-1 (fastest-changing) mode is first,
//! ensuring threads accessing consecutive linear indices hit consecutive addresses.
//!
//! @param shape  Array of mode shapes (reordered in-place)
//! @param stride Array of mode strides (reordered in-place)
//! @param rank   Number of modes
template <size_t N>
_CCCL_HOST void sort_modes_by_stride(::cuda::std::array<int, N>& shape, ::cuda::std::array<int, N>& stride, int rank)
{
  for (int i = 1; i < rank; ++i)
  {
    int s  = shape[i];
    int st = stride[i];
    int j  = i - 1;
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
//! Both layouts must have been coalesced and sorted before comparison.
//! Modes with shape==1 are skipped (they carry no data).
//!
//! @return true if both layouts have the same effective shapes and strides
template <size_t N>
_CCCL_HOST bool layouts_match(
  const ::cuda::std::array<int, N>& src_shape,
  const ::cuda::std::array<int, N>& src_stride,
  const ::cuda::std::array<int, N>& dst_shape,
  const ::cuda::std::array<int, N>& dst_stride,
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
//! @param shape      Array of mode shapes (after coalescing/sorting)
//! @param stride     Array of mode strides (after coalescing/sorting)
//! @param rank       Number of modes
//! @param elem_bytes sizeof(T) for the element type
//! @return Maximum safe vectorization width in bytes
template <typename T, size_t N>
_CCCL_HOST int compute_vec_bytes(
  const T* src_ptr,
  const T* dst_ptr,
  const ::cuda::std::array<int, N>& shape,
  const ::cuda::std::array<int, N>& stride,
  int rank,
  int elem_bytes)
{
  int vec_bytes = ::cuda::std::gcd(ptr_alignment(src_ptr), ptr_alignment(dst_ptr));

  for (int i = 0; i < rank; ++i)
  {
    if (shape[i] <= 1)
    {
      continue;
    }
    int stride_bytes = (stride[i] < 0 ? -stride[i] : stride[i]) * elem_bytes;
    vec_bytes        = ::cuda::std::gcd(vec_bytes, stride_bytes);
  }

  if (vec_bytes > 16)
  {
    vec_bytes = 16;
  }

  int elems_per_vec = vec_bytes / elem_bytes;
  for (int i = 0; i < rank; ++i)
  {
    if (shape[i] <= 1)
    {
      continue;
    }
    while (elems_per_vec > 1 && shape[i] % elems_per_vec != 0)
    {
      elems_per_vec /= 2;
    }
  }
  vec_bytes = elems_per_vec * elem_bytes;

  return vec_bytes;
}

//! @brief Extract shapes and strides from a CuTe layout into plain arrays.
//!
//! For dynamic layouts, this extracts runtime values. For static layouts,
//! it converts compile-time values to runtime integers.
//!
//! @tparam Layout CuTe layout type
//! @param layout  The CuTe layout to extract from
//! @param shape   Output array for shapes (must have at least rank elements)
//! @param stride  Output array for strides (must have at least rank elements)
template <typename Layout, size_t N>
_CCCL_HOST void
extract_layout(const Layout& layout, ::cuda::std::array<int, N>& out_shape, ::cuda::std::array<int, N>& out_stride)
{
  constexpr int R = decltype(::cute::rank(layout))::value;
  ::cute::for_each(::cute::make_seq<R>{}, [&](auto i) {
    out_shape[i]  = static_cast<int>(::cute::get<i>(::cute::shape(layout)));
    out_stride[i] = static_cast<int>(::cute::get<i>(::cute::stride(layout)));
  });
}

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_LAYOUT_UTILS
