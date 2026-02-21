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
#include <cuda/__utility/static_for.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/stable_sort.h>
#include <cuda/std/__cstdlib/abs.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__numeric/gcd_lcm.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>

#include <cuda/experimental/__copy/types.cuh>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
//
#include <cuda/std/__cccl/prologue.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
//
#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
//! @brief Create a CuTe tensor backed by global memory.
template <class _Tp, class _Layout>
[[nodiscard]] _CCCL_HOST_DEVICE auto make_gmem_tensor(_Tp* __ptr, const _Layout& __layout) noexcept
{
  return ::cute::make_tensor(::cute::make_gmem_ptr(__ptr), __layout);
}

//! @brief Create a compact row-major CuTe layout with all-static sizes.
template <int... _Sizes>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto make_static_layout() noexcept
{
  return ::cute::make_layout(::cute::make_shape(::cute::Int<_Sizes>{}...));
}

//! @brief Compute the product of static extents from a cuda::std::extents type.
template <class>
struct __extents_product;

template <class _IndexType, ::cuda::std::size_t... _Es>
struct __extents_product<::cuda::std::extents<_IndexType, _Es...>>
{
  static constexpr int value = static_cast<int>((_Es * ...));
};

//! @brief Extract the compile-time block thread count from a kernel_config type.
//!
//! Uses the hierarchy type embedded in _Config to find the block-level descriptor,
//! then computes the product of its static extents.
template <class _Config>
inline constexpr int __config_block_threads =
  __extents_product<typename _Config::hierarchy_type::template level_desc_type<::cuda::block_level>::extents_type>::value;

//! @brief Compute the product of static extents from a cuda::std::extents type.
template <typename>
struct __extents_product;

template <class _IndexType, ::cuda::std::size_t... _Es>
struct __extents_product<::cuda::std::extents<_IndexType, _Es...>>
{
  static constexpr int value = static_cast<int>((_Es * ...));
};

//! @brief Extract the compile-time block thread count from a kernel_config type.
//!
//! Uses the hierarchy type embedded in _Config to find the block-level descriptor,
//! then computes the product of its static extents.
template <class _Config>
inline constexpr int __config_block_threads =
  __extents_product<typename _Config::hierarchy_type::template level_desc_type<::cuda::block_level>::extents_type>::value;

//! @brief Merge adjacent contiguous modes in-place.
//!
//! When shape[i] * stride[i] == stride[i+1], modes i and i+1 are contiguous
//! and can be merged into a single mode. Consumed modes are set to shape=1, stride=0.
//!
//! @param[in,out] __shapes Array of mode shapes (modified in-place)
//! @param[in,out] __stride Array of mode strides (modified in-place)
//! @param[in]     __rank   Number of modes
//!
//! @pre Modes must be sorted by ascending absolute stride before calling.
template <::cuda::std::size_t _Np>
_CCCL_HOST void
runtime_coalesce(::cuda::std::array<int, _Np>& __shapes, ::cuda::std::array<int, _Np>& __stride, int __rank) noexcept
{
  for (int __i = 0; __i < __rank - 1; ++__i)
  {
    if (__shapes[__i] == 1)
    {
      continue;
    }
    if (__shapes[__i] * __stride[__i] == __stride[__i + 1])
    {
      __shapes[__i + 1] = __shapes[__i] * __shapes[__i + 1];
      __stride[__i + 1] = __stride[__i];
      __shapes[__i]     = 1;
      __stride[__i]     = 0;
    }
  }
}

//! @brief Sort modes by ascending absolute stride (insertion sort).
//!
//! After sorting, the stride-1 (fastest-changing) mode is first,
//! ensuring threads accessing consecutive linear indices hit consecutive addresses.
//!
//! @param[in,out] __shapes Array of mode shapes (reordered in-place)
//! @param[in,out] __stride Array of mode strides (reordered in-place)
//! @param[in]     __rank   Number of modes
template <::cuda::std::size_t _Np>
_CCCL_HOST void sort_modes_by_stride(
  ::cuda::std::array<int, _Np>& __shapes, ::cuda::std::array<int, _Np>& __stride, int __rank) noexcept
{
  for (int __i = 1; __i < __rank; ++__i)
  {
    const int __shape = __shapes[__i];
    const int __st    = __stride[__i];
    int __j           = __i - 1;
    while (__j >= 0 && ::cuda::std::abs(__stride[__j]) > ::cuda::std::abs(__st))
    {
      __shapes[__j + 1] = __shapes[__j];
      __stride[__j + 1] = __stride[__j];
      --__j;
    }
    __shapes[__j + 1] = __shape;
    __stride[__j + 1] = __st;
  }
}

//! @brief Check whether two preprocessed layouts are effectively identical.
//!
//! Both layouts must have been sorted before comparison.
//! Modes with shape==1 are skipped (they carry no data).
//!
//! @param[in] __src_shape   Source shape array
//! @param[in] __src_strides Source stride array
//! @param[in] __dst_shape   Destination shape array
//! @param[in] __dst_strides Destination stride array
//! @param[in] __rank        Number of modes
//! @return true if both layouts have the same effective shapes and strides
template <::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST bool layouts_match(
  const ::cuda::std::array<int, _Np>& __src_shape,
  const ::cuda::std::array<int, _Np>& __src_strides,
  const ::cuda::std::array<int, _Np>& __dst_shape,
  const ::cuda::std::array<int, _Np>& __dst_strides,
  int __rank) noexcept
{
  int __si = 0;
  int __di = 0;
  while (__si < __rank && __di < __rank)
  {
    while (__si < __rank && __src_shape[__si] == 1)
    {
      ++__si;
    }
    while (__di < __rank && __dst_shape[__di] == 1)
    {
      ++__di;
    }
    if (__si >= __rank && __di >= __rank)
    {
      return true;
    }
    if (__si >= __rank || __di >= __rank)
    {
      return false;
    }
    if (__src_shape[__si] != __dst_shape[__di] || __src_strides[__si] != __dst_strides[__di])
    {
      return false;
    }
    ++__si;
    ++__di;
  }
  while (__si < __rank && __src_shape[__si] == 1)
  {
    ++__si;
  }
  while (__di < __rank && __dst_shape[__di] == 1)
  {
    ++__di;
  }
  return __si >= __rank && __di >= __rank;
}

//! @brief Compute the alignment of a pointer in bytes.
//!
//! Returns the largest power-of-two that divides the pointer address.
//! For a null pointer, returns 16 (maximum vectorization width).
//!
//! @param[in] __ptr Pointer to check
//! @return Alignment in bytes (always a power of two, at most the true alignment)
[[nodiscard]] _CCCL_HOST inline int ptr_alignment(const void* __ptr) noexcept
{
  _CCCL_ASSERT(__ptr != nullptr, "Pointer is null");
  const auto __addr = reinterpret_cast<::cuda::std::uintptr_t>(__ptr);
  return static_cast<int>(__addr & (~__addr + 1));
}

#if !_CCCL_COMPILER(NVRTC)

inline constexpr auto __remove_extent1_mode = ::cuda::std::true_type{};

//! @brief Construct a raw tensor from a pointer and a CuTe layout.
//!
//! Extracts runtime shapes and strides from the CuTe layout into a __raw_tensor.
//! For static layouts, compile-time values are converted to runtime integers.
//!
//! @tparam _MaxRank Maximum rank capacity for the raw tensor
//! @tparam _Tp      Element type (deduced from pointer)
//! @tparam _Layout  CuTe layout type
//! @param[in] __data   Pointer to tensor data
//! @param[in] __layout The CuTe layout to extract from
//! @return A __raw_tensor populated with the layout's shapes and strides
template <::cuda::std::size_t _MaxRank, class _Tp, class _Layout, bool _RemoveExtent1 = false>
[[nodiscard]] _CCCL_HOST __raw_tensor<_Tp, _MaxRank>
__to_raw_tensor(_Tp* __data, const _Layout& __layout, ::cuda::std::bool_constant<_RemoveExtent1> = {}) noexcept
{
  constexpr auto __rank = decltype(::cute::rank(__layout))::value;
  static_assert(__rank <= _MaxRank, "Layout rank exceeds maximum supported rank");
  __raw_tensor<_Tp, _MaxRank> __result{__data, 0, {}, {}};
  int __r = 0;
  ::cuda::static_for<__rank>([&] __host__ __device__(auto __i) {
    const auto __shape = ::cute::shape<__i>(__layout);
    if (!_RemoveExtent1 || __shape != 1)
    {
      __result.__shapes[__r]  = __shape;
      __result.__strides[__r] = ::cute::stride<__i>(__layout);
      ++__r;
    }
  });
  __result.__rank = __r;
  return __result;
}

//! @brief Remove degenerate (shape == 1) modes from a raw tensor in-place.
//!
//! Compacts the shapes and strides arrays, shifting non-degenerate modes
//! to the front and decreasing the rank accordingly.
//!
//! @param[in,out] __tensor Raw tensor to compact
template <class _Tp, ::cuda::std::size_t _MaxRank>
_CCCL_HOST void __remove_extent1(__raw_tensor<_Tp, _MaxRank>& __tensor) noexcept
{
  ::cuda::std::size_t __out = 0;
  for (::cuda::std::size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    if (__tensor.__shapes[__i] != 1)
    {
      __tensor.__shapes[__out]  = __tensor.__shapes[__i];
      __tensor.__strides[__out] = __tensor.__strides[__i];
      ++__out;
    }
  }
  __tensor.__rank = __out;
}

//! @brief Sort modes of a src/dst tensor pair by dst's ascending absolute stride.
//!
//! @pre Both tensors must have the same shapes (mode-by-mode).
//! The same permutation is applied to both, so shapes remain identical after sorting.
//!
//! @param[in,out] __src Source tensor (shapes and strides reordered)
//! @param[in,out] __dst Destination tensor (shapes and strides reordered by same permutation)
template <class _TpSrc, class _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST void
__sort_by_stride_paired(__raw_tensor<_TpSrc, _MaxRank>& __src, __raw_tensor<_TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__shapes == __dst.__shapes, "Source and destination shapes must be identical");
  ::cuda::std::array<::cuda::std::size_t, _MaxRank> __orders{};
  for (::cuda::std::size_t __i = 0; __i < __src.__rank; ++__i)
  {
    __orders[__i] = __i;
  }
  ::cuda::std::stable_sort(__orders.begin(), __orders.begin() + __dst.__rank, [&](auto __a, auto __b) {
    return ::cuda::std::abs(__dst.__strides[__a]) < ::cuda::std::abs(__dst.__strides[__b]);
  });
  const auto __shapes_copy      = __src.__shapes;
  const auto __src_strides_copy = __src.__strides;
  const auto __dst_strides_copy = __dst.__strides;
  for (int __i = 0; __i < __dst.__rank; ++__i)
  {
    __src.__shapes[__i]  = __shapes_copy[__orders[__i]];
    __src.__strides[__i] = __src_strides_copy[__orders[__i]];
    __dst.__strides[__i] = __dst_strides_copy[__orders[__i]];
  }
  __dst.__shapes = __src.__shapes;
}

//! @brief Coalesce adjacent contiguous modes in a src/dst tensor pair.
//!
//! Two adjacent modes (i, i+1) are merged when shape[i] * stride[i] == stride[i+1]
//! holds for BOTH tensors. After merging, consumed modes are compacted out and
//! both tensors' ranks are reduced accordingly.
//! lose information
//! (2,4,16):(80,20,3) -->
//!   (8,16):   (20,3)
//!
//! @pre Both tensors must be sorted by ascending absolute stride.
//! @pre Both tensors must have the same shapes (mode-by-mode).
//!
//! @param[in,out] __src Source tensor (shapes, strides, and rank updated)
//! @param[in,out] __dst Destination tensor (shapes, strides, and rank updated)
template <class _TpSrc, class _TpDst, ::cuda::std::size_t _MaxRank>
_CCCL_HOST void __coalesce_paired(__raw_tensor<_TpSrc, _MaxRank>& __src, __raw_tensor<_TpDst, _MaxRank>& __dst) noexcept
{
  _CCCL_ASSERT(__src.__rank == __dst.__rank, "Source and destination ranks must be equal");
  _CCCL_ASSERT(__src.__shapes == __dst.__shapes, "Source and destination shapes must be identical");
  const auto __rank         = __src.__rank;
  auto& __shapes            = __src.__shapes;
  ::cuda::std::size_t __out = 0;
  for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
  {
    if (__shapes[__i] == 1)
    {
      continue;
    }
    if (__out > 0)
    {
      const auto __prev_shape     = static_cast<::cuda::std::int64_t>(__shapes[__out - 1]);
      const bool __src_contiguous = (__prev_shape * __src.__strides[__out - 1] == __src.__strides[__i]);
      const bool __dst_contiguous = (__prev_shape * __dst.__strides[__out - 1] == __dst.__strides[__i]);
      if (__src_contiguous && __dst_contiguous)
      {
        __shapes[__out - 1] *= __shapes[__i];
        continue;
      }
    }
    __shapes[__out]        = __shapes[__i];
    __src.__strides[__out] = __src.__strides[__i];
    __dst.__strides[__out] = __dst.__strides[__i];
    ++__out;
  }
  __src.__rank   = __out;
  __dst.__rank   = __out;
  __dst.__shapes = __shapes;
}

template <typename _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST_API constexpr __raw_tensor<_Tp, _MaxRank>
__append(const __raw_tensor<_Tp, _MaxRank>& __tensor_in, ::cuda::std::size_t __target_rank) noexcept
{
  __raw_tensor<_Tp, _MaxRank> __result{__tensor_in.__data, __target_rank};
  for (::cuda::std::size_t __i = 0; __i < __target_rank; ++__i)
  {
    const bool __is_in      = __i < __tensor_in.__rank;
    __result.__shapes[__i]  = __is_in ? __tensor_in.__shapes[__i] : 1;
    __result.__strides[__i] = __is_in ? __tensor_in.__strides[__i] : 1;
  }
  return __result;
}

//! @brief Compute the maximum vectorization width in bytes for a raw tensor.
//!
//! For recast<VecType> to be valid:
//! - Non-contiguous strides (|stride| > 1) must be divisible by vec_elems.
//! - The stride-1 (contiguous) mode's shape must be divisible by vec_elems.
//! - The pointer must be aligned to vec_bytes.
//!
//! @param[in] __tensor Raw tensor with sorted shapes and strides
//! @return Maximum safe vectorization width in bytes
template <class _Tp, ::cuda::std::size_t _MaxRank>
[[nodiscard]] _CCCL_HOST ::cuda::std::size_t
__max_vector_size_bytes(const __raw_tensor<_Tp, _MaxRank>& __tensor) noexcept
{
  using ::cuda::std::size_t;
  const auto& __shapes  = __tensor.__shapes;
  const auto& __strides = __tensor.__strides;
  if (__strides[0] != 1)
  {
    return sizeof(_Tp);
  }
  size_t __alignment = ::cuda::experimental::ptr_alignment(__tensor.__data);
  for (size_t __i = 0; __i < __tensor.__rank; ++__i)
  {
    if (__shapes[__i] > 1 && (::cuda::std::abs(__strides[__i]) != 1))
    {
      const auto __stride_bytes = static_cast<size_t>(::cuda::std::abs(__strides[__i])) * sizeof(_Tp);
      __alignment               = ::cuda::std::gcd(__alignment, __stride_bytes);
    }
  }
  _CCCL_ASSERT(__alignment % sizeof(_Tp) == 0, "Maximum vector size is not a multiple of the element size");
  size_t __items_per_vector = __alignment / sizeof(_Tp);
  if (__shapes[0] > 1)
  {
    __items_per_vector = ::cuda::std::gcd(__items_per_vector, static_cast<size_t>(__shapes[0]));
  }
  const auto __vector_bytes           = __items_per_vector * sizeof(_Tp);
  constexpr size_t __max_vector_bytes = 16;
  return ::cuda::std::min(__vector_bytes, __max_vector_bytes);
}

//! @brief Sort modes by src's ascending absolute stride, applying the same permutation to dst strides.
//!
//! After sorting, src's stride-1 (fastest-changing) mode is in position 0.
//! The dst strides are reordered by the same permutation so that corresponding
//! modes stay paired.
//!
//! @param[in,out] __shapes      Array of mode shapes (shared, reordered in-place)
//! @param[in,out] __src_strides Array of src strides (reordered in-place; used as sort key)
//! @param[in,out] __dst_strides Array of dst strides (reordered in-place by same permutation)
//! @param[in]     __rank        Number of modes
template <::cuda::std::size_t _Np>
_CCCL_HOST void __sort_by_stride_paired(
  ::cuda::std::array<int, _Np>& __shapes,
  ::cuda::std::array<int, _Np>& __src_strides,
  ::cuda::std::array<int, _Np>& __dst_strides,
  int __rank) noexcept
{
  for (int __i = 1; __i < __rank; ++__i)
  {
    const int __shape      = __shapes[__i];
    const int __src_stride = __src_strides[__i];
    const int __dst_stride = __dst_strides[__i];
    int __j                = __i - 1;
    while (__j >= 0 && (::cuda::std::abs(__src_strides[__j]) > ::cuda::std::abs(__src_stride)))
    {
      __shapes[__j + 1]      = __shapes[__j];
      __src_strides[__j + 1] = __src_strides[__j];
      __dst_strides[__j + 1] = __dst_strides[__j];
      --__j;
    }
    __shapes[__j + 1]      = __shape;
    __src_strides[__j + 1] = __src_stride;
    __dst_strides[__j + 1] = __dst_stride;
  }
}

//! @brief Compute the contiguous extent from mode 0 upward.
//!
//! Starting from mode 0 (which must have stride == 1), greedily merges
//! adjacent modes as long as the accumulated extent equals the next stride.
//! Returns the product of merged shapes, i.e. the number of logically
//! consecutive elements that are physically contiguous.
//!
//! @param[in] __shapes Array of mode shapes (sorted by ascending stride)
//! @param[in] __stride Array of mode strides (sorted by ascending stride)
//! @param[in] __rank   Number of modes
//! @return The contiguous extent in elements, or 0 if mode 0 is not stride-1
template <::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST int __contiguous_extent(
  const ::cuda::std::array<int, _Np>& __shapes, const ::cuda::std::array<int, _Np>& __stride, int __rank) noexcept
{
  if (__rank == 0 || __stride[0] != 1)
  {
    return 0;
  }
  int __extent = __shapes[0];
  for (int __i = 1; __i < __rank; ++__i)
  {
    if (__extent == __stride[__i])
    {
      __extent *= __shapes[__i];
    }
    else
    {
      break;
    }
  }
  return __extent;
}

//! @brief Compute the maximum vectorization width in bytes for a single tensor.
//!
//! For recast<VecType> to be valid:
//! - Non-contiguous strides (|stride| > 1) must be divisible by vec_elems.
//! - The stride-1 (contiguous) mode's shape must be divisible by vec_elems.
//! - The pointer must be aligned to vec_bytes.
//!
//! @tparam _Tp           Element type (sizeof(_Tp) determines the element width)
//! @tparam _ShapeIndex   Shape array element type
//! @tparam _StrideIndex  Stride array element type
//! @tparam _Np           Array capacity
//! @param __ptr          Pointer to tensor data (used for alignment check)
//! @param __shapes       Array of mode shapes (after sorting)
//! @param __strides      Array of mode strides (after sorting)
//! @return Maximum safe vectorization width in bytes
template <class _Tp, class _ShapeIndex, class _StrideIndex, ::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST ::cuda::std::size_t __max_vector_size_bytes(
  const _Tp* __ptr,
  const ::cuda::std::array<_ShapeIndex, _Np>& __shapes,
  const ::cuda::std::array<_StrideIndex, _Np>& __strides) noexcept
{
  using ::cuda::std::size_t;
  if (__strides[0] != 1) // the tensor is not contiguous
  {
    return sizeof(_Tp);
  }
  size_t __ptr_alignment = ::cuda::experimental::ptr_alignment(__ptr);
  for (size_t __i = 0; __i < _Np; ++__i)
  {
    if (__shapes[__i] > 1 && (::cuda::std::abs(__strides[__i]) != 1))
    {
      const auto __stride_bytes = static_cast<size_t>(::cuda::std::abs(__strides[__i])) * sizeof(_Tp);
      __ptr_alignment           = ::cuda::std::gcd(__ptr_alignment, __stride_bytes);
    }
  }
  _CCCL_ASSERT(__ptr_alignment % sizeof(_Tp) == 0, "Maximum vector size is not a multiple of the element size");
  size_t __items_per_vector = __ptr_alignment / sizeof(_Tp);
  if (__shapes[0] > 1)
  {
    __items_per_vector = ::cuda::std::gcd(__items_per_vector, static_cast<size_t>(__shapes[0]));
  }
  const auto __vector_bytes           = __items_per_vector * sizeof(_Tp);
  constexpr size_t __max_vector_bytes = 16;
  return ::cuda::std::min(__vector_bytes, __max_vector_bytes);
}

//! @brief Sort modes by src's ascending absolute stride, applying the same permutation to dst strides.
//!
//! After sorting, src's stride-1 (fastest-changing) mode is in position 0.
//! The dst strides are reordered by the same permutation so that corresponding
//! modes stay paired.
//!
//! @param[in,out] __shapes      Array of mode shapes (shared, reordered in-place)
//! @param[in,out] __src_strides Array of src strides (reordered in-place; used as sort key)
//! @param[in,out] __dst_strides Array of dst strides (reordered in-place by same permutation)
//! @param[in]     __rank        Number of modes
template <::cuda::std::size_t _Np>
_CCCL_HOST void __sort_by_stride_paired(
  ::cuda::std::array<int, _Np>& __shapes,
  ::cuda::std::array<int, _Np>& __src_strides,
  ::cuda::std::array<int, _Np>& __dst_strides,
  int __rank) noexcept
{
  for (int __i = 1; __i < __rank; ++__i)
  {
    const int __shape      = __shapes[__i];
    const int __src_stride = __src_strides[__i];
    const int __dst_stride = __dst_strides[__i];
    int __j                = __i - 1;
    while (__j >= 0 && (::cuda::std::abs(__src_strides[__j]) > ::cuda::std::abs(__src_stride)))
    {
      __shapes[__j + 1]      = __shapes[__j];
      __src_strides[__j + 1] = __src_strides[__j];
      __dst_strides[__j + 1] = __dst_strides[__j];
      --__j;
    }
    __shapes[__j + 1]      = __shape;
    __src_strides[__j + 1] = __src_stride;
    __dst_strides[__j + 1] = __dst_stride;
  }
}

//! @brief Compute the contiguous extent from mode 0 upward.
//!
//! Starting from mode 0 (which must have stride == 1), greedily merges
//! adjacent modes as long as the accumulated extent equals the next stride.
//! Returns the product of merged shapes, i.e. the number of logically
//! consecutive elements that are physically contiguous.
//!
//! @param[in] __shapes Array of mode shapes (sorted by ascending stride)
//! @param[in] __stride Array of mode strides (sorted by ascending stride)
//! @param[in] __rank   Number of modes
//! @return The contiguous extent in elements, or 0 if mode 0 is not stride-1
template <::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST int __contiguous_extent(
  const ::cuda::std::array<int, _Np>& __shapes, const ::cuda::std::array<int, _Np>& __stride, int __rank) noexcept
{
  if (__rank == 0 || __stride[0] != 1)
  {
    return 0;
  }
  int __extent = __shapes[0];
  for (int __i = 1; __i < __rank; ++__i)
  {
    if (__extent == __stride[__i])
    {
      __extent *= __shapes[__i];
    }
    else
    {
      break;
    }
  }
  return __extent;
}

//! @brief Extract shapes and strides from a CuTe layout into plain arrays.
//!
//! For dynamic layouts, this extracts runtime values. For static layouts,
//! it converts compile-time values to runtime integers.
//!
//! @tparam _Layout      CuTe layout type
//! @tparam _ShapeIndex  Shape array element type
//! @tparam _StrideIndex Stride array element type
//! @tparam _Np          Array capacity
//! @param[in]  __layout    The CuTe layout to extract from
//! @param[out] __out_shape  Output array for shapes (must have at least rank elements)
//! @param[out] __out_stride Output array for strides (must have at least rank elements)
template <class _Layout, class _ShapeIndex, class _StrideIndex, ::cuda::std::size_t _Np>
_CCCL_HOST void __extract_layout(const _Layout& __layout,
                                 ::cuda::std::array<_ShapeIndex, _Np>& __out_shape,
                                 ::cuda::std::array<_StrideIndex, _Np>& __out_stride) noexcept
{
  constexpr auto __rank = decltype(::cute::rank(__layout))::value;
  ::cuda::static_for<__rank>([&](auto __i) {
    __out_shape[__i]  = ::cute::shape<__i>(__layout);
    __out_stride[__i] = ::cute::stride<__i>(__layout);
  });
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDAX__COPY_BYTES_LAYOUT_UTILS
