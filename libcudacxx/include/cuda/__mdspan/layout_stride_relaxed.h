//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_LAYOUT_STRIDE_RELAXED_H
#define _CUDA___MDSPAN_LAYOUT_STRIDE_RELAXED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__numeric/overflow_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__fwd/mdspan.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Layout policy with relaxed stride mapping that supports negative strides and offsets.
//!
//! Unlike `layout_stride`, this layout allows:
//! - Negative strides (for reverse iteration)
//! - Zero strides (for broadcasting)
//! - A base offset (to accommodate negative strides)
//!
//! @note This layout is NOT always unique, exhaustive, or strided in the standard sense.
struct layout_stride_relaxed
{
  template <class _Extents>
  class mapping;
};

template <class _StridedMapping, ::cuda::std::size_t... _Pos>
[[nodiscard]] _CCCL_API constexpr auto
__layout_stride_relaxed_compute_offset(const _StridedMapping& __mapping, ::cuda::std::index_sequence<_Pos...>) noexcept
{
  using __index_type = typename _StridedMapping::index_type;
  return static_cast<__index_type>(__mapping((static_cast<void>(_Pos), __index_type{0})...));
}

template <class _StridedMapping>
[[nodiscard]] _CCCL_API constexpr auto __layout_stride_relaxed_compute_offset(const _StridedMapping& __mapping) noexcept
{
  using __stride_extents = typename _StridedMapping::extents_type;
  if constexpr (__stride_extents::rank() == 0)
  {
    return static_cast<typename _StridedMapping::index_type>(__mapping());
  }
  else
  {
    // Check if any extent is zero
    using __stride_rank = typename _StridedMapping::rank_type;
    for (__stride_rank __r = 0; __r < __stride_extents::rank(); ++__r)
    {
      if (__mapping.extents().extent(__r) == 0)
      {
        return typename _StridedMapping::index_type{0};
      }
    }
    return ::cuda::__layout_stride_relaxed_compute_offset(
      __mapping, ::cuda::std::make_index_sequence<__stride_extents::rank()>());
  }
}

template <class _StridedLayoutMapping, class _Extents>
_CCCL_CONCEPT __layout_stride_relaxed_can_convert_from_strided = _CCCL_REQUIRES_EXPR((_StridedLayoutMapping, _Extents))(
  requires(::cuda::std::__mdspan_detail::__layout_mapping_alike<_StridedLayoutMapping>),
  requires(_StridedLayoutMapping::is_always_unique()),
  requires(_StridedLayoutMapping::is_always_strided()),
  requires(::cuda::std::is_constructible_v<_Extents, typename _StridedLayoutMapping::extents_type>));

template <class _StridedLayoutMapping, class _Extents>
_CCCL_CONCEPT __layout_stride_relaxed_converts_implicit_from_strided =
  _CCCL_REQUIRES_EXPR((_StridedLayoutMapping, _Extents))(
    requires(::cuda::std::is_convertible_v<typename _StridedLayoutMapping::extents_type, _Extents>),
    requires(::cuda::std::__mdspan_detail::__is_mapping_of<::cuda::std::layout_left, _StridedLayoutMapping>
             || ::cuda::std::__mdspan_detail::__is_mapping_of<::cuda::std::layout_right, _StridedLayoutMapping>
             || ::cuda::std::__mdspan_detail::__is_mapping_of<::cuda::std::layout_stride, _StridedLayoutMapping>));

/***********************************************************************************************************************
 * layout_stride_relaxed::mapping
 **********************************************************************************************************************/

template <class _Extents>
class layout_stride_relaxed::mapping
{
public:
  static_assert(::cuda::std::__is_cuda_std_extents_v<_Extents>,
                "layout_stride_relaxed::mapping template argument must be a specialization of extents.");

  using extents_type = _Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_stride_relaxed;

private:
  static constexpr rank_type __rank_    = extents_type::rank();
  static constexpr auto __rank_sequence = ::cuda::std::make_index_sequence<extents_type::rank()>();

  using __offset_type     = ::cuda::std::intptr_t;
  using __stride_array    = ::cuda::std::array<__offset_type, __rank_>;
  using __index_compute_t = decltype(index_type{} + __offset_type{});

  extents_type __extents_{};
  __offset_type __offset_ = 0;
  __stride_array __strides_{};

public:
  _CCCL_API constexpr mapping() noexcept
  {
    if constexpr (__rank_ > 0)
    {
      constexpr ::cuda::std::layout_right::mapping<extents_type> __map{};
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__map.stride(__d)),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = static_cast<__offset_type>(__map.stride(__d));
      }
    }
  }

  _CCCL_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(::cuda::std::is_convertible_v<const _OtherIndexType&, __offset_type> _CCCL_AND(
    ::cuda::std::is_nothrow_constructible_v<__offset_type, const _OtherIndexType&>))
  _CCCL_API constexpr mapping(const extents_type& __ext,
                              ::cuda::std::span<_OtherIndexType, __rank_> __span_strides,
                              __offset_type __offset = 0) noexcept
      : __extents_(__ext)
      , __offset_(__offset)
  {
    if constexpr (__rank_ > 0)
    {
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__span_strides[__d]),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = static_cast<__offset_type>(__span_strides[__d]);
      }
    }
  }

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(::cuda::std::is_convertible_v<const _OtherIndexType&, __offset_type> _CCCL_AND(
    ::cuda::std::is_nothrow_constructible_v<__offset_type, const _OtherIndexType&>))
  _CCCL_API constexpr mapping(const extents_type& __ext,
                              const ::cuda::std::array<_OtherIndexType, __rank_>& __array_strides,
                              __offset_type __offset = 0) noexcept
      : __extents_(__ext)
      , __offset_(__offset)
  {
    if constexpr (__rank_ > 0)
    {
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__array_strides[__d]),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = static_cast<__offset_type>(__array_strides[__d]);
      }
    }
  }

  //! @brief (non-explicit) Converting constructor from another layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping>
                   _CCCL_AND ::cuda::std::is_constructible_v<extents_type, typename _OtherMapping::extents_type>
                     _CCCL_AND ::cuda::std::is_same_v<typename _OtherMapping::layout_type, layout_stride_relaxed> //
                       _CCCL_AND(::cuda::std::is_convertible_v<typename _OtherMapping::extents_type, extents_type>))
  _CCCL_API constexpr mapping(const _OtherMapping& __other) noexcept
      : __extents_(__other.extents())
      , __offset_(__other.offset())
  {
    if constexpr (__rank_ > 0)
    {
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__other.stride(__d)),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = __other.stride(__d);
      }
    }
  }

  //! @brief (explicit) Converting constructor from another layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping>
                   _CCCL_AND ::cuda::std::is_constructible_v<extents_type, typename _OtherMapping::extents_type>
                     _CCCL_AND ::cuda::std::is_same_v<typename _OtherMapping::layout_type, layout_stride_relaxed> //
                       _CCCL_AND(!::cuda::std::is_convertible_v<typename _OtherMapping::extents_type, extents_type>))
  _CCCL_API explicit constexpr mapping(const _OtherMapping& __other) noexcept
      : __extents_(__other.extents())
      , __offset_(__other.offset())
  {
    if constexpr (__rank_ > 0)
    {
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__other.stride(__d)),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = __other.stride(__d);
      }
    }
  }

  //! @brief (non-explicit) Converting constructor from a strided layout mapping (not layout_stride_relaxed)
  _CCCL_TEMPLATE(class _StridedLayoutMapping)
  _CCCL_REQUIRES(__layout_stride_relaxed_can_convert_from_strided<_StridedLayoutMapping, extents_type> _CCCL_AND
                   __layout_stride_relaxed_converts_implicit_from_strided<_StridedLayoutMapping, extents_type>)
  _CCCL_API constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __extents_(__other.extents())
  {
    if constexpr (__rank_ > 0)
    {
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__other.stride(__d)),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = static_cast<__offset_type>(__other.stride(__d));
      }
    }
  }

  //! @brief Explicit converting constructor from a strided layout mapping (not layout_stride_relaxed)
  _CCCL_TEMPLATE(class _StridedLayoutMapping)
  _CCCL_REQUIRES(__layout_stride_relaxed_can_convert_from_strided<_StridedLayoutMapping, extents_type> _CCCL_AND(
    !__layout_stride_relaxed_converts_implicit_from_strided<_StridedLayoutMapping, extents_type>))
  _CCCL_API explicit constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __extents_(__other.extents())
  {
    if constexpr (__rank_ > 0)
    {
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<__offset_type>(__other.stride(__d)),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __strides_[__d] = static_cast<__offset_type>(__other.stride(__d));
      }
    }
  }

  // [mdspan.layout.stride.obs], observers

  [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
  {
    return __extents_;
  }

  [[nodiscard]] _CCCL_API constexpr __stride_array strides() const noexcept
  {
    return __strides_;
  }

  [[nodiscard]] _CCCL_API constexpr __offset_type offset() const noexcept
  {
    return __offset_;
  }

  //! @brief Returns the required span size to cover all valid indices
  [[nodiscard]] _CCCL_API constexpr index_type required_span_size() const noexcept
  {
    if constexpr (__rank_ == 0)
    {
      // For rank-0 mappings, there is exactly one valid index.
      // The required span size is the maximum mapped index + 1.
      return static_cast<index_type>(__offset_ + __offset_type{1});
    }
    else
    {
      // The dot product of indices and strides is linear.
      // Thus, over all valid indices, the max value of the dot product is achieved at the extrema: either the min
      // index (0) if the stride is negative, or the max index (extent(r) - 1) if the stride is non-negative.
      index_type __max_indices[__rank_];
      for (rank_type r = 0; r < __rank_; ++r)
      {
        const index_type __ext         = __extents_.extent(r);
        const index_type __ext_minus_1 = __ext == 0 ? index_type{0} : __ext - index_type{1};
        __max_indices[r]               = __strides_[r] < 0 ? index_type{0} : __ext_minus_1;
      }
      __index_compute_t dot = 1;
      for (rank_type r = 0; r < __rank_; ++r)
      {
        dot += static_cast<__index_compute_t>(__max_indices[r]) * static_cast<__index_compute_t>(__strides_[r]);
      }
      return static_cast<index_type>(__offset_ + dot);
    }
  }

  template <::cuda::std::size_t... _Pos, class... _Indices>
  [[nodiscard]] _CCCL_API constexpr index_type
  __compute_index(::cuda::std::index_sequence<_Pos...>, _Indices... __indices) const noexcept
  {
    return (static_cast<__index_compute_t>(__offset_) + ...
            + (static_cast<__index_compute_t>(__indices) * static_cast<__index_compute_t>(__strides_[_Pos])));
  }

  //! @brief Maps multidimensional indices to a linear index
  _CCCL_TEMPLATE(class... _Indices)
  _CCCL_REQUIRES((sizeof...(_Indices) == __rank_) _CCCL_AND(::cuda::std::is_convertible_v<_Indices, index_type>&&...)
                   _CCCL_AND(::cuda::std::is_nothrow_constructible_v<index_type, _Indices>&&...))
  [[nodiscard]] _CCCL_API constexpr index_type operator()(_Indices... __indices) const noexcept
  {
    return __compute_index(__rank_sequence, __indices...);
  }

  //! @brief Returns false - not always unique due to zero/negative strides
  [[nodiscard]] _CCCL_API static constexpr bool is_always_unique() noexcept
  {
    return false;
  }

  //! @brief Returns false - not always exhaustive due to zero/negative strides
  [[nodiscard]] _CCCL_API static constexpr bool is_always_exhaustive() noexcept
  {
    return false;
  }

  //! @brief Returns false - not always strided due to offset (to accommodate negative strides)
  [[nodiscard]] _CCCL_API static constexpr bool is_always_strided() noexcept
  {
    return false;
  }

  //! @brief Returns false - uniqueness depends on strides (conservative)
  [[nodiscard]] _CCCL_API constexpr bool is_unique() const noexcept
  {
    // Conservative: negative/zero strides make uniqueness hard to determine
    return false;
  }

  //! @brief Returns false - exhaustiveness depends on strides (conservative)
  [[nodiscard]] _CCCL_API constexpr bool is_exhaustive() const noexcept
  {
    // Conservative: negative/zero strides make exhaustiveness hard to determine
    return false;
  }

  //! @brief Returns true if offset is zero (standard strided behavior)
  [[nodiscard]] _CCCL_API constexpr bool is_strided() const noexcept
  {
    return __offset_ == 0;
  }

  [[nodiscard]] _CCCL_API constexpr index_type stride(rank_type __r) const noexcept
  {
    if constexpr (__rank_ > 0) // avoid pointless comparison of unsigned integer with zero warning
    {
      _CCCL_ASSERT(__r < __rank_, "layout_stride_relaxed::mapping::stride(): invalid rank index");
      return __strides_[__r];
    }
    else
    {
      return index_type{0};
    }
  }

  // [mdspan.layout.stride.cmp], comparison

  template <class _OtherMapping, size_t... _Pos>
  [[nodiscard]] _CCCL_API static constexpr bool
  __op_eq(const mapping& __lhs, const _OtherMapping& __rhs, ::cuda::std::index_sequence<_Pos...>) noexcept
  {
    // avoid warning when comparing signed and unsigner integers and pick the wider of two types
    using _CommonType = ::cuda::std::common_type_t<index_type, typename _OtherMapping::index_type>;
    return ((static_cast<_CommonType>(__lhs.stride(_Pos)) == static_cast<_CommonType>(__rhs.stride(_Pos))) && ...
            && true);
  }

  // __rhs is also a layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping> //
                   _CCCL_AND(__rank_ == _OtherMapping::extents_type::rank())
                     _CCCL_AND(::cuda::std::is_same_v<layout_type, typename _OtherMapping::layout_type>))
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return __lhs.extents() == __rhs.extents() && __lhs.offset() == __rhs.offset()
        && __op_eq(__lhs, __rhs, __rank_sequence);
  }

  // __rhs is NOT a layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping> //
                   _CCCL_AND(__rank_ == _OtherMapping::extents_type::rank())
                     _CCCL_AND(_OtherMapping::is_always_strided()))
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return __lhs.extents() == __rhs.extents() && __lhs.offset() == ::cuda::__layout_stride_relaxed_compute_offset(__rhs)
        && __op_eq(__lhs, __rhs, __rank_sequence);
  }

#if _CCCL_STD_VER <= 2017
  // __rhs is also a layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping> //
                   _CCCL_AND(__rank_ == _OtherMapping::extents_type::rank())
                     _CCCL_AND(::cuda::std::is_same_v<layout_type, typename _OtherMapping::layout_type>))
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }

  // __rhs is NOT a layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping> //
                   _CCCL_AND(__rank_ == _OtherMapping::extents_type::rank())
                     _CCCL_AND(_OtherMapping::is_always_strided()))
  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_LAYOUT_STRIDE_RELAXED_H
