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

#include <cuda/__fwd/mdspan.h>
#include <cuda/__numeric/add_overflow.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/concepts.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__mdspan/layout_right.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief Computes the effective offset of a strided mapping by calling mapping(0, 0, ...).
//! Standard layouts (layout_left, layout_right, layout_stride) don't have an explicit offset() member,
//! but layout_stride_relaxed does. This helper enables comparing offsets in operator==.
template <class _StridedMapping, ::cuda::std::size_t... _Pos>
[[nodiscard]] _CCCL_API constexpr auto
__layout_stride_relaxed_compute_offset(const _StridedMapping& __mapping, ::cuda::std::index_sequence<_Pos...>) noexcept
{
  using __index_type = typename _StridedMapping::index_type;
  return static_cast<__index_type>(__mapping((static_cast<void>(_Pos), __index_type{0})...));
}

//! @brief Overload that handles rank-0 and zero-extent cases before delegating to the index_sequence version.
template <class _StridedMapping>
[[nodiscard]] _CCCL_API constexpr auto __layout_stride_relaxed_compute_offset(const _StridedMapping& __mapping) noexcept
{
  using _Extents        = typename _StridedMapping::extents_type;
  using _RankType       = typename _StridedMapping::rank_type;
  constexpr auto __rank = _Extents::rank();
  if constexpr (__rank > 0) // avoid pointless comparison of unsigned integer with zero warning
  {
    // Check if any extent is zero - can't call mapping(0,...) in that case
    for (_RankType __r = 0; __r < __rank; ++__r)
    {
      if (__mapping.extents().extent(__r) == 0)
      {
        return typename _StridedMapping::index_type{0};
      }
    }
  }
  return ::cuda::__layout_stride_relaxed_compute_offset(__mapping, ::cuda::std::make_index_sequence<__rank>{});
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

template <class _Extents, class _Stride>
class layout_stride_relaxed::mapping
{
public:
  static_assert(::cuda::std::__is_cuda_std_extents_v<_Extents>,
                "layout_stride_relaxed::mapping template argument must be a specialization of extents.");

  static_assert(_Extents::rank() == _Stride::rank(),
                "layout_stride_relaxed::mapping: extents and strides must have the same rank");

  using extents_type = _Extents;
  using strides_type = _Stride;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using offset_type  = typename strides_type::offset_type;
  using layout_type  = layout_stride_relaxed;

private:
  static constexpr rank_type __rank_    = extents_type::rank();
  static constexpr auto __rank_sequence = ::cuda::std::make_index_sequence<extents_type::rank()>();

  extents_type __extents_{};
  strides_type __strides_{};
  offset_type __offset_ = 0;

  //! @brief Helper to construct strides from another mapping using stride(r) calls
  template <class _StridedLayoutMapping>
  [[nodiscard]] _CCCL_API static constexpr strides_type __make_strides(const _StridedLayoutMapping& __other) noexcept
  {
    if constexpr (__rank_ == 0)
    {
      return strides_type{};
    }
    else
    {
      ::cuda::std::array<offset_type, __rank_> __init_strides{};
      for (rank_type __d = 0; __d < __rank_; ++__d)
      {
        _CCCL_ASSERT(!::cuda::overflow_cast<offset_type>(__other.stride(__d)),
                     "layout_stride_relaxed::mapping: stride is out of range");
        __init_strides[__d] = static_cast<offset_type>(__other.stride(__d));
      }
      return strides_type(__init_strides);
    }
  }

public:
  //! @brief Default constructor delegates to converting constructor from layout_right
  _CCCL_API constexpr mapping() noexcept
      : mapping(::cuda::std::layout_right::mapping<extents_type>{})
  {}

  _CCCL_HIDE_FROM_ABI constexpr mapping(const mapping&) noexcept            = default;
  _CCCL_HIDE_FROM_ABI constexpr mapping& operator=(const mapping&) noexcept = default;

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(::cuda::std::is_convertible_v<const _OtherIndexType&, offset_type> _CCCL_AND(
    ::cuda::std::is_nothrow_constructible_v<offset_type, const _OtherIndexType&>))
  _CCCL_API constexpr mapping(const extents_type& __ext,
                              ::cuda::std::span<_OtherIndexType, __rank_> __span_strides,
                              offset_type __offset = 0) noexcept
      : __extents_(__ext)
      , __strides_(__span_strides)
      , __offset_(__offset)
  {
    // not catching this could lead to out-of-bounds access later when used inside mdspan
    // using ext_t = dextents<char, 2>;
    // mapping<ext_t> map(ext_t(40,40));
    // map(10, 3) == -126
    _CCCL_ASSERT((static_cast<void>(required_span_size()), true),
                 "layout_stride_relaxed::mapping extents ctor: required_span_size() is not representable");
  }

  _CCCL_TEMPLATE(class _OtherIndexType)
  _CCCL_REQUIRES(::cuda::std::is_convertible_v<const _OtherIndexType&, offset_type> _CCCL_AND(
    ::cuda::std::is_nothrow_constructible_v<offset_type, const _OtherIndexType&>))
  _CCCL_API constexpr mapping(const extents_type& __ext,
                              const ::cuda::std::array<_OtherIndexType, __rank_>& __array_strides,
                              offset_type __offset = 0) noexcept
      : mapping(__ext, ::cuda::std::span<const _OtherIndexType, __rank_>(__array_strides), __offset)
  {}

  //! @brief (non-explicit) Converting constructor from another layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping>
                   _CCCL_AND ::cuda::std::is_constructible_v<extents_type, typename _OtherMapping::extents_type>
                     _CCCL_AND ::cuda::std::is_same_v<typename _OtherMapping::layout_type, layout_stride_relaxed> //
                       _CCCL_AND(::cuda::std::is_convertible_v<typename _OtherMapping::extents_type, extents_type>))
  _CCCL_API constexpr mapping(const _OtherMapping& __other) noexcept
      : __extents_(__other.extents())
      , __strides_(__other.strides())
      , __offset_(__other.offset())
  {
    _CCCL_ASSERT((static_cast<void>(required_span_size()), true),
                 "layout_stride_relaxed::mapping converting ctor: required_span_size() is not representable");
  }

  //! @brief (explicit) Converting constructor from another layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping>
                   _CCCL_AND ::cuda::std::is_constructible_v<extents_type, typename _OtherMapping::extents_type>
                     _CCCL_AND ::cuda::std::is_same_v<typename _OtherMapping::layout_type, layout_stride_relaxed> //
                       _CCCL_AND(!::cuda::std::is_convertible_v<typename _OtherMapping::extents_type, extents_type>))
  _CCCL_API explicit constexpr mapping(const _OtherMapping& __other) noexcept
      : __extents_(__other.extents())
      , __strides_(__other.strides())
      , __offset_(__other.offset())
  {
    _CCCL_ASSERT((static_cast<void>(required_span_size()), true),
                 "layout_stride_relaxed::mapping converting ctor: required_span_size() is not representable");
  }

  //! @brief (non-explicit) Converting constructor from a strided layout mapping (NOT layout_stride_relaxed)
  _CCCL_TEMPLATE(class _StridedLayoutMapping)
  _CCCL_REQUIRES(__layout_stride_relaxed_can_convert_from_strided<_StridedLayoutMapping, extents_type> _CCCL_AND
                   __layout_stride_relaxed_converts_implicit_from_strided<_StridedLayoutMapping, extents_type>)
  _CCCL_API constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __extents_(__other.extents())
      , __strides_(__make_strides(__other))
  {
    _CCCL_ASSERT((static_cast<void>(required_span_size()), true),
                 "layout_stride_relaxed::mapping strided ctor: required_span_size() is not representable");
  }

  //! @brief Explicit converting constructor from a strided layout mapping (NOT layout_stride_relaxed)
  _CCCL_TEMPLATE(class _StridedLayoutMapping)
  _CCCL_REQUIRES(__layout_stride_relaxed_can_convert_from_strided<_StridedLayoutMapping, extents_type> _CCCL_AND(
    !__layout_stride_relaxed_converts_implicit_from_strided<_StridedLayoutMapping, extents_type>))
  _CCCL_API explicit constexpr mapping(const _StridedLayoutMapping& __other) noexcept
      : __extents_(__other.extents())
      , __strides_(__make_strides(__other))
  {
    _CCCL_ASSERT((static_cast<void>(required_span_size()), true),
                 "layout_stride_relaxed::mapping strided ctor: required_span_size() is not representable");
  }

  // [mdspan.layout.stride.obs], observers

  [[nodiscard]] _CCCL_API constexpr const extents_type& extents() const noexcept
  {
    return __extents_;
  }

  [[nodiscard]] _CCCL_API constexpr const strides_type& strides() const noexcept
  {
    return __strides_;
  }

  [[nodiscard]] _CCCL_API constexpr offset_type offset() const noexcept
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
      _CCCL_ASSERT(!::cuda::add_overflow<index_type>(__offset_, offset_type{1}),
                   "layout_stride_relaxed::mapping: required_span_size is not representable as index_type");
      return static_cast<index_type>(__offset_ + offset_type{1});
    }
    else
    {
      // The dot product of indices and strides is linear.
      // Thus, over all valid indices, the max value of the dot product is achieved at the extrema: either the min
      // index (0) if the stride is negative, or the max index (extent(r) - 1) if the stride is non-negative.
      // For non-negative stride: contribution is (extent - 1) * stride
      // For negative stride: contribution is 0 (max achieved at index 0)
      index_type __dot = index_type{1};
      for (rank_type __r = 0; __r < __rank_; ++__r)
      {
        const auto __ext = __extents_.extent(__r);
        if (__ext == index_type{0})
        {
          return index_type{0};
        }
        //_CCCL_ASSERT(!::cuda::overflow_cast<index_type>(__strides_.stride(__r)),
        //              "layout_stride_relaxed::mapping: stride is out of range");
        const auto __max_index = __strides_.stride(__r) < 0 ? index_type{0} : static_cast<index_type>(__ext - 1);
        const auto __stride    = static_cast<index_type>(__strides_.stride(__r));
        _CCCL_ASSERT(!::cuda::std::__mdspan_detail::__mul_overflow(__max_index, __stride)
                       && !::cuda::add_overflow(__max_index * __stride, __dot),
                     "layout_stride_relaxed::mapping: required_span_size is not representable as index_type");
        __dot += __max_index * __stride;
      }
      _CCCL_ASSERT(!::cuda::add_overflow<index_type>(__offset_, __dot),
                   "layout_stride_relaxed::mapping: required_span_size is not representable as index_type");
      return static_cast<index_type>(__offset_ + __dot);
    }
  }

  template <class _Index>
  [[nodiscard]] _CCCL_API constexpr bool __is_valid_index([[maybe_unused]] _Index __index) const noexcept
  {
    if constexpr (::cuda::std::__cccl_is_integer_v<_Index>)
    {
      return ::cuda::std::cmp_greater_equal(__index, index_type{0}) && !::cuda::overflow_cast<index_type>(__index);
    }
    else
    {
      return true; // cannot be verified for custom index types
    }
  }

  template <::cuda::std::size_t... _Pos, class... _Indices>
  [[nodiscard]] _CCCL_API constexpr index_type
  __compute_index(::cuda::std::index_sequence<_Pos...>, _Indices... __indices) const noexcept
  {
    _CCCL_ASSERT((__is_valid_index(__indices) && ...),
                 "layout_stride_relaxed::mapping: index is not representable as index_type");
    _CCCL_ASSERT(((static_cast<index_type>(__indices) < __extents_.extent(_Pos)) && ...),
                 "layout_stride_relaxed::mapping: index is out of bounds");
    return (static_cast<index_type>(__offset_) + ...
            + (static_cast<index_type>(__indices) * static_cast<index_type>(__strides_.stride(_Pos))));
  }

  //! @brief Maps multidimensional indices to a linear index
  _CCCL_TEMPLATE(class... _Indices)
  _CCCL_REQUIRES((sizeof...(_Indices) == __rank_) //
                 _CCCL_AND(::cuda::std::is_convertible_v<_Indices, index_type>&&...)
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

  [[nodiscard]] _CCCL_API constexpr offset_type stride(rank_type __r) const noexcept
  {
    _CCCL_ASSERT(__r < __rank_, "layout_stride_relaxed::mapping::stride(): invalid rank index");
    return __strides_.stride(__r);
  }

  // [mdspan.layout.stride.cmp], comparison

  // __rhs is also a layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping> //
                   _CCCL_AND(__rank_ == _OtherMapping::extents_type::rank())
                     _CCCL_AND(::cuda::std::is_same_v<layout_type, typename _OtherMapping::layout_type>))
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return __lhs.extents() == __rhs.extents() && ::cuda::std::cmp_equal(__lhs.offset(), __rhs.offset())
        && __lhs.strides() == __rhs.strides();
  }

  // __rhs is NOT a layout_stride_relaxed::mapping
  _CCCL_TEMPLATE(class _OtherMapping)
  _CCCL_REQUIRES(::cuda::std::__mdspan_detail::__layout_mapping_alike<_OtherMapping> //
                   _CCCL_AND(__rank_ == _OtherMapping::extents_type::rank())
                     _CCCL_AND(_OtherMapping::is_always_strided()))
  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const mapping& __lhs, const _OtherMapping& __rhs) noexcept
  {
    return __lhs.extents() == __rhs.extents()
        && ::cuda::std::cmp_equal(__lhs.offset(), ::cuda::__layout_stride_relaxed_compute_offset(__rhs))
        && __lhs.strides() == __make_strides(__rhs);
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
