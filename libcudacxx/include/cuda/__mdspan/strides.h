//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_STRIDES_H
#define _CUDA___MDSPAN_STRIDES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/mdspan.h>
#include <cuda/__numeric/overflow_cast.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__mdspan/extents.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_signed_integer.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/array>
#include <cuda/std/span>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// ------------------------------------------------------------------
// ------------ strides ---------------------------------------------
// ------------------------------------------------------------------

//! @brief Class to describe the strides of a multi-dimensional array layout.
//!
//! Similar to extents, but for strides. Supports both static (compile-time known)
//! and dynamic (runtime) stride values. Uses dynamic_stride as the tag for dynamic values.
//!
//! @tparam _OffsetType The signed integer type for stride values (supports negative strides)
//! @tparam _Strides... The stride values, where dynamic_stride indicates a runtime value
template <class _OffsetType, ::cuda::std::ptrdiff_t... _Strides>
class strides
    : private ::cuda::std::__mdspan_detail::
        __maybe_static_array<_OffsetType, ::cuda::std::ptrdiff_t, dynamic_stride, _Strides...>
{
public:
  using offset_type = _OffsetType;
  using size_type   = ::cuda::std::make_unsigned_t<offset_type>;
  using rank_type   = ::cuda::std::size_t;

private:
  static_assert(::cuda::std::__cccl_is_signed_integer_v<_OffsetType>,
                "strides::offset_type must be a signed integer type");

  template <class... _From>
  [[nodiscard]] _CCCL_API static constexpr bool __is_representable_as(_From... __values) noexcept
  {
    return (
      (!::cuda::overflow_cast<offset_type>(__values) || static_cast<::cuda::std::ptrdiff_t>(__values) == dynamic_stride)
      && ...);
  }

  static_assert(__is_representable_as(_Strides...), "_Strides must be representable as offset_type");

  static constexpr rank_type __rank_ = sizeof...(_Strides);
  static constexpr rank_type __rank_dynamic_ =
    ::cuda::std::__mdspan_detail::__count_dynamic_v<::cuda::std::ptrdiff_t, dynamic_stride, _Strides...>;

  template <class _OtherIndexType>
  static constexpr bool __is_convertible_to_index_type =
    ::cuda::std::is_convertible_v<const _OtherIndexType&, offset_type>
    && ::cuda::std::is_nothrow_constructible_v<offset_type, const _OtherIndexType&>;

  using _Values =
    ::cuda::std::__mdspan_detail::__maybe_static_array<_OffsetType, ::cuda::std::ptrdiff_t, dynamic_stride, _Strides...>;

public:
  [[nodiscard]] _CCCL_API static constexpr rank_type rank() noexcept
  {
    return __rank_;
  }

  [[nodiscard]] _CCCL_API static constexpr rank_type rank_dynamic() noexcept
  {
    return __rank_dynamic_;
  }

  [[nodiscard]] _CCCL_API constexpr offset_type stride(rank_type __r) const noexcept
  {
    return this->__value(__r);
  }

  [[nodiscard]] _CCCL_API static constexpr ::cuda::std::ptrdiff_t static_stride(rank_type __r) noexcept
  {
    return _Values::__static_value(__r);
  }

  _CCCL_HIDE_FROM_ABI constexpr strides() noexcept = default;

  // Construction from just dynamic or all values
  _CCCL_TEMPLATE(class... _OtherIndexTypes)
  _CCCL_REQUIRES((sizeof...(_OtherIndexTypes) == __rank_ || sizeof...(_OtherIndexTypes) == __rank_dynamic_)
                   _CCCL_AND(::cuda::std::is_convertible_v<_OtherIndexTypes, offset_type>&&...)
                     _CCCL_AND(::cuda::std::is_nothrow_constructible_v<offset_type, _OtherIndexTypes>&&...))
  _CCCL_API constexpr explicit strides(_OtherIndexTypes... __dynvals) noexcept
      : _Values(static_cast<offset_type>(__dynvals)...)
  {
    _CCCL_ASSERT(__is_representable_as(__dynvals...),
                 "strides ctor: arguments must be representable as offset_type and nonnegative");
  }

  _CCCL_TEMPLATE(class _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES((_Size == __rank_dynamic_) _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API constexpr strides(::cuda::std::span<_OtherIndexType, _Size> __strs) noexcept
      : _Values(__strs)
  {
    for ([[maybe_unused]] const auto& __value : __strs)
    {
      _CCCL_ASSERT(__is_representable_as(__value),
                   "strides ctor: arguments must be representable as offset_type and nonnegative");
    }
  }

  _CCCL_TEMPLATE(class _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES((_Size != __rank_dynamic_) _CCCL_AND(_Size == __rank_)
                   _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API explicit constexpr strides(::cuda::std::span<_OtherIndexType, _Size> __strs) noexcept
      : _Values(__strs)
  {
    for ([[maybe_unused]] const auto& __value : __strs)
    {
      _CCCL_ASSERT(__is_representable_as(__value),
                   "strides ctor: arguments must be representable as offset_type and nonnegative");
    }
  }

  _CCCL_TEMPLATE(class _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES((_Size == __rank_dynamic_) _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API constexpr strides(const ::cuda::std::array<_OtherIndexType, _Size>& __strs) noexcept
      : strides(::cuda::std::span<const _OtherIndexType, _Size>(__strs))
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, ::cuda::std::size_t _Size)
  _CCCL_REQUIRES((_Size == __rank_) _CCCL_AND(_Size != __rank_dynamic_)
                   _CCCL_AND __is_convertible_to_index_type<_OtherIndexType>)
  _CCCL_API explicit constexpr strides(const ::cuda::std::array<_OtherIndexType, _Size>& __strs) noexcept
      : strides(::cuda::std::span<const _OtherIndexType, _Size>(__strs))
  {}

private:
  // Helper to construct from other strides
  template <::cuda::std::size_t _DynCount, ::cuda::std::size_t _Idx, class _OtherStrides, class... _DynamicValues>
  [[nodiscard]] _CCCL_API constexpr _Values __construct_vals_from_strides(
    [[maybe_unused]] const _OtherStrides& __strs, _DynamicValues... __dynamic_values) noexcept
  {
    if constexpr (_Idx == __rank_)
    {
      static_assert(_DynCount == __rank_dynamic_, "Constructor of invalid strides passed to strides::strides");
      return _Values{static_cast<offset_type>(__dynamic_values)...};
    }
    else if constexpr (static_stride(_Idx) == dynamic_stride)
    {
      return __construct_vals_from_strides<_DynCount + 1, _Idx + 1>(__strs, __dynamic_values..., __strs.stride(_Idx));
    }
    else
    {
      return __construct_vals_from_strides<_DynCount, _Idx + 1>(__strs, __dynamic_values...);
    }
  }

  struct __strides_delegate_tag
  {};

  template <class _OtherIndexType, ::cuda::std::ptrdiff_t... _OtherStrides>
  _CCCL_API constexpr strides(__strides_delegate_tag, const strides<_OtherIndexType, _OtherStrides...>& __other) noexcept
      : _Values(__construct_vals_from_strides<0, 0>(__other))
  {
    if constexpr (__rank_ != 0)
    {
      for (::cuda::std::size_t __r = 0; __r != __rank_; __r++)
      {
        _CCCL_ASSERT(_Values::__static_value(__r) == dynamic_stride
                       || ::cuda::std::cmp_equal(__other.stride(__r), _Values::__static_value(__r)),
                     "strides construction: mismatch of provided arguments with static strides.");
      }
    }
  }

public:
  // Converting constructor from other strides specializations
  template <class _OtherIndexType, ::cuda::std::ptrdiff_t... _OtherStrides>
  static constexpr bool __is_explicit_conversion =
    (((_Strides != dynamic_stride) && (_OtherStrides == dynamic_stride)) || ...);

  template <::cuda::std::ptrdiff_t... _OtherStrides>
  static constexpr bool __is_matching_strides =
    ((_OtherStrides == dynamic_stride || _Strides == dynamic_stride || _OtherStrides == _Strides) && ...);

  _CCCL_TEMPLATE(class _OtherIndexType, ::cuda::std::ptrdiff_t... _OtherStrides)
  _CCCL_REQUIRES((sizeof...(_OtherStrides) == sizeof...(_Strides)) _CCCL_AND __is_matching_strides<_OtherStrides...>
                   _CCCL_AND(!__is_explicit_conversion<_OtherIndexType, _OtherStrides...>))
  _CCCL_API constexpr strides(const strides<_OtherIndexType, _OtherStrides...>& __other) noexcept
      : strides(__strides_delegate_tag{}, __other)
  {}

  _CCCL_TEMPLATE(class _OtherIndexType, ::cuda::std::ptrdiff_t... _OtherStrides)
  _CCCL_REQUIRES((sizeof...(_OtherStrides) == sizeof...(_Strides))
                   _CCCL_AND __is_matching_strides<_OtherStrides...> _CCCL_AND
                     __is_explicit_conversion<_OtherIndexType, _OtherStrides...>)
  _CCCL_API explicit constexpr strides(const strides<_OtherIndexType, _OtherStrides...>& __other) noexcept
      : strides(__strides_delegate_tag{}, __other)
  {}

  // Comparison operator
  template <class _OtherIndexType, ::cuda::std::ptrdiff_t... _OtherStrides>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator==(const strides& __lhs, const strides<_OtherIndexType, _OtherStrides...>& __rhs) noexcept
  {
    if constexpr (__rank_ != sizeof...(_OtherStrides))
    {
      return false;
    }
    else
    {
      for (rank_type __r = 0; __r != __rank_; __r++)
      {
        if (::cuda::std::cmp_not_equal(__lhs.stride(__r), __rhs.stride(__r)))
        {
          return false;
        }
      }
      return true;
    }
  }

#if _CCCL_STD_VER <= 2017
  template <class _OtherIndexType, ::cuda::std::ptrdiff_t... _OtherStrides>
  [[nodiscard]] _CCCL_API friend constexpr bool
  operator!=(const strides& __lhs, const strides<_OtherIndexType, _OtherStrides...>& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif // _CCCL_STD_VER <= 2017
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___MDSPAN_STRIDES_H
