//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_RESTRICT_ACCESSOR
#define _CUDA___MDSPAN_RESTRICT_ACCESSOR

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _Accessor>
class __restrict_accessor;

template <typename _Accessor>
using restrict_accessor = __restrict_accessor<_Accessor>;

/***********************************************************************************************************************
 * Accessor Traits
 **********************************************************************************************************************/

template <typename>
inline constexpr bool is_restrict_accessor_v = false;

template <typename _Accessor>
inline constexpr bool is_restrict_accessor_v<__restrict_accessor<_Accessor>> = true;

/***********************************************************************************************************************
 * Restrict Accessor
 **********************************************************************************************************************/

template <typename _Accessor>
class __restrict_accessor : public _Accessor
{
  static_assert(_CUDA_VSTD::is_pointer_v<typename _Accessor::data_handle_type>, "Accessor must be pointer based");

  using __data_handle_type = typename _Accessor::data_handle_type;
  using __element_type     = _CUDA_VSTD::remove_pointer_t<__data_handle_type>;

  static constexpr bool __is_access_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().access(_CUDA_VSTD::declval<__data_handle_type>(), 0));

  static constexpr bool __is_offset_noexcept =
    noexcept(_CUDA_VSTD::declval<_Accessor>().offset(_CUDA_VSTD::declval<__data_handle_type>(), 0));

public:
  using offset_policy    = __restrict_accessor<typename _Accessor::offset_policy>;
  using data_handle_type = __element_type* _CCCL_RESTRICT;
  using reference        = typename _Accessor::reference;
  using element_type     = typename _Accessor::element_type;

  _CCCL_TEMPLATE(class _Accessor2 = _Accessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_default_constructible, _Accessor2))
  _LIBCUDACXX_HIDE_FROM_ABI __restrict_accessor() noexcept(_CUDA_VSTD::is_nothrow_default_constructible_v<_Accessor2>)
      : _Accessor{}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __restrict_accessor(const _Accessor& __acc) noexcept(
    _CUDA_VSTD::is_nothrow_copy_constructible_v<_Accessor>)
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr __restrict_accessor(const __restrict_accessor<_OtherAccessor>& __acc) noexcept(
    noexcept(_Accessor{_CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _CCCL_TEMPLATE(typename _OtherAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_constructible, _OtherAccessor)
                   _CCCL_AND(!_CUDA_VSTD::is_convertible_v<_OtherAccessor, _Accessor>))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr explicit __restrict_accessor(
    const __restrict_accessor<_OtherAccessor>& __acc) noexcept(noexcept(_Accessor{
    _CUDA_VSTD::declval<_OtherAccessor>()}))
      : _Accessor{__acc}
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(__element_type* _CCCL_RESTRICT __p, size_t __i) const
    noexcept(__is_access_noexcept)
  {
    return _Accessor::access(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr data_handle_type offset(__element_type* _CCCL_RESTRICT __p, size_t __i) const
    noexcept(__is_offset_noexcept)
  {
    return _Accessor::offset(__p, __i);
  }
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___MDSPAN_RESTRICT_ACCESSOR
