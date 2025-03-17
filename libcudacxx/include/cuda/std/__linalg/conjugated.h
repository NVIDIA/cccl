//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
// ************************************************************************
//@HEADER

#ifndef _LIBCUDACXX___LINALG_CONJUGATED_HPP
#define _LIBCUDACXX___LINALG_CONJUGATED_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__linalg/conj_if_needed.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/mdspan>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace linalg
{

template <class _NestedAccessor>
class conjugated_accessor
{
private:
  using __nested_element_type = typename _NestedAccessor::element_type;
  using __nc_result_type      = decltype(conj_if_needed(_CUDA_VSTD::declval<__nested_element_type>()));

public:
  using element_type     = add_const_t<__nc_result_type>;
  using reference        = remove_const_t<element_type>;
  using data_handle_type = typename _NestedAccessor::data_handle_type;
  using offset_policy    = conjugated_accessor<typename _NestedAccessor::offset_policy>;

  _CCCL_HIDE_FROM_ABI constexpr conjugated_accessor() = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr conjugated_accessor(const _NestedAccessor& __acc)
      : __nested_accessor_(__acc)
  {}

  _CCCL_TEMPLATE(class _OtherNestedAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _NestedAccessor, const _OtherNestedAccessor&)
                   _CCCL_AND _CCCL_TRAIT(is_convertible, _OtherNestedAccessor, _NestedAccessor))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr conjugated_accessor(const conjugated_accessor<_OtherNestedAccessor>& __other)
      : __nested_accessor_(__other.nested_accessor())
  {}

  _CCCL_TEMPLATE(class _OtherNestedAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _NestedAccessor, const _OtherNestedAccessor&)
                   _CCCL_AND(!_CCCL_TRAIT(is_convertible, _OtherNestedAccessor, _NestedAccessor)))
  _LIBCUDACXX_HIDE_FROM_ABI explicit constexpr conjugated_accessor(
    const conjugated_accessor<_OtherNestedAccessor>& __other)
      : __nested_accessor_(__other.nested_accessor())
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return conj_if_needed(__nested_element_type(__nested_accessor_.access(__p, __i)));
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr typename offset_policy::data_handle_type
  offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __nested_accessor_.offset(__p, __i);
  }

  _CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr const _NestedAccessor& nested_accessor() const noexcept
  {
    return __nested_accessor_;
  }

private:
  _NestedAccessor __nested_accessor_;
};

template <class _ElementType, class _Extents, class _Layout, class _Accessor>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
conjugated(mdspan<_ElementType, _Extents, _Layout, _Accessor> __a)
{
  using __value_type = typename decltype(__a)::value_type;
  // Current status of [linalg] only optimizes if _Accessor is conjugated_accessor<_Accessor> for some _Accessor.
  // There's a separate specialization for that case below.

  // P3050 optimizes conjugated's accessor type for when we know that it can't be complex: arithmetic types,
  // and types for which `conj` is not ADL-findable.
  if constexpr (is_arithmetic_v<__value_type> || !__conj_if_needed::_HasConj<__value_type>)
  {
    return mdspan<_ElementType, _Extents, _Layout, _Accessor>(__a.data_handle(), __a.mapping(), __a.accessor());
  }
  else
  {
    using __return_element_type  = typename conjugated_accessor<_Accessor>::element_type;
    using __return_accessor_type = conjugated_accessor<_Accessor>;
    return mdspan<__return_element_type, _Extents, _Layout, __return_accessor_type>{
      __a.data_handle(), __a.mapping(), __return_accessor_type(__a.accessor())};
  }
  _CCCL_UNREACHABLE();
}

// Conjugation is self-annihilating
template <class _ElementType, class _Extents, class _Layout, class _NestedAccessor>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI constexpr auto
conjugated(mdspan<_ElementType, _Extents, _Layout, conjugated_accessor<_NestedAccessor>> __a)
{
  using __return_element_type  = typename _NestedAccessor::element_type;
  using __return_accessor_type = _NestedAccessor;
  return mdspan<__return_element_type, _Extents, _Layout, __return_accessor_type>(
    __a.data_handle(), __a.mapping(), __a.accessor().nested_accessor());
}

} // end namespace linalg

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LINALG_CONJUGATED_HPP
