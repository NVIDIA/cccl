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

#ifndef _LIBCUDACXX___LINALG_SCALED_HPP
#define _LIBCUDACXX___LINALG_SCALED_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/mdspan>

#include <cuda/std/__cccl/prologue.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace linalg
{

template <class _ScalingFactor, class _NestedAccessor>
class scaled_accessor
{
public:
  using element_type = add_const_t<
    decltype(_CUDA_VSTD::declval<_ScalingFactor>() * _CUDA_VSTD::declval<typename _NestedAccessor::element_type>())>;
  using reference        = remove_const_t<element_type>;
  using data_handle_type = typename _NestedAccessor::data_handle_type;
  using offset_policy    = scaled_accessor<_ScalingFactor, typename _NestedAccessor::offset_policy>;

  _CCCL_HIDE_FROM_ABI constexpr scaled_accessor() = default;

  _CCCL_TEMPLATE(class _OtherScalingFactor, class _OtherNestedAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _NestedAccessor, const _OtherNestedAccessor&)
                   _CCCL_AND _CCCL_TRAIT(is_constructible, _ScalingFactor, _OtherScalingFactor)
                     _CCCL_AND(!_CCCL_TRAIT(is_convertible, _OtherNestedAccessor, _NestedAccessor)))
  _CCCL_API explicit constexpr scaled_accessor(const scaled_accessor<_OtherScalingFactor, _OtherNestedAccessor>& __other)
      : __scaling_factor_(__other.scaling_factor())
      , __nested_accessor_(__other.nested_accessor())
  {}

  _CCCL_TEMPLATE(class _OtherScalingFactor, class _OtherNestedAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _NestedAccessor, const _OtherNestedAccessor&)
                   _CCCL_AND _CCCL_TRAIT(is_constructible, _ScalingFactor, _OtherScalingFactor)
                     _CCCL_AND _CCCL_TRAIT(is_convertible, _OtherNestedAccessor, _NestedAccessor))
  _CCCL_API constexpr scaled_accessor(const scaled_accessor<_OtherScalingFactor, _OtherNestedAccessor>& __other)
      : __scaling_factor_(__other.scaling_factor())
      , __nested_accessor_(__other.nested_accessor())
  {}

  _CCCL_API constexpr scaled_accessor(const _ScalingFactor& __s, const _NestedAccessor& __a)
      : __scaling_factor_(__s)
      , __nested_accessor_(__a)
  {}

  _CCCL_API constexpr reference access(data_handle_type __p, size_t __i) const
  {
    return __scaling_factor_ * typename _NestedAccessor::element_type(__nested_accessor_.access(__p, __i));
  }

  [[nodiscard]]
  _CCCL_API inline typename offset_policy::data_handle_type constexpr offset(data_handle_type __p, size_t __i) const
  {
    return __nested_accessor_.offset(__p, __i);
  }

  [[nodiscard]] _CCCL_API constexpr _NestedAccessor nested_accessor() const noexcept
  {
    return __nested_accessor_;
  }

  [[nodiscard]] _CCCL_API constexpr _ScalingFactor scaling_factor() const noexcept
  {
    return __scaling_factor_;
  }

private:
  _ScalingFactor __scaling_factor_;
  _NestedAccessor __nested_accessor_;
};

namespace __detail
{

template <class _ScalingFactor, class _NestedAccessor>
using __scaled_element_type = add_const_t<typename scaled_accessor<_ScalingFactor, _NestedAccessor>::element_type>;

} // namespace __detail

template <class _ScalingFactor, class _ElementType, class _Extents, class _Layout, class _Accessor>
[[nodiscard]]
_CCCL_API constexpr mdspan<__detail::__scaled_element_type<_ScalingFactor, _Accessor>,
                           _Extents,
                           _Layout,
                           scaled_accessor<_ScalingFactor, _Accessor>>
scaled(_ScalingFactor __scaling_factor, mdspan<_ElementType, _Extents, _Layout, _Accessor> __x)
{
  using __acc_type = scaled_accessor<_ScalingFactor, _Accessor>;
  return {__x.data_handle(), __x.mapping(), __acc_type{__scaling_factor, __x.accessor()}};
}

} // end namespace linalg

_LIBCUDACXX_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX___LINALG_SCALED_HPP
