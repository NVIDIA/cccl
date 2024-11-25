/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2019) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software. //
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or __other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
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

#include <cuda/std/__type_traits/add_const.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/mdspan>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER > 2011

namespace linalg
{

template <class _ScalingFactor, class _NestedAccessor>
class scaled_accessor
{
public:
  using __element_type = _CUDA_VSTD::add_const_t<
    decltype(_CUDA_VSTD::declval<_ScalingFactor>() * _CUDA_VSTD::declval<typename _NestedAccessor::__element_type>())>;
  using __reference        = _CUDA_VSTD::remove_const_t<__element_type>;
  using __data_handle_type = typename _NestedAccessor::__data_handle_type;
  using __offset_policy    = scaled_accessor<_ScalingFactor, typename _NestedAccessor::__offset_policy>;

  _CCCL_HIDE_FROM_ABI constexpr scaled_accessor() = default;

  _CCCL_TEMPLATE(class _OtherScalingFactor, class _OtherNestedAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(is_constructible, _NestedAccessor, const _OtherNestedAccessor&)
                   _CCCL_AND _CCCL_TRAIT(is_constructible, _ScalingFactor, _OtherScalingFactor))
  __MDSPAN_CONDITIONAL_EXPLICIT(_CCCL_TRAIT(is_convertible, _OtherNestedAccessor, _NestedAccessor))
  constexpr scaled_accessor(const scaled_accessor<_OtherScalingFactor, _OtherNestedAccessor>& __other)
      : __scaling_factor_(__other.scaling_factor())
      , __nested_accessor_(__other.nested_accessor())
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr scaled_accessor(const _ScalingFactor& __s, const _NestedAccessor& __a)
      : __scaling_factor_(__s)
      , __nested_accessor_(__a)
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr __reference access(__data_handle_type __p, size_t __i) const
  {
    return __scaling_factor_ * typename _NestedAccessor::__element_type(__nested_accessor_.access(__p, __i));
  }

  _LIBCUDACXX_HIDE_FROM_ABI
  typename __offset_policy::__data_handle_type constexpr offset(__data_handle_type __p, size_t __i) const
  {
    return __nested_accessor_.offset(__p, __i);
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr _NestedAccessor nested_accessor() const noexcept
  {
    return __nested_accessor_;
  }

  __MDSPAN_FORCE_INLINE_FUNCTION constexpr _ScalingFactor scaling_factor() const noexcept
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
using __scaled_element_type =
  _CUDA_VSTD::add_const_t<typename _CUDA_VSTD::linalg::scaled_accessor<_ScalingFactor, _NestedAccessor>::__element_type>;

} // namespace __detail

template <class _ScalingFactor, class _ElementType, class _Extents, class _Layout, class _Accessor>
_LIBCUDACXX_HIDE_FROM_ABI
mdspan<__detail::__scaled_element_type<_ScalingFactor, _Accessor>,
       _Extents,
       _Layout,
       _CUDA_VSTD::linalg::scaled_accessor<_ScalingFactor, _Accessor>>
scaled(_ScalingFactor __scaling_factor, mdspan<_ElementType, _Extents, _Layout, _Accessor> __x)
{
  using __acc_type = _CUDA_VSTD::linalg::scaled_accessor<_ScalingFactor, _Accessor>;
  return {__x.data_handle(), __x.mapping(), __acc_type{__scaling_factor, __x.accessor()}};
}

} // end namespace linalg

#endif // _CCCL_STD_VER > 2011

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LINALG_SCALED_HPP
