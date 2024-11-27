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

#include <cuda/std/version>

#if defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017

#  include <cuda/std/__linalg/conj_if_needed.h>
#  include <cuda/std/__type_traits/add_const.h>
#  include <cuda/std/__type_traits/is_arithmetic.h>
#  include <cuda/std/__type_traits/remove_const.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/mdspan>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace linalg
{

template <class _NestedAccessor>
class conjugated_accessor
{
private:
  using __nested_element_type = typename _NestedAccessor::element_type;
  using __nc_result_type      = decltype(__detail::__conj_if_needed{}(_CUDA_VSTD::declval<__nested_element_type>()));

public:
  using element_type     = _CUDA_VSTD::add_const_t<__nc_result_type>;
  using reference        = _CUDA_VSTD::remove_const_t<element_type>;
  using data_handle_type = typename _NestedAccessor::data_handle_type;
  using offset_policy    = conjugated_accessor<typename _NestedAccessor::offset_policy>;

  _CCCL_HIDE_FROM_ABI constexpr conjugated_accessor() = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr conjugated_accessor(const _NestedAccessor& __acc)
      : __nested_accessor_(__acc)
  {}

  _CCCL_TEMPLATE(class _OtherNestedAccessor)
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _NestedAccessor, const _OtherNestedAccessor&))
  __MDSPAN_CONDITIONAL_EXPLICIT(!_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _OtherNestedAccessor, _NestedAccessor))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr conjugated_accessor(const conjugated_accessor<_OtherNestedAccessor>& __other)
      : __nested_accessor_(__other.nested_accessor())
  {}

  _LIBCUDACXX_HIDE_FROM_ABI constexpr reference access(data_handle_type __p, size_t __i) const noexcept
  {
    return __detail::__conj_if_needed{}(__nested_element_type(__nested_accessor_.access(__p, __i)));
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr typename offset_policy::data_handle_type
  offset(data_handle_type __p, size_t __i) const noexcept
  {
    return __nested_accessor_.offset(__p, __i);
  }

  [[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI const _NestedAccessor& nested_accessor() const noexcept
  {
    return __nested_accessor_;
  }

private:
  _NestedAccessor __nested_accessor_;
};

template <class _ElementType, class _Extents, class _Layout, class _Accessor>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI auto
conjugated(_CUDA_VSTD::mdspan<_ElementType, _Extents, _Layout, _Accessor> __a)
{
  using __value_type = typename decltype(__a)::value_type;
  // Current status of [linalg] only optimizes if _Accessor is conjugated_accessor<_Accessor> for some _Accessor.
  // There's __a separate specialization for that case below.

  // P3050 optimizes conjugated's accessor type for when we know that it can't be complex: arithmetic types,
  // and types for which `conj` is not ADL-findable.
  if constexpr (_CUDA_VSTD::is_arithmetic_v<__value_type>)
  {
    return _CUDA_VSTD::mdspan<_ElementType, _Extents, _Layout, _Accessor>(
      __a.data_handle(), __a.mapping(), __a.accessor());
  }
  else if constexpr (!__detail::__has_conj<__value_type>::value)
  {
    return _CUDA_VSTD::mdspan<_ElementType, _Extents, _Layout, _Accessor>(
      __a.data_handle(), __a.mapping(), __a.accessor());
  }
  else
  {
    using __return_element_type  = typename conjugated_accessor<_Accessor>::element_type;
    using __return_accessor_type = conjugated_accessor<_Accessor>;
    return _CUDA_VSTD::mdspan<__return_element_type, _Extents, _Layout, __return_accessor_type>(
      __a.data_handle(), __a.mapping(), __return_accessor_type(__a.accessor()));
  }
}

// Conjugation is self-annihilating
template <class _ElementType, class _Extents, class _Layout, class _NestedAccessor>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI auto
conjugated(_CUDA_VSTD::mdspan<_ElementType, _Extents, _Layout, conjugated_accessor<_NestedAccessor>> __a)
{
  using __return_element_type  = typename _NestedAccessor::element_type;
  using __return_accessor_type = _NestedAccessor;
  return _CUDA_VSTD::mdspan<__return_element_type, _Extents, _Layout, __return_accessor_type>(
    __a.data_handle(), __a.mapping(), __a.accessor().nested_accessor());
}

} // end namespace linalg

_LIBCUDACXX_END_NAMESPACE_STD

#endif // defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017
#endif // _LIBCUDACXX___LINALG_CONJUGATED_HPP
