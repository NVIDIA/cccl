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
// documentation and/or other materials provided with the distribution.
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
#ifndef _LIBCUDACXX___LINALG_CONJUGATE_IF_NEEDED_HPP
#define _LIBCUDACXX___LINALG_CONJUGATE_IF_NEEDED_HPP

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/complex>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

#if _CCCL_STD_VER >= 2017

namespace linalg
{

namespace __detail
{

template <class _Type, class = void>
struct __has_conj : _CUDA_VSTD::false_type
{};

// If I can find unqualified conj via overload resolution, then assume that conj(__t) returns the conjugate of __t.
template <class _Type>
struct __has_conj<_Type, _CUDA_VSTD::void_t<decltype(_CUDA_VSTD::conj(_CUDA_VSTD::declval<_Type>()))>>
    : _CUDA_VSTD::true_type
{};

template <class _Type>
_LIBCUDACXX_HIDE_FROM_ABI _Type __conj_if_needed_impl(const _Type& __t, _CUDA_VSTD::false_type)
{
  return __t;
}

template <class _Type>
_LIBCUDACXX_HIDE_FROM_ABI auto __conj_if_needed_impl(const _Type& __t, _CUDA_VSTD::true_type)
{
  if constexpr (_CUDA_VSTD::is_arithmetic_v<_Type>)
  {
    return __t;
  }
  else
  {
    return _CUDA_VSTD::conj(__t);
  }
}

struct __conj_if_needed
{
  template <class _Type>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(const _Type& __t) const
  {
    return __conj_if_needed_impl(__t, __has_conj<_Type>{});
  }
};

} // end namespace __detail
} // end namespace linalg

#endif // _CCCL_STD_VER >= 2017

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___LINALG_CONJUGATED_HPP
