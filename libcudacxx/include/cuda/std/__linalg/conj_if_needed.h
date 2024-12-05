// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
//===---------------------------------------------------------------------===//

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

#include <cuda/std/version>

#if defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017

#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__type_traits/integral_constant.h>
#  include <cuda/std/__type_traits/is_arithmetic.h>
#  include <cuda/std/__type_traits/remove_const.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/complex>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

namespace linalg
{

_LIBCUDACXX_BEGIN_NAMESPACE_CPO(__conj_if_needed)

template <class _Type>
_CCCL_CONCEPT _HasConj = _CCCL_REQUIRES_EXPR((_Type), _Type __a)(static_cast<void>(_CUDA_VSTD::conj(__a)));

struct __conj_if_needed
{
  template <class _Type>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr auto operator()(const _Type& __t) const
  {
    if constexpr (is_arithmetic_v<_Type> || !_HasConj<_Type>)
    {
      return __t;
    }
    else
    {
      return _CUDA_VSTD::conj(__t);
    }
    _CCCL_UNREACHABLE();
  }
};

_LIBCUDACXX_END_NAMESPACE_CPO

inline namespace __cpo
{
_CCCL_GLOBAL_CONSTANT auto conj_if_needed = __conj_if_needed::__conj_if_needed{};

} // namespace __cpo
} // end namespace linalg

_LIBCUDACXX_END_NAMESPACE_STD

#endif // defined(__cccl_lib_mdspan) && _CCCL_STD_VER >= 2017
#endif // _LIBCUDACXX___LINALG_CONJUGATED_HPP
