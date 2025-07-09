//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_BASIC_ANY_OVERRIDES_H
#define __CUDAX_DETAIL_BASIC_ANY_OVERRIDES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/is_const.h>

#include <cuda/experimental/__utility/basic_any/basic_any_fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _Interface, class _Tp = __remove_ireference_t<_Interface>>
using __overrides_for _CCCL_NODEBUG_ALIAS = typename _Interface::template overrides<_Tp>;

//!
//! overrides_for
//!
template <class _InterfaceOrModel, auto... _VirtualFnsOrOverrides>
struct overrides_for
{
  static_assert(!_CUDA_VSTD::is_const_v<_InterfaceOrModel>, "expected a class type");
  using __vtable _CCCL_NODEBUG_ALIAS = __basic_vtable<_InterfaceOrModel, _VirtualFnsOrOverrides...>;
  using __vptr_t _CCCL_NODEBUG_ALIAS = __vtable const*;
};

template <class... _Interfaces>
struct overrides_for<__iset<_Interfaces...>>
{
  using __vtable _CCCL_NODEBUG_ALIAS = __basic_vtable<__iset<_Interfaces...>>;
  using __vptr_t _CCCL_NODEBUG_ALIAS = __iset_vptr<_Interfaces...>;
};

template <>
struct overrides_for<iunknown>
{
  using __vtable _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__ignore_t; // no vtable, rtti is added explicitly in __vtable_tuple
  using __vptr_t _CCCL_NODEBUG_ALIAS = __rtti const*;
};

} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDAX_DETAIL_BASIC_ANY_OVERRIDES_H
