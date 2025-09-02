//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_TYPE_TRAITS
#define __CUDAX_EXECUTION_TYPE_TRAITS

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/type_traits.cuh> // IWYU pragma: export

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Ret, class... _Args>
using __fn_t _CCCL_NODEBUG_ALIAS = _Ret(_Args...);

template <class _Ret, class... _Args>
using __fn_ptr_t _CCCL_NODEBUG_ALIAS = _Ret (*)(_Args...);

template <class _Ty>
using __cref_t _CCCL_NODEBUG_ALIAS = _Ty const&;

using __cp _CCCL_NODEBUG_ALIAS    = ::cuda::std::__type_self;
using __cpclr _CCCL_NODEBUG_ALIAS = ::cuda::std::__type_quote1<__cref_t>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_TYPE_TRAITS
