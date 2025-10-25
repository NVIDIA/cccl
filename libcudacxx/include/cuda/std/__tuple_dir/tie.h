//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TIE_H
#define _CUDA_STD___TUPLE_TIE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/unwrap_ref.h>
#include <cuda/std/__tuple_dir/tuple.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__utility/forward.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple<_Tp&...> tie(_Tp&... __t) noexcept
{
  return tuple<_Tp&...>(__t...);
}

template <class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple<unwrap_ref_decay_t<_Tp>...>
make_tuple(_Tp&&... __t) noexcept(is_nothrow_constructible_v<tuple<unwrap_ref_decay_t<_Tp>...>, _Tp...>)
{
  return tuple<unwrap_ref_decay_t<_Tp>...>(::cuda::std::forward<_Tp>(__t)...);
}

template <class... _Tp>
[[nodiscard]] _CCCL_API constexpr tuple<_Tp&&...> forward_as_tuple(_Tp&&... __t) noexcept
{
  return tuple<_Tp&&...>(::cuda::std::forward<_Tp>(__t)...);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TIE_H
