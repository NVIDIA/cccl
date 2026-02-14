//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FUNCTIONAL_PROCLAIM_RETURN_TYPE_H
#define _CUDA___FUNCTIONAL_PROCLAIM_RETURN_TYPE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA
namespace __detail
{
template <class _Ret, class _DecayFn>
class __return_type_wrapper
{
private:
  _DecayFn __fn_;

public:
  __return_type_wrapper() = delete;

  template <class _Fn, class = ::cuda::std::enable_if_t<::cuda::std::is_same_v<::cuda::std::decay_t<_Fn>, _DecayFn>>>
  _CCCL_API constexpr explicit __return_type_wrapper(_Fn&& __fn) noexcept
      : __fn_(::cuda::std::forward<_Fn>(__fn))
  {}

  template <class... _As>
  _CCCL_API constexpr _Ret operator()(_As&&... __as) & noexcept
  {
#if !_CCCL_CUDA_COMPILER(NVCC) || defined(__CUDA_ARCH__)
    static_assert(::cuda::std::is_same_v<_Ret, ::cuda::std::invoke_result_t<_DecayFn&, _As...>>,
                  "Return type shall match the proclaimed one exactly");
#endif // !_CCCL_CUDA_COMPILER(NVCC) || __CUDA_ARCH__

    return ::cuda::std::__invoke(__fn_, ::cuda::std::forward<_As>(__as)...);
  }

  template <class... _As>
  _CCCL_API constexpr _Ret operator()(_As&&... __as) && noexcept
  {
#if !_CCCL_CUDA_COMPILER(NVCC) || defined(__CUDA_ARCH__)
    static_assert(::cuda::std::is_same_v<_Ret, ::cuda::std::invoke_result_t<_DecayFn, _As...>>,
                  "Return type shall match the proclaimed one exactly");
#endif // !_CCCL_CUDA_COMPILER(NVCC) || __CUDA_ARCH__

    return ::cuda::std::__invoke(::cuda::std::move(__fn_), ::cuda::std::forward<_As>(__as)...);
  }

  template <class... _As>
  _CCCL_API constexpr _Ret operator()(_As&&... __as) const& noexcept
  {
#if !_CCCL_CUDA_COMPILER(NVCC) || defined(__CUDA_ARCH__)
    static_assert(::cuda::std::is_same_v<_Ret, ::cuda::std::invoke_result_t<const _DecayFn&, _As...>>,
                  "Return type shall match the proclaimed one exactly");
#endif // !_CCCL_CUDA_COMPILER(NVCC) || __CUDA_ARCH__

    return ::cuda::std::__invoke(__fn_, ::cuda::std::forward<_As>(__as)...);
  }

  template <class... _As>
  _CCCL_API constexpr _Ret operator()(_As&&... __as) const&& noexcept
  {
#if !_CCCL_CUDA_COMPILER(NVCC) || defined(__CUDA_ARCH__)
    static_assert(::cuda::std::is_same_v<_Ret, ::cuda::std::invoke_result_t<const _DecayFn, _As...>>,
                  "Return type shall match the proclaimed one exactly");
#endif // !_CCCL_CUDA_COMPILER(NVCC) || __CUDA_ARCH__

    return ::cuda::std::__invoke(::cuda::std::move(__fn_), ::cuda::std::forward<_As>(__as)...);
  }
};
} // namespace __detail

template <class _Ret, class _Fn>
_CCCL_API inline __detail::__return_type_wrapper<_Ret, ::cuda::std::decay_t<_Fn>>
proclaim_return_type(_Fn&& __fn) noexcept
{
  return __detail::__return_type_wrapper<_Ret, ::cuda::std::decay_t<_Fn>>(::cuda::std::forward<_Fn>(__fn));
}
_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FUNCTIONAL_PROCLAIM_RETURN_TYPE_H
