//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_GET_MEMORY_RESOURCE_CUH
#define _CUDAX__MEMORY_RESOURCE_GET_MEMORY_RESOURCE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// If the memory resource header was included without the experimental flag,
// tell the user to define the experimental flag.
#if defined(_CUDA_MEMORY_RESOURCE) && !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  error "To use the experimental memory resource, define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE"
#endif

// cuda::mr is unavable on MSVC 2017
#if defined(_CCCL_COMPILER_MSVC_2017)
#  error "The any_resource header is not supported on MSVC 2017"
#endif

#if !defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)
#  define LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE
#endif

#include <cuda/__memory_resource/properties.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__memory_resource/any_resource.cuh>

namespace cuda::experimental
{

template <class _Tp>
_CCCL_CONCEPT_FRAGMENT(__has_member_get_resource_,
                       requires(const _Tp& __t)(
                         requires(_CUDA_VMR::async_resource<cuda::std::remove_cvref_t<decltype(__t.get_resource())>>)));

template <class _Tp>
_CCCL_CONCEPT __has_member_get_resource = _CCCL_FRAGMENT(__has_member_get_resource_, _Tp);

struct get_memory_resource_t;

template <class _Env>
_CCCL_CONCEPT_FRAGMENT(__has_query_get_memory_resource_,
                       requires(const _Env& __env, const get_memory_resource_t& __cpo)(
                         requires(!__has_member_get_resource<_Env>),
                         requires(_CUDA_VMR::async_resource<cuda::std::remove_cvref_t<decltype(__env.query(__cpo))>>)));

template <class _Env>
_CCCL_CONCEPT __has_query_get_memory_resource = _CCCL_FRAGMENT(__has_query_get_memory_resource_, _Env);

//! @brief `get_memory_resource_t` is a customization point object that queries a type `T` for an associated memory
//! resource
struct get_memory_resource_t
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_resource<_Tp>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr decltype(auto) operator()(const _Tp& __t) const
    noexcept(noexcept(__t.get_resource()))
  {
    return __t.get_resource();
  } // namespace cuda::experimental

  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__has_query_get_memory_resource<_Env>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr decltype(auto) operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)), "");
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT auto get_memory_resource = get_memory_resource_t{};

} // namespace cuda::experimental

#endif //_CUDAX__MEMORY_RESOURCE_GET_MEMORY_RESOURCE_CUH