//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__MEMORY_RESOURCE_RESOURCE_H
#define _CUDA__MEMORY_RESOURCE_RESOURCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !defined(_CCCL_COMPILER_MSVC_2017) && defined(LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE)

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/std/__concepts/all_of.h>
#  include <cuda/std/__concepts/equality_comparable.h>
#  include <cuda/std/__concepts/same_as.h>
#  include <cuda/std/__type_traits/decay.h>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

/// \concept resource
/// \brief The \c resource concept
template <class _Resource>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __resource_,
  requires(_Resource& __res, void* __ptr, size_t __bytes, size_t __alignment)(
    requires(_CUDA_VSTD::same_as<void*, decltype(__res.allocate(__bytes, __alignment))>),
    requires(_CUDA_VSTD::same_as<void, decltype(__res.deallocate(__ptr, __bytes, __alignment))>),
    requires(_CUDA_VSTD::equality_comparable<_Resource>)));

template <class _Resource>
_LIBCUDACXX_CONCEPT resource = _LIBCUDACXX_FRAGMENT(__resource_, _Resource);

/// \concept async_resource
/// \brief The \c async_resource concept
template <class _Resource>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __async_resource_,
  requires(_Resource& __res, void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)(
    requires(resource<_Resource>),
    requires(_CUDA_VSTD::same_as<void*, decltype(__res.allocate_async(__bytes, __alignment, __stream))>),
    requires(_CUDA_VSTD::same_as<void, decltype(__res.deallocate_async(__ptr, __bytes, __alignment, __stream))>)));

template <class _Resource>
_LIBCUDACXX_CONCEPT async_resource = _LIBCUDACXX_FRAGMENT(__async_resource_, _Resource);

/// \concept resource_with
/// \brief The \c resource_with concept
template <class _Resource, class... _Properties>
_LIBCUDACXX_CONCEPT resource_with =
  resource<_Resource> && _CUDA_VSTD::__all_of<has_property<_Resource, _Properties>...>;

/// \concept async_resource_with
/// \brief The \c async_resource_with concept
template <class _Resource, class... _Properties>
_LIBCUDACXX_CONCEPT async_resource_with =
  async_resource<_Resource> && _CUDA_VSTD::__all_of<has_property<_Resource, _Properties>...>;

template <class _Resource, class _OtherResource>
_LIBCUDACXX_CONCEPT __different_resource =
  (!_CUDA_VSTD::same_as<_CUDA_VSTD::decay_t<_Resource>, _CUDA_VSTD::decay_t<_OtherResource>>) &&resource<_OtherResource>;

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_RESOURCE_H
