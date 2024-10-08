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
#  include <cuda/std/__concepts/__concept_macros.h>
#  include <cuda/std/__concepts/convertible_to.h>
#  include <cuda/std/__concepts/equality_comparable.h>
#  include <cuda/std/__concepts/same_as.h>
#  include <cuda/std/__type_traits/decay.h>
#  include <cuda/std/__type_traits/fold.h>
#  include <cuda/stream_ref>

#  if _CCCL_STD_VER >= 2014

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA_MR

template <class _Resource>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __resource_,
  requires(_Resource& __res, void* __ptr, size_t __bytes, size_t __alignment)(
    requires(_CUDA_VSTD::same_as<void*, decltype(__res.allocate(__bytes, __alignment))>),
    requires(_CUDA_VSTD::same_as<void, decltype(__res.deallocate(__ptr, __bytes, __alignment))>),
    requires(_CUDA_VSTD::equality_comparable<_Resource>)));

//! @brief The \c resource concept verifies that a type Resource satisfies the basic requirements of a memory
//! resource
//! @rst
//! We require that a resource supports the following interface
//!
//!   - ``allocate(size_t bytes, size_t alginment)``
//!   - ``deallocate(void* ptr, size_t bytes, size_t alginment)``
//!   - ``T() == T()``
//!   - ``T() != T()``
//!
//! @endrst
//! @tparam _Resource The type that should implement the resource concept
template <class _Resource>
_LIBCUDACXX_CONCEPT resource = _LIBCUDACXX_FRAGMENT(__resource_, _Resource);

template <class _Resource>
_LIBCUDACXX_CONCEPT_FRAGMENT(
  __async_resource_,
  requires(_Resource& __res, void* __ptr, size_t __bytes, size_t __alignment, ::cuda::stream_ref __stream)(
    requires(resource<_Resource>),
    requires(_CUDA_VSTD::same_as<void*, decltype(__res.allocate_async(__bytes, __alignment, __stream))>),
    requires(_CUDA_VSTD::same_as<void, decltype(__res.deallocate_async(__ptr, __bytes, __alignment, __stream))>)));

//! @brief The \c async_resource concept verifies that a type Resource satisfies the basic requirements of a
//! memory resource and additionally supports stream ordered allocations
//! @rst
//! We require that an async resource supports the following interface
//!
//!   - ``allocate(size_t bytes, size_t alignment)``
//!   - ``deallocate(void* ptr, size_t bytes, size_t alignment)``
//!   - ``T() == T()``
//!   - ``T() != T()``
//!
//!   - ``allocate_async(size_t bytes, size_t alignment, cuda::stream_ref stream)``
//!   - ``deallocate_async(void* ptr, size_t bytes, size_t alignment, cuda::stream_ref stream)``
//!
//! @endrst
//! @tparam _Resource The type that should implement the async resource concept
template <class _Resource>
_LIBCUDACXX_CONCEPT async_resource = _LIBCUDACXX_FRAGMENT(__async_resource_, _Resource);

//! @brief The \c resource_with concept verifies that a type Resource satisfies the `resource` concept and
//! also satisfies all the provided Properties
//! @tparam _Resource
//! @tparam _Properties
template <class _Resource, class... _Properties>
_LIBCUDACXX_CONCEPT resource_with =
  resource<_Resource> && _CUDA_VSTD::__fold_and<has_property<_Resource, _Properties>...>;

//! @brief The \c async_resource_with concept verifies that a type Resource satisfies the `async_resource`
//! concept and also satisfies all the provided Properties
//! @tparam _Resource
//! @tparam _Properties
template <class _Resource, class... _Properties>
_LIBCUDACXX_CONCEPT async_resource_with =
  async_resource<_Resource> && _CUDA_VSTD::__fold_and<has_property<_Resource, _Properties>...>;

template <bool _Convertible>
struct __different_resource__
{
  template <class _OtherResource>
  static constexpr bool __value(_OtherResource*) noexcept
  {
    return resource<_OtherResource>;
  }
};

template <>
struct __different_resource__<true>
{
  static constexpr bool __value(void*) noexcept
  {
    return false;
  }
};

template <class _Resource, class _OtherResource>
_LIBCUDACXX_CONCEPT __different_resource =
  __different_resource__<_CUDA_VSTD::convertible_to<_OtherResource const&, _Resource const&>>::__value(
    static_cast<_OtherResource*>(nullptr));

_LIBCUDACXX_END_NAMESPACE_CUDA_MR

#  endif // _CCCL_STD_VER >= 2014

#endif // !_CCCL_COMPILER_MSVC_2017 && LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE

#endif //_CUDA__MEMORY_RESOURCE_RESOURCE_H
