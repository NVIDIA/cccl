//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_ADAPTER_CUH
#define _CUDAX__MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_ADAPTER_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__memory_resource/properties.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/stream>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <class _Resource>
_CCCL_CONCEPT __has_member_allocate =
  _CCCL_REQUIRES_EXPR((_Resource), _Resource& __res, ::cuda::stream_ref __stream, size_t __bytes, size_t __alignment)(
    _Same_as(void*) __res.allocate(__stream, __bytes, __alignment));

template <class _Resource>
_CCCL_CONCEPT __has_member_deallocate = _CCCL_REQUIRES_EXPR(
  (_Resource), _Resource& __res, ::cuda::stream_ref __stream, void* __ptr, size_t __bytes, size_t __alignment)(
  _Same_as(void) __res.deallocate(__stream, __ptr, __bytes, __alignment));

template <class _Resource>
struct synchronous_resource_adapter : ::cuda::mr::__copy_default_queries<_Resource>
{
  synchronous_resource_adapter(const _Resource& __resource) noexcept
      : __resource(__resource)
  {}

  synchronous_resource_adapter(_Resource&& __resource) noexcept
      : __resource(__resource)
  {}

  [[nodiscard]] _CCCL_HOST_API void*
  allocate(const ::cuda::stream_ref __stream, const size_t __bytes, const size_t __alignment)
  {
    if constexpr (__has_member_allocate<_Resource>)
    {
      return __resource.allocate(__stream, __bytes, __alignment);
    }
    else
    {
      return __resource.allocate_sync(__bytes, __alignment);
    }
  }

  [[nodiscard]] _CCCL_HOST_API void* allocate_sync(const size_t __bytes, const size_t __alignment)
  {
    return __resource.allocate_sync(__bytes, __alignment);
  }

  _CCCL_HOST_API void
  deallocate(const ::cuda::stream_ref __stream, void* __ptr, const size_t __bytes, const size_t __alignment) noexcept
  {
    if constexpr (__has_member_deallocate<_Resource>)
    {
      __resource.deallocate(__stream, __ptr, __bytes, __alignment);
    }
    else
    {
      ::cuda::__driver::__streamSynchronizeNoThrow(__stream.get());
      __resource.deallocate_sync(__ptr, __bytes, __alignment);
    }
  }

  _CCCL_HOST_API void deallocate_sync(void* __ptr, const size_t __bytes, const size_t __alignment) noexcept
  {
    __resource.deallocate_sync(__ptr, __bytes, __alignment);
  }

  [[nodiscard]] _CCCL_HOST_API bool operator==(const synchronous_resource_adapter& __rhs) const noexcept
  {
    return __resource == __rhs.__resource;
  }

#if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API bool operator!=(const synchronous_resource_adapter& __rhs) const noexcept
  {
    return __resource != __rhs.__resource;
  }
#endif // _CCCL_STD_VER <= 2017

  template <class _Property>
  friend constexpr void get_property(const synchronous_resource_adapter& __res, _Property __prop) noexcept
  {
    __res.__resource.get_property(__prop);
  }

private:
  _Resource __resource;
};

template <class _Resource>
_CCCL_HOST_API auto __adapt_if_synchronous(_Resource&& __resource) noexcept
{
  if constexpr (::cuda::mr::resource<::cuda::std::decay_t<_Resource>>)
  {
    return __resource;
  }
  else
  {
    return synchronous_resource_adapter<::cuda::std::decay_t<_Resource>>(::cuda::std::forward<_Resource>(__resource));
  }
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //_CUDAX__MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_ADAPTER_CUH
