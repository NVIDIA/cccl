//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_ADAPTER_H
#define _CUDA___MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_ADAPTER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK()

#  include <cuda/__memory_resource/get_property.h>
#  include <cuda/__memory_resource/properties.h>
#  include <cuda/__memory_resource/resource.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/stream>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_MR

template <class _Resource>
_CCCL_CONCEPT __has_member_allocate =
  _CCCL_REQUIRES_EXPR((_Resource), _Resource& __res, ::cuda::stream_ref __stream, size_t __bytes, size_t __alignment)(
    _Same_as(void*) __res.allocate(__stream, __bytes, __alignment));

template <class _Resource>
_CCCL_CONCEPT __has_member_deallocate = _CCCL_REQUIRES_EXPR(
  (_Resource), _Resource& __res, ::cuda::stream_ref __stream, void* __ptr, size_t __bytes, size_t __alignment)(
  _Same_as(void) __res.deallocate(__stream, __ptr, __bytes, __alignment));

//! @brief Adapter that allows a synchronous resource to be used as a resource
//! It examines the resource for the presence of the allocate and deallocate
//! members. If they are present, it passes through the allocate and deallocate
//! calls to the contained resource. Otherwise, it uses the allocate_sync and
//! deallocate_sync members (with proper synchronization in case of deallocate).
//! @note This adapter takes ownership of the contained resource.
//! @tparam _Resource The type of the resource to be adapted
template <class _Resource>
struct synchronous_resource_adapter
    : ::cuda::mr::__copy_default_queries<_Resource>
    , ::cuda::forward_property<synchronous_resource_adapter<_Resource>, _Resource>
{
  _CCCL_HOST_API synchronous_resource_adapter(const _Resource& __resource) noexcept
      : __resource(__resource)
  {}

  _CCCL_HOST_API synchronous_resource_adapter(_Resource&& __resource) noexcept
      : __resource(__resource)
  {}

  [[nodiscard]] _CCCL_HOST_API void*
  allocate([[maybe_unused]] const ::cuda::stream_ref __stream, const size_t __bytes, const size_t __alignment)
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

#  if _CCCL_STD_VER <= 2017
  [[nodiscard]] _CCCL_HOST_API bool operator!=(const synchronous_resource_adapter& __rhs) const noexcept
  {
    return __resource != __rhs.__resource;
  }
#  endif // _CCCL_STD_VER <= 2017

  _CCCL_HOST_API _Resource& upstream_resource() noexcept
  {
    return __resource;
  }

  _CCCL_HOST_API const _Resource& upstream_resource() const noexcept
  {
    return __resource;
  }

private:
  _Resource __resource;
};

template <class _Resource>
_CCCL_HOST_API decltype(auto) __adapt_if_synchronous(_Resource&& __resource) noexcept
{
  if constexpr (::cuda::mr::resource<::cuda::std::decay_t<_Resource>>)
  {
    return ::cuda::std::forward<_Resource>(__resource);
  }
  else
  {
    return synchronous_resource_adapter<::cuda::std::decay_t<_Resource>>(::cuda::std::forward<_Resource>(__resource));
  }
}
_CCCL_END_NAMESPACE_CUDA_MR

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif //_CUDA___MEMORY_RESOURCE_SYNCHRONOUS_RESOURCE_ADAPTER_H
