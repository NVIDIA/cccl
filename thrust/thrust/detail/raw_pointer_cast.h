// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA Corporation. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/pointer_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename Pointer>
_CCCL_HOST_DEVICE auto raw_pointer_cast(Pointer ptr)
{
  return ::cuda::std::to_address(ptr);
}

template <typename ToPointer, typename FromPointer>
_CCCL_HOST_DEVICE ToPointer reinterpret_pointer_cast(FromPointer ptr)
{
  using to_element = typename ::cuda::std::pointer_traits<ToPointer>::element_type;
  return ToPointer(reinterpret_cast<to_element*>(::cuda::std::to_address(ptr)));
}

template <typename ToPointer, typename FromPointer>
_CCCL_HOST_DEVICE ToPointer static_pointer_cast(FromPointer ptr)
{
  using to_element = typename ::cuda::std::pointer_traits<ToPointer>::element_type;
  return ToPointer(static_cast<to_element*>(::cuda::std::to_address(ptr)));
}

THRUST_NAMESPACE_END
