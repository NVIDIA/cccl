// SPDX-FileCopyrightText: Copyright (c) 2008-2018, NVIDIA Corporation. All rights reserved.
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
#include <thrust/system/cpp/detail/malloc_and_free.h>
#include <thrust/system/cpp/memory.h>

#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN
namespace system::cpp
{
pointer<void> malloc(std::size_t n)
{
  tag t;
  return pointer<void>(thrust::system::detail::sequential::malloc(t, n));
} // end malloc()

template <typename T>
pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::system::cpp::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

void free(pointer<void> ptr)
{
  tag t;
  return thrust::system::detail::sequential::free(t, ptr);
} // end free()
} // namespace system::cpp
THRUST_NAMESPACE_END
