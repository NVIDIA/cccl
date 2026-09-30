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

#include <thrust/detail/type_traits.h>
#include <thrust/device_ptr.h>
#include <thrust/device_reference.h>
#include <thrust/iterator/iterator_traits.h>

THRUST_NAMESPACE_BEGIN

template <typename T>
_CCCL_HOST_DEVICE device_ptr<T> device_pointer_cast(T* ptr)
{
  return device_ptr<T>(ptr);
} // end device_pointer_cast()

template <typename T>
_CCCL_HOST_DEVICE device_ptr<T> device_pointer_cast(const device_ptr<T>& ptr)
{
  return ptr;
} // end device_pointer_cast()

THRUST_NAMESPACE_END
