/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file device_make_unique.h
 *  \brief A factory function for creating `unique_ptr`s to device objects.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/allocate_unique.h>
#include <thrust/detail/type_deduction.h>
#include <thrust/device_allocator.h>
#include <thrust/device_new.h>
#include <thrust/device_ptr.h>

THRUST_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////

template <typename T, typename... Args>
_CCCL_HOST auto device_make_unique(Args&&... args)
  -> decltype(uninitialized_allocate_unique<T>(::cuda::std::declval<device_allocator<T>>()))
{
  // FIXME: This is crude - we construct an unnecessary T on the host for
  // `device_new`. We need a proper dispatched `construct` algorithm to
  // do this properly.
  auto p = uninitialized_allocate_unique<T>(device_allocator<T>());
  device_new<T>(p.get(), T(THRUST_FWD(args)...));
  return p;
}

///////////////////////////////////////////////////////////////////////////////

THRUST_NAMESPACE_END
