/*
 *  Copyright 2008-2013 NVIDIA Corporation
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

/*! \file thrust/future.h
 *  \brief `thrust::future`, an asynchronous value type.
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
#include <thrust/detail/cpp14_required.h>

#if _CCCL_STD_VER >= 2014

#  include <thrust/detail/static_assert.h>
#  include <thrust/execution_policy.h>

#  include <utility>

/*
// #include the host system's pointer.h header.
#define __THRUST_HOST_SYSTEM_POINTER_HEADER <__THRUST_HOST_SYSTEM_ROOT/pointer.h>
  #include __THRUST_HOST_SYSTEM_POINTER_HEADER
#undef __THRUST_HOST_SYSTEM_POINTER_HEADER
*/

// #include the device system's pointer.h header.
#  define __THRUST_DEVICE_SYSTEM_POINTER_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/pointer.h>
#  include __THRUST_DEVICE_SYSTEM_POINTER_HEADER
#  undef __THRUST_DEVICE_SYSTEM_POINTER_HEADER

/*
// #include the host system's future.h header.
#define __THRUST_HOST_SYSTEM_FUTURE_HEADER <__THRUST_HOST_SYSTEM_ROOT/future.h>
  #include __THRUST_HOST_SYSTEM_FUTURE_HEADER
#undef __THRUST_HOST_SYSTEM_FUTURE_HEADER
*/

// #include the device system's future.h header.
#  define __THRUST_DEVICE_SYSTEM_FUTURE_HEADER <__THRUST_DEVICE_SYSTEM_ROOT/future.h>
#  include __THRUST_DEVICE_SYSTEM_FUTURE_HEADER
#  undef __THRUST_DEVICE_SYSTEM_FUTURE_HEADER

_CCCL_SUPPRESS_DEPRECATED_PUSH
THRUST_NAMESPACE_BEGIN

///////////////////////////////////////////////////////////////////////////////

// `select_unique_(future|event)_type` is a hook for choosing the
// `unique_eager_event`/`unique_eager_future` type for a system. `decltype` is
// used to determine the return type of an ADL call to
// `select_unique_eager_(future|event)_type(system)`; that return type should
// be the correct event/future type for `system`. Overloads should only be
// declared, not defined.

namespace unimplemented
{

struct CCCL_DEPRECATED no_unique_eager_event_type_found
{};

CCCL_DEPRECATED _CCCL_HOST inline no_unique_eager_event_type_found unique_eager_event_type(...) noexcept;

struct CCCL_DEPRECATED no_unique_eager_future_type_found
{};

template <typename T>
CCCL_DEPRECATED _CCCL_HOST no_unique_eager_future_type_found unique_eager_future_type(...) noexcept;

} // namespace unimplemented

namespace unique_eager_event_type_detail
{

using unimplemented::unique_eager_event_type;

template <typename System>
using select CCCL_DEPRECATED = decltype(unique_eager_event_type(std::declval<System>()));

} // namespace unique_eager_event_type_detail

namespace unique_eager_future_type_detail
{

using unimplemented::unique_eager_future_type;

template <typename System, typename T>
using select CCCL_DEPRECATED = decltype(unique_eager_future_type<T>(std::declval<System>()));

} // namespace unique_eager_future_type_detail

///////////////////////////////////////////////////////////////////////////////

template <typename System>
using unique_eager_event CCCL_DEPRECATED = unique_eager_event_type_detail::select<System>;

template <typename System>
using event CCCL_DEPRECATED = unique_eager_event<System>;

///////////////////////////////////////////////////////////////////////////////

template <typename System, typename T>
using unique_eager_future CCCL_DEPRECATED = unique_eager_future_type_detail::select<System, T>;

template <typename System, typename T>
using future CCCL_DEPRECATED = unique_eager_future<System, T>;

/*
///////////////////////////////////////////////////////////////////////////////

using host_unique_eager_event = unique_eager_event_type_detail::select<
  thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::tag
>;
using host_event = host_unique_eager_event;

///////////////////////////////////////////////////////////////////////////////

template <typename T>
using host_unique_eager_future = unique_eager_future_type_detail::select<
  thrust::system::__THRUST_HOST_SYSTEM_NAMESPACE::tag, T
>;
template <typename T>
using host_future = host_unique_eager_future<T>;
*/

///////////////////////////////////////////////////////////////////////////////

using device_unique_eager_event CCCL_DEPRECATED =
  unique_eager_event_type_detail::select<thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::tag>;

using device_event CCCL_DEPRECATED = device_unique_eager_event;

///////////////////////////////////////////////////////////////////////////////

template <typename T>
using device_unique_eager_future CCCL_DEPRECATED =
  unique_eager_future_type_detail::select<thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::tag, T>;

template <typename T>
using device_future CCCL_DEPRECATED = device_unique_eager_future<T>;

///////////////////////////////////////////////////////////////////////////////

struct CCCL_DEPRECATED new_stream_t final
{};

#  ifndef CCCL_HEADER_MACRO_CHECK
// when building header tests, we get a deprecation warning from cudafe1.stub.c if we deprecate a global variable
CCCL_DEPRECATED
#  endif
_CCCL_GLOBAL_CONSTANT new_stream_t new_stream{};

///////////////////////////////////////////////////////////////////////////////

using thrust::system::__THRUST_DEVICE_SYSTEM_NAMESPACE::when_all;

///////////////////////////////////////////////////////////////////////////////

_CCCL_SUPPRESS_DEPRECATED_POP
THRUST_NAMESPACE_END

#endif
