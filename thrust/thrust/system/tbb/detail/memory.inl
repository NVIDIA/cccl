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
#include <thrust/system/cpp/detail/execution_policy.h>
#include <thrust/system/cpp/memory.h>
#include <thrust/system/tbb/memory.h>

#include <cuda/std/limits>

THRUST_NAMESPACE_BEGIN
namespace system::tbb
{

namespace detail
{

// XXX circular #inclusion problems cause the compiler to believe that cpp::malloc
//     is not defined
//     WAR the problem by using adl to call cpp::malloc, which requires it to depend
//     on a template parameter
template <typename Tag>
pointer<void> malloc_workaround(Tag t, std::size_t n)
{
  return pointer<void>(malloc(t, n));
} // end malloc_workaround()

// XXX circular #inclusion problems cause the compiler to believe that cpp::free
//     is not defined
//     WAR the problem by using adl to call cpp::free, which requires it to depend
//     on a template parameter
template <typename Tag>
void free_workaround(Tag t, pointer<void> ptr)
{
  free(t, ptr.get());
} // end free_workaround()

} // namespace detail

inline pointer<void> malloc(std::size_t n)
{
  // XXX this is how we'd like to implement this function,
  //     if not for circular #inclusion problems:
  //
  // return pointer<void>(thrust::system::cpp::malloc(n))
  //
  return detail::malloc_workaround(cpp::tag(), n);
} // end malloc()

template <typename T>
pointer<T> malloc(std::size_t n)
{
  pointer<void> raw_ptr = thrust::system::tbb::malloc(sizeof(T) * n);
  return pointer<T>(reinterpret_cast<T*>(raw_ptr.get()));
} // end malloc()

inline void free(pointer<void> ptr)
{
  // XXX this is how we'd like to implement this function,
  //     if not for circular #inclusion problems:
  //
  // thrust::system::cpp::free(ptr)
  //
  detail::free_workaround(cpp::tag(), ptr);
} // end free()

} // namespace system::tbb
THRUST_NAMESPACE_END
