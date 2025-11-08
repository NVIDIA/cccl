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

/*! \file trivial_copy.h
 *  \brief Sequential copy algorithms for plain-old-data.
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
#include <thrust/system/detail/sequential/general_copy.h>

#include <cuda/std/cstring>

#include <nv/target>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
template <typename T>
_CCCL_HOST_DEVICE T* trivial_copy_n(const T* first, std::ptrdiff_t n, T* result)
{
  if (n == 0)
  {
    // If `first` or `result` is an invalid pointer,
    // the behavior of `cuda::std::memmove` is undefined, even if `n` is zero.
    return result;
  }

  ::cuda::std::memmove(result, first, n * sizeof(T));

  return result + n;
} // end trivial_copy_n()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
