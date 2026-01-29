/*
 *  Copyright 2008-2025 NVIDIA Corporation
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

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace system
{
namespace hpx
{
namespace detail
{

using thrust::is_contiguous_iterator;
using thrust::is_contiguous_iterator_v;

using thrust::try_unwrap_contiguous_iterator;
using thrust::unwrap_contiguous_iterator;

template <typename Pointer, typename Iterator>
constexpr Iterator rewrap_contiguous_iterator(Pointer it, Iterator base)
{
  return base + (it - detail::unwrap_contiguous_iterator(base));
}

template <typename Iterator>
constexpr Iterator rewrap_contiguous_iterator(Iterator it, Iterator /*base*/)
{
  return it;
}

} // namespace detail
} // namespace hpx
} // namespace system

THRUST_NAMESPACE_END
