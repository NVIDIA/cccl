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

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/iterator/iterator_traits.h>
#include <thrust/system/detail/generic/advance.h>

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN
namespace system::detail::generic
{
_CCCL_EXEC_CHECK_DISABLE
template <typename InputIterator, typename Distance>
_CCCL_HOST_DEVICE void advance(InputIterator& i, Distance n)
{
  using traversal = typename iterator_traversal<InputIterator>::type;
  if constexpr (::cuda::std::is_convertible_v<traversal, random_access_traversal_tag>)
  {
    i += n;
  }
  else
  {
    while (n)
    {
      ++i;
      --n;
    }
  }
}
} // namespace system::detail::generic
THRUST_NAMESPACE_END
