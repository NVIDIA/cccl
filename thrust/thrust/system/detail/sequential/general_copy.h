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

/*! \file general_copy.h
 *  \brief Sequential copy algorithms for general iterators.
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
#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/type_traits.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
namespace general_copy_detail
{
// sometimes OutputIterator's reference type is reported as void
// in that case, just assume that we're able to assign to it OK
template <typename InputIterator, typename OutputIterator>
struct reference_is_assignable
{
  template <typename OI>
  static constexpr bool h()
  {
    if constexpr (::cuda::std::is_same_v<thrust::detail::it_reference_t<OI>, void>)
    {
      return true;
    }
    else
    {
      return ::cuda::std::is_assignable_v<thrust::detail::it_reference_t<OI>,
                                          thrust::detail::it_reference_t<InputIterator>>;
    }
  }

  static constexpr bool value = h<OutputIterator>();
};

// introduce an iterator assign helper to deal with assignments from
// a wrapped reference

_CCCL_EXEC_CHECK_DISABLE
template <typename OutputIterator, typename InputIterator>
inline _CCCL_HOST_DEVICE ::cuda::std::enable_if_t<reference_is_assignable<InputIterator, OutputIterator>::value>
iter_assign(OutputIterator dst, InputIterator src)
{
  *dst = *src;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename OutputIterator, typename InputIterator>
inline _CCCL_HOST_DEVICE
typename thrust::detail::disable_if<reference_is_assignable<InputIterator, OutputIterator>::value>::type
iter_assign(OutputIterator dst, InputIterator src)
{
  using value_type = thrust::detail::it_value_t<InputIterator>;

  // insert a temporary and hope for the best
  *dst = static_cast<value_type>(*src);
}
} // namespace general_copy_detail

_CCCL_EXEC_CHECK_DISABLE
template <typename InputIterator, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator general_copy(InputIterator first, InputIterator last, OutputIterator result)
{
  for (; first != last; ++first, (void) ++result)
  {
    // gcc 4.2 crashes while instantiating iter_assign
#if _CCCL_COMPILER(GCC, <, 4, 3)
    *result = *first;
#else
    general_copy_detail::iter_assign(result, first);
#endif
  }

  return result;
} // end general_copy()

_CCCL_EXEC_CHECK_DISABLE
template <typename InputIterator, typename Size, typename OutputIterator>
_CCCL_HOST_DEVICE OutputIterator general_copy_n(InputIterator first, Size n, OutputIterator result)
{
  for (; n > Size(0); ++first, (void) ++result, (void) --n)
  {
    // gcc 4.2 crashes while instantiating iter_assign
#if _CCCL_COMPILER(GCC, <, 4, 3)
    *result = *first;
#else
    general_copy_detail::iter_assign(result, first);
#endif
  }

  return result;
} // end general_copy_n()
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
