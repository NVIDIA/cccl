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
#include <thrust/system/detail/sequential/execution_policy.h>

#include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace system::detail::sequential
{
_CCCL_EXEC_CHECK_DISABLE
template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator1,
          typename OutputIterator2,
          typename BinaryPredicate,
          typename BinaryFunction>
_CCCL_HOST_DEVICE ::cuda::std::pair<OutputIterator1, OutputIterator2> reduce_by_key(
  sequential::execution_policy<DerivedPolicy>&,
  InputIterator1 keys_first,
  InputIterator1 keys_last,
  InputIterator2 values_first,
  OutputIterator1 keys_output,
  OutputIterator2 values_output,
  BinaryPredicate binary_pred,
  BinaryFunction binary_op)
{
  using InputKeyType   = thrust::detail::it_value_t<InputIterator1>;
  using InputValueType = thrust::detail::it_value_t<InputIterator2>;

  // Use the input iterator's value type per https://wg21.link/P0571
  using TemporaryType = thrust::detail::it_value_t<InputIterator2>;

  if (keys_first != keys_last)
  {
    InputKeyType temp_key    = *keys_first;
    TemporaryType temp_value = *values_first;

    for (++keys_first, ++values_first; keys_first != keys_last; ++keys_first, (void) ++values_first)
    {
      InputKeyType key     = *keys_first;
      InputValueType value = *values_first;

      if (binary_pred(temp_key, key))
      {
        temp_value = binary_op(temp_value, value);
      }
      else
      {
        *keys_output   = temp_key;
        *values_output = temp_value;

        ++keys_output;
        ++values_output;

        temp_key   = key;
        temp_value = value;
      }
    }

    *keys_output   = temp_key;
    *values_output = temp_value;

    ++keys_output;
    ++values_output;
  }

  return ::cuda::std::make_pair(keys_output, values_output);
}
} // namespace system::detail::sequential
THRUST_NAMESPACE_END
