/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()
#  include <thrust/system/cuda/config.h>

#  include <thrust/system/cuda/detail/execution_policy.h>

#  include <cuda/__iterator/counting_iterator.h>
#  include <cuda/__iterator/transform_iterator.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
// XXX forward declare to circumvent circular dependency
template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate);

template <class Derived, class InputIt, class T>
InputIt _CCCL_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value);
}; // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/system/cuda/detail/reduce.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __find_if
{
template <typename TupleType>
struct functor
{
  _CCCL_DEVICE_API _CCCL_FORCEINLINE TupleType operator()(const TupleType& lhs, const TupleType& rhs) const
  {
    // select the smallest index among true results
    if (thrust::get<0>(lhs) && thrust::get<0>(rhs))
    {
      return TupleType(true, (::cuda::std::min) (thrust::get<1>(lhs), thrust::get<1>(rhs)));
    }
    else if (thrust::get<0>(lhs))
    {
      return lhs;
    }
    else
    {
      return rhs;
    }
  }
};
} // namespace __find_if

template <class Derived, class InputIt, class Size, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_n(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  using result_type = ::cuda::std::tuple<bool, Size>;

  // empty sequence
  if (num_items == 0)
  {
    return first;
  }

  // this implementation breaks up the sequence into separate intervals
  // in an attempt to early-out as soon as a value is found
  //
  // XXX compose find_if from a look-back prefix scan algorithm
  //     and abort kernel when the first element is found

  // TODO incorporate sizeof(InputType) into interval_threshold and round to multiple of 32
  const Size interval_threshold = 1 << 20;
  const Size interval_size      = (::cuda::std::min) (interval_threshold, num_items);

  const auto begin = ::cuda::make_zip_iterator(
    ::cuda::make_transform_iterator(try_unwrap_contiguous_iterator(first), predicate),
    ::cuda::counting_iterator<Size>(0));
  const auto end = begin + num_items;

  for (auto interval_begin = begin; interval_begin < end; interval_begin += interval_size)
  {
    auto interval_end = interval_begin + interval_size;
    if (end < interval_end)
    {
      interval_end = end;
    } // end if

    const result_type result = reduce(
      policy, interval_begin, interval_end, result_type(false, interval_end - begin), __find_if::functor<result_type>());

    // see if we found something
    if (thrust::get<0>(result))
    {
      return first + thrust::get<1>(result);
    }
  }

  // nothing was found if we reach here...
  return first + num_items;
}

template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE find_if(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return cuda_cub::find_if_n(policy, first, ::cuda::std::distance(first, last), predicate);
}

template <class Derived, class InputIt, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_not(execution_policy<Derived>& policy, InputIt first, InputIt last, Predicate predicate)
{
  return cuda_cub::find_if(policy, first, last, ::cuda::std::not_fn(predicate));
}

template <class Derived, class InputIt, class T>
InputIt _CCCL_HOST_DEVICE find(execution_policy<Derived>& policy, InputIt first, InputIt last, T const& value)
{
  using thrust::placeholders::_1;

  return cuda_cub::find_if(policy, first, last, _1 == value);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
