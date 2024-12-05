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

#if _CCCL_HAS_CUDA_COMPILER
#  include <thrust/distance.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{

namespace __fill
{

// fill functor
template <class Iterator, class T>
struct functor
{
  Iterator it;
  T value;

  THRUST_FUNCTION
  functor(Iterator it, T value)
      : it(it)
      , value(value)
  {}

  template <class Size>
  THRUST_DEVICE_FUNCTION void operator()(Size idx)
  {
    it[idx] = value;
  }
}; // struct functor

} // namespace __fill

template <class Derived, class OutputIterator, class Size, class T>
OutputIterator _CCCL_HOST_DEVICE
fill_n(execution_policy<Derived>& policy, OutputIterator first, Size count, const T& value)
{
  cuda_cub::parallel_for(policy, __fill::functor<OutputIterator, T>(first, value), count);

  return first + count;
} // func fill_n

template <class Derived, class ForwardIterator, class T>
void _CCCL_HOST_DEVICE
fill(execution_policy<Derived>& policy, ForwardIterator first, ForwardIterator last, const T& value)
{
  cuda_cub::fill_n(policy, first, thrust::distance(first, last), value);
} // func filll

} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif
