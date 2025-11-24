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
#  include <thrust/iterator/transform_iterator.h>
#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/cuda/detail/reduce.h>
#  include <thrust/zip_function.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
template <class Derived, class InputIt1, class InputIt2, class T, class ReduceOp, class ProductOp>
T _CCCL_HOST_DEVICE inner_product(
  execution_policy<Derived>& policy,
  InputIt1 first1,
  InputIt1 last1,
  InputIt2 first2,
  T init,
  ReduceOp reduce_op,
  ProductOp product_op)
{
  const auto n     = ::cuda::std::distance(first1, last1);
  const auto first = make_transform_iterator(make_zip_iterator(first1, first2), make_zip_function(product_op));
  return cuda_cub::reduce_n(policy, first, n, init, reduce_op);
}

template <class Derived, class InputIt1, class InputIt2, class T>
T _CCCL_HOST_DEVICE
inner_product(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, T init)
{
  return cuda_cub::inner_product(
    policy, first1, last1, first2, init, ::cuda::std::plus<T>(), ::cuda::std::multiplies<T>());
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
