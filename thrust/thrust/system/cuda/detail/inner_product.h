// SPDX-FileCopyrightText: Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

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
