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
#  include <thrust/system/cuda/config.h>

#  include <thrust/system/cuda/detail/mismatch.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class InputIt1, class InputIt2, class BinaryPred>
bool _CCCL_HOST_DEVICE
equal(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPred binary_pred)
{
  return cuda_cub::mismatch(policy, first1, last1, first2, binary_pred).first == last1;
}

template <class Derived, class InputIt1, class InputIt2>
bool _CCCL_HOST_DEVICE equal(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
  using InputType1 = thrust::detail::it_value_t<InputIt1>;
  return cuda_cub::equal(policy, first1, last1, first2, ::cuda::std::equal_to<InputType1>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
