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

#  include <thrust/system/cuda/detail/execution_policy.h>

#  include <cuda/__iterator/zip_function.h>
#  include <cuda/__iterator/zip_iterator.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class InputIt1, class InputIt2, class BinaryPred>
::cuda::std::pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPred binary_pred);

template <class Derived, class InputIt1, class InputIt2>
::cuda::std::pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2);
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/system/cuda/detail/find.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class InputIt1, class InputIt2, class BinaryPred>
::cuda::std::pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2, BinaryPred binary_pred)
{
  const auto n            = ::cuda::std::distance(first1, last1);
  const auto first        = ::cuda::make_zip_iterator(first1, first2);
  const auto last         = ::cuda::make_zip_iterator(last1, first2 + n);
  const auto mismatch_pos = cuda_cub::find_if_not(policy, first, last, ::cuda::zip_function(binary_pred));
  const auto dist         = ::cuda::std::distance(first, mismatch_pos);
  return ::cuda::std::make_pair(first1 + dist, first2 + dist);
}

template <class Derived, class InputIt1, class InputIt2>
::cuda::std::pair<InputIt1, InputIt2> _CCCL_HOST_DEVICE
mismatch(execution_policy<Derived>& policy, InputIt1 first1, InputIt1 last1, InputIt2 first2)
{
  using InputType1 = thrust::detail::it_value_t<InputIt1>;
  return cuda_cub::mismatch(policy, first1, last1, first2, ::cuda::std::equal_to<InputType1>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
