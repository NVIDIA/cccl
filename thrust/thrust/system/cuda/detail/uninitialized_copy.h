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
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/transform.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/type_traits/is_trivially_relocatable.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/incrementable_traits.h>
#  include <cuda/std/__new/device_new.h>
#  include <cuda/std/__type_traits/is_trivially_copy_assignable.h>
#  include <cuda/std/__type_traits/is_trivially_copy_constructible.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
// forward declare and not include thrust/system/cuda/detail/copy.h to avoid a circular dependency
template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator _CCCL_HOST_DEVICE
copy_n(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result);

namespace __uninitialized_copy
{
template <class InputIt, class OutputIt>
struct functor
{
  InputIt input;
  OutputIt output;

  using InputType  = thrust::detail::it_value_t<InputIt>;
  using OutputType = thrust::detail::it_value_t<OutputIt>;

  template <class Size>
  void _CCCL_DEVICE_API _CCCL_FORCEINLINE operator()(Size idx)
  {
    InputType const& in = raw_reference_cast(input[idx]);
    OutputType& out     = raw_reference_cast(output[idx]);
    ::new (static_cast<void*>(&out)) OutputType(in);
  }
};
} // namespace __uninitialized_copy

template <class Derived, class InputIt, class Size, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
uninitialized_copy_n(execution_policy<Derived>& policy, InputIt first, Size count, OutputIt result)
{
  // if the output type is trivially constructible from the input, it has no side effect, and we can skip placement new
  // and calling a constructor. Furthermore, if assigning an input to an output element is also trivial, there is no
  // copy constructor which could have a side effect and we can delegate to copy_n.
  using input_ref_t  = thrust::detail::raw_reference_t<::cuda::std::iter_reference_t<InputIt>>;
  using output_ref_t = thrust::detail::raw_reference_t<::cuda::std::iter_reference_t<OutputIt>>;
  using output_t     = thrust::detail::it_value_t<OutputIt>;
  if constexpr (::cuda::std::is_trivially_constructible_v<output_t, input_ref_t>
                && ::cuda::std::is_trivially_assignable_v<output_ref_t, input_ref_t>)
  {
    cuda_cub::copy_n(policy, first, count, result);
  }
  else
  {
    cuda_cub::parallel_for(policy, __uninitialized_copy::functor<InputIt, OutputIt>{first, result}, count);
  }

  return result + count;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
uninitialized_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  return cuda_cub::uninitialized_copy_n(policy, first, ::cuda::std::distance(first, last), result);
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
