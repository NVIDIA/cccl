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
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/fill.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/uninitialized_fill.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__new/device_new.h>
#  include <cuda/std/__type_traits/is_trivially_assignable.h>
#  include <cuda/std/__type_traits/is_trivially_constructible.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
namespace __uninitialized_fill
{
template <class Iterator, class T>
struct functor
{
  Iterator items;
  T value;

  using value_type = thrust::detail::it_value_t<Iterator>;

  template <class Size>
  void _CCCL_DEVICE_API _CCCL_FORCEINLINE operator()(Size idx)
  {
    value_type& out = raw_reference_cast(items[idx]);
    ::new (static_cast<void*>(&out)) value_type(value);
  }
};

// Like functor, but reads the value through a pointer, see __fill::pass_value_in_device_memory
template <class Iterator, class T>
struct construct_from_pointer
{
  Iterator items;
  const T* value;

  using value_type = thrust::detail::it_value_t<Iterator>;

  template <class Size>
  void _CCCL_DEVICE_API _CCCL_FORCEINLINE operator()(Size idx)
  {
    value_type& out = raw_reference_cast(items[idx]);
    ::new (static_cast<void*>(&out)) value_type(*value);
  }
};

template <class Derived, class Iterator, class Size, class T>
_CCCL_HOST void
uninitialized_fill_n_from_device_copy(execution_policy<Derived>& policy, Iterator first, Size count, const T& value)
{
  if (count == 0)
  {
    return;
  }

  const __fill::device_copy<T, Derived> device_value(policy, value);
  cuda_cub::parallel_for(policy, construct_from_pointer<Iterator, T>{first, device_value.get()}, count);
}
} // namespace __uninitialized_fill

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class Iterator, class Size, class T>
Iterator _CCCL_HOST_DEVICE
uninitialized_fill_n(execution_policy<Derived>& policy, Iterator first, Size count, T const& x)
{
  // if the output type is trivially constructible from the input, it has no side effect, and we can skip placement new
  // and calling a constructor. Furthermore, if assigning the input value to an output element is also trivial, there is
  // no copy constructor which could have a side effect and we can delegate to fill_n (which uses
  // cub::DeviceTransform::Fill).
  using value_t = thrust::detail::it_value_t<Iterator>;
  if constexpr (::cuda::std::is_trivially_constructible_v<value_t, T const&>
                && ::cuda::std::is_trivially_assignable_v<value_t, T const&>)
  {
    cuda_cub::fill_n(policy, first, count, x);
  }
  else if constexpr (__fill::pass_value_in_device_memory<T>)
  {
    THRUST_CDP_DISPATCH((__uninitialized_fill::uninitialized_fill_n_from_device_copy(policy, first, count, x);),
                        (thrust::uninitialized_fill_n(cvt_to_seq(derived_cast(policy)), first, count, x);));
  }
  else
  {
    cuda_cub::parallel_for(policy, __uninitialized_fill::functor<Iterator, T>{first, x}, count);
  }
  return first + count;
}

template <class Derived, class Iterator, class T>
void _CCCL_HOST_DEVICE uninitialized_fill(execution_policy<Derived>& policy, Iterator first, Iterator last, T const& x)
{
  cuda_cub::uninitialized_fill_n(policy, first, ::cuda::std::distance(first, last), x);
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
