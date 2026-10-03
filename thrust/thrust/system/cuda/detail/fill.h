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
#  include <cub/device/device_transform.cuh>

#  include <thrust/detail/allocator/temporary_allocator.h>
#  include <thrust/detail/pointer.h>
#  include <thrust/fill.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__memory/addressof.h>
#  include <cuda/std/cstddef>
#  include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __fill
{
// PTX ISA 8.1 (CTK 12.1) raised the kernel parameter limit from 4096 to 32764 bytes.
inline constexpr ::cuda::std::size_t kernel_param_limit = __cccl_ptx_isa >= 810ULL ? 32764 : 4096;

// Largest value passed to a kernel by value. The margin leaves room for the kernel's other parameters, like the output
// iterator and the number of items. Larger values are copied to device memory and read through a pointer.
inline constexpr ::cuda::std::size_t max_value_param_size = kernel_param_limit - 256;

template <class T>
inline constexpr bool pass_value_in_device_memory = sizeof(T) > max_value_param_size;

// Owns a bytewise copy of a host value in temporary device memory, like the copy a kernel parameter would receive. The
// copy synchronizes the stream, so the caller may modify or destroy the value afterwards. Not built on temporary_array,
// because its header transitively includes cuda/detail/uninitialized_fill.h, which needs the definitions in this
// header.
template <class T, class Derived>
class device_copy
{
  thrust::detail::temporary_allocator<T, Derived> allocator;
  thrust::pointer<T, Derived> ptr;

  _CCCL_HOST explicit device_copy(execution_policy<Derived>& policy)
      : allocator(policy)
      , ptr(allocator.allocate(1))
  {}

public:
  // Delegates the allocation, so the destructor releases it if the copy throws
  _CCCL_HOST device_copy(execution_policy<Derived>& policy, const T& value)
      : device_copy(policy)
  {
    throw_on_error(trivial_copy_to_device(ptr.get(), ::cuda::std::addressof(value), 1, cuda_cub::stream(policy)),
                   "fill: failed to copy the value to device memory");
  }

  device_copy(const device_copy&)            = delete;
  device_copy& operator=(const device_copy&) = delete;

  _CCCL_HOST ~device_copy()
  {
    allocator.deallocate(ptr, 1);
  }

  [[nodiscard]] _CCCL_HOST const T* get() const
  {
    return ptr.get();
  }
};

template <class Iterator, class T>
struct assign_from_pointer
{
  Iterator items;
  const T* value;

  template <class Size>
  void _CCCL_DEVICE_API _CCCL_FORCEINLINE operator()(Size idx)
  {
    items[idx] = *value;
  }
};

template <class Derived, class OutputIterator, class Size, class T>
_CCCL_HOST void
fill_n_from_device_copy(execution_policy<Derived>& policy, OutputIterator first, Size count, const T& value)
{
  if (count == 0)
  {
    return;
  }

  const device_copy<T, Derived> device_value(policy, value);
  cuda_cub::parallel_for(policy, assign_from_pointer<OutputIterator, T>{first, device_value.get()}, count);
}
} // namespace __fill

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class OutputIterator, class Size, class T>
OutputIterator _CCCL_HOST_DEVICE
fill_n(execution_policy<Derived>& policy, OutputIterator first, Size count, const T& value)
{
  THRUST_CDP_DISPATCH(({
                        if constexpr (__fill::pass_value_in_device_memory<T>)
                        {
                          __fill::fill_n_from_device_copy(policy, first, count, value);
                        }
                        else
                        {
                          cudaError_t status;
                          THRUST_INDEX_TYPE_DISPATCH(
                            status,
                            (CUB_NS_QUALIFIER::DeviceTransform::Fill),
                            count,
                            (first, count_fixed, value, cuda_cub::stream(policy)));
                          throw_on_error(status, "fill_n: failed inside CUB");
                          throw_on_error(synchronize_optional(policy), "fill_n: failed to synchronize");
                        }
                        return first + count;
                      }),
                      ({ return thrust::fill_n(cvt_to_seq(derived_cast(policy)), first, count, value); }));
}

template <class Derived, class ForwardIterator, class T>
void _CCCL_HOST_DEVICE
fill(execution_policy<Derived>& policy, ForwardIterator first, ForwardIterator last, const T& value)
{
  cuda_cub::fill_n(policy, first, ::cuda::std::distance(first, last), value);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
