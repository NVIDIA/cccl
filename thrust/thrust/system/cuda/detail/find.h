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

#  include <cub/device/device_find.cuh>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/util.h>

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

#  include <thrust/find.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{
template <typename Derived, typename InputIt, typename Size, typename Predicate>
THRUST_RUNTIME_FUNCTION Size
find_if_n_impl(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  cudaStream_t stream = cuda_cub::stream(policy);

  auto call_with_adjusted_size_type = [&](auto num_items_fixed) -> Size {
    // we use the same offset type that CUB uses internally for writing the result. avoids an extra kernel
    using adjusted_size_type = cub::detail::choose_offset_t<decltype(num_items_fixed)>;

    size_t tmp_size = 0;
    auto status     = cub::DeviceFind::FindIf(
      nullptr, tmp_size, first, static_cast<adjusted_size_type*>(nullptr), predicate, num_items_fixed, stream);
    cuda_cub::throw_on_error(status, "find_if: failed to get temp storage size");

    // Allocate temporary storage for both the algorithm and the result.
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, sizeof(adjusted_size_type) + tmp_size);

    // Run find_if.
    adjusted_size_type* result_ptr = thrust::detail::aligned_reinterpret_cast<adjusted_size_type*>(tmp.data().get());
    void* tmp_ptr                  = static_cast<void*>((tmp.data() + sizeof(adjusted_size_type)).get());

    status = cub::DeviceFind::FindIf(tmp_ptr, tmp_size, first, result_ptr, predicate, num_items_fixed, stream);
    cuda_cub::throw_on_error(status, "find_if: failed to run algorithm");

    return static_cast<Size>(cuda_cub::get_value(policy, result_ptr));
  };

  Size result;
  THRUST_INDEX_TYPE_DISPATCH(result, call_with_adjusted_size_type, num_items, (num_items_fixed));
  return result;
}
} // namespace detail

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class Size, class Predicate>
InputIt _CCCL_HOST_DEVICE
find_if_n(execution_policy<Derived>& policy, InputIt first, Size num_items, Predicate predicate)
{
  if (num_items == 0)
  {
    return first;
  }

  Size result_idx = num_items;
  THRUST_CDP_DISPATCH(
    (result_idx = cuda_cub::detail::find_if_n_impl(policy, first, num_items, predicate);),
    (result_idx = thrust::find_if(cvt_to_seq(derived_cast(policy)), first, first + num_items, predicate) - first;));

  return first + result_idx;
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
