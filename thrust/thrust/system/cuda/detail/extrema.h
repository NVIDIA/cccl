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

#  include <thrust/detail/temporary_array.h>
#  include <thrust/extrema.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>

#  include <cuda/__cmath/round_up.h>
#  include <cuda/__iterator/discard_iterator.h>
#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__utility/pair.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace __extrema
{
template <class Derived, class ItemsIt, class BinaryPred>
ItemsIt CUB_RUNTIME_FUNCTION
cub_min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  cudaStream_t stream      = cuda_cub::stream(policy);
  using offset_t           = thrust::detail::it_difference_t<ItemsIt>;
  const offset_t num_items = ::cuda::std::distance(first, last);

  if (num_items == 0)
  {
    return last;
  }

  ::cuda::std::size_t tmp_size = 0;
  auto error                   = cub::DeviceReduce::ArgMin(
    nullptr,
    tmp_size,
    first,
    ::cuda::discard_iterator{},
    static_cast<offset_t*>(nullptr),
    num_items,
    binary_pred,
    stream);
  throw_on_error(error, "min_element failed to allocate temporary storages");

  // We allocate both the temporary storage needed for the algorithm, and a `size_type` to store the result.
  thrust::detail::temporary_array<char, Derived> tmp(policy, sizeof(offset_t) + tmp_size);
  offset_t* index_ptr = thrust::detail::aligned_reinterpret_cast<offset_t*>(tmp.data().get());
  auto tmp_ptr        = static_cast<void*>(tmp.data().get() + sizeof(offset_t));

  error = cub::DeviceReduce::ArgMin(
    tmp_ptr, tmp_size, first, ::cuda::discard_iterator{}, index_ptr, num_items, binary_pred, stream);
  cuda_cub::throw_on_error(error, "min_element failed to launch cub::DeviceReduce::ArgMin");

  cuda_cub::throw_on_error(cuda_cub::synchronize(policy), "min_element failed to synchronize");

  return first + get_value(policy, index_ptr);
}

template <class Derived, class ItemsIt, class BinaryPred>
::cuda::std::pair<ItemsIt, ItemsIt> CUB_RUNTIME_FUNCTION
cub_minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred)
{
  using offset_t           = thrust::detail::it_difference_t<ItemsIt>;
  const offset_t num_items = ::cuda::std::distance(first, last);
  if (num_items == 0)
  {
    return {first, first};
  }

  const cudaStream_t stream = cuda_cub::stream(policy);

  // TODO(bgruber): with CCCL 4.0 switch to cub::DeviceReduce::ArgMinLastMax to conform to the C++ standard
  ::cuda::std::size_t tmp_size = 0;
  auto error                   = cub::DeviceReduce::ArgMinMax(
    nullptr,
    tmp_size,
    first,
    ::cuda::discard_iterator{},
    static_cast<offset_t*>(nullptr),
    ::cuda::discard_iterator{},
    static_cast<offset_t*>(nullptr),
    num_items,
    binary_pred,
    stream);
  throw_on_error(error, "minmax_element failed to determine temporary storage size");

  // Round tmp_size up to the alignment of offset_t so the index slots are properly aligned.
  const auto aligned_tmp_size = ::cuda::round_up(tmp_size, alignof(offset_t));
  // Allocate: the algorithm's temporary storage followed by two index slots (min, max).
  thrust::detail::temporary_array<char, Derived> tmp(policy, aligned_tmp_size + 2 * sizeof(offset_t));
  void* const tmp_ptr       = static_cast<void*>(tmp.data().get());
  offset_t* const min_index = thrust::detail::aligned_reinterpret_cast<offset_t*>(tmp.data().get() + aligned_tmp_size);
  offset_t* const max_index = min_index + 1;

  // TODO(bgruber): with CCCL 4.0 switch to cub::DeviceReduce::ArgMinLastMax to conform to the C++ standard
  error = cub::DeviceReduce::ArgMinMax(
    tmp_ptr,
    tmp_size,
    first,
    ::cuda::discard_iterator{},
    min_index,
    ::cuda::discard_iterator{},
    max_index,
    num_items,
    binary_pred,
    stream);
  cuda_cub::throw_on_error(error, "minmax_element failed to launch cub::DeviceReduce::ArgMinMax");

  offset_t host_indices[2];
  cuda_cub::throw_on_error(
    ::cudaMemcpyAsync(host_indices, min_index, 2 * sizeof(offset_t), ::cudaMemcpyDeviceToHost, stream),
    "minmax_element failed to copy indices to host");
  cuda_cub::throw_on_error(cuda_cub::synchronize(policy), "minmax_element failed to synchronize");
  return {first + host_indices[0], first + host_indices[1]};
}
} // namespace __extrema

/// min element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<thrust::detail::it_value_t<ItemsIt>>>
ItemsIt _CCCL_HOST_DEVICE
min_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  THRUST_CDP_DISPATCH(({ return __extrema::cub_min_element(policy, first, last, binary_pred); }),
                      ({ return thrust::min_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred); }));
}

/// max element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<thrust::detail::it_value_t<ItemsIt>>>
ItemsIt _CCCL_HOST_DEVICE
max_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  THRUST_CDP_DISPATCH(
    ({ return __extrema::cub_min_element(policy, first, last, cub::detail::swap_args{binary_pred}); }),
    ({ return thrust::max_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred); }));
}

/// minmax element

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class ItemsIt, class BinaryPred = ::cuda::std::less<thrust::detail::it_value_t<ItemsIt>>>
::cuda::std::pair<ItemsIt, ItemsIt> _CCCL_HOST_DEVICE
minmax_element(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, BinaryPred binary_pred = {})
{
  THRUST_CDP_DISPATCH(({ return __extrema::cub_minmax_element(policy, first, last, binary_pred); }),
                      ({ return thrust::minmax_element(cvt_to_seq(derived_cast(policy)), first, last, binary_pred); }));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
