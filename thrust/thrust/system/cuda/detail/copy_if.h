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

#  include <cub/device/device_select.cuh>
#  include <cub/util_temporary_storage.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__iterator/advance.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN
// XXX declare generic copy_if interface
// to avoid circulular dependency from thrust/copy.h
template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename Predicate>
_CCCL_HOST_DEVICE OutputIterator copy_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  Predicate pred);

template <typename DerivedPolicy,
          typename InputIterator1,
          typename InputIterator2,
          typename OutputIterator,
          typename Predicate>
_CCCL_HOST_DEVICE OutputIterator copy_if(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator1 first,
  InputIterator1 last,
  InputIterator2 stencil,
  OutputIterator result,
  Predicate pred);

namespace cuda_cub
{
namespace detail
{
template <bool InPlace, typename Derived, typename InputIt, typename StencilIt, typename OutputIt, typename Predicate>
THRUST_RUNTIME_FUNCTION OutputIt copy_if_with_stencil(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt output,
  Predicate predicate)
{
  using offset_t            = std::int64_t;
  const auto num_items      = ::cuda::std::distance(first, last);
  const cudaStream_t stream = cuda_cub::stream(policy);

  // We need to allocate space for the num_selected output alongside the algorithm's temp storage
  std::size_t allocation_sizes[2] = {0, sizeof(offset_t)};
  void* allocations[2]            = {nullptr, nullptr};

  // Query temp storage for the algorithm
  cudaError_t status;
  if constexpr (InPlace)
  {
    status = cub::DeviceSelect::FlaggedIf(
      nullptr,
      allocation_sizes[0],
      first,
      stencil,
      static_cast<offset_t*>(nullptr),
      static_cast<offset_t>(num_items),
      predicate,
      stream);
  }
  else
  {
    status = cub::DeviceSelect::FlaggedIf(
      nullptr,
      allocation_sizes[0],
      first,
      stencil,
      output,
      static_cast<offset_t*>(nullptr),
      static_cast<offset_t>(num_items),
      predicate,
      stream);
  }
  cuda_cub::throw_on_error(status, "copy_if failed on 1st step");

  size_t temp_storage_bytes = 0;
  status = cub::detail::alias_temporaries(nullptr, temp_storage_bytes, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "copy_if failed on temp storage query");

  // Allocate temporary storage
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  status = cub::detail::alias_temporaries(temp_storage, temp_storage_bytes, allocations, allocation_sizes);
  cuda_cub::throw_on_error(status, "copy_if failed on temp storage alias");

  offset_t* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<offset_t*>(allocations[1]);

  // Run algorithm
  if constexpr (InPlace)
  {
    status = cub::DeviceSelect::FlaggedIf(
      allocations[0],
      allocation_sizes[0],
      first,
      stencil,
      d_num_selected_out,
      static_cast<offset_t>(num_items),
      predicate,
      stream);
  }
  else
  {
    status = cub::DeviceSelect::FlaggedIf(
      allocations[0],
      allocation_sizes[0],
      first,
      stencil,
      output,
      d_num_selected_out,
      static_cast<offset_t>(num_items),
      predicate,
      stream);
  }
  cuda_cub::throw_on_error(status, "copy_if failed on 2nd step");

  // Get number of selected items
  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "copy_if failed on sync");
  offset_t num_selected = get_value(policy, d_num_selected_out);
  ::cuda::std::advance(output, num_selected);
  return output;
}
} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------
_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIterator, class OutputIterator, class Predicate>
OutputIterator _CCCL_HOST_DEVICE copy_if(
  execution_policy<Derived>& policy, InputIterator first, InputIterator last, OutputIterator result, Predicate pred)
{
  THRUST_CDP_DISPATCH((return detail::copy_if_with_stencil</* InPlace */ false>(
                                policy, first, last, static_cast<cub::NullType*>(nullptr), result, pred);),
                      (return thrust::copy_if(cvt_to_seq(derived_cast(policy)), first, last, result, pred);));
}

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIterator, class StencilIterator, class OutputIterator, class Predicate>
OutputIterator _CCCL_HOST_DEVICE copy_if(
  execution_policy<Derived>& policy,
  InputIterator first,
  InputIterator last,
  StencilIterator stencil,
  OutputIterator result,
  Predicate pred)
{
  THRUST_CDP_DISPATCH(
    (return detail::copy_if_with_stencil</* InPlace */ false>(policy, first, last, stencil, result, pred);),
    (return thrust::copy_if(cvt_to_seq(derived_cast(policy)), first, last, stencil, result, pred);));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/copy.h>
#endif // _CCCL_CUDA_COMPILATION()
