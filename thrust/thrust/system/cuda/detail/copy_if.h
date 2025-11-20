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

#  include <cub/device/device_select.cuh>
#  include <cub/util_math.cuh>
#  include <cub/util_temporary_storage.cuh>
#  include <cub/util_type.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/function.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/util.h>
#  include <thrust/system/cuda/detail/dispatch.h>
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
template <cub::SelectImpl SelectionOpt,
          typename Derived,
          typename InputIt,
          typename StencilIt,
          typename OutputIt,
          typename Predicate,
          typename OffsetT>
struct DispatchCopyIf
{
  static cudaError_t THRUST_RUNTIME_FUNCTION dispatch(
    execution_policy<Derived>& policy,
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIt first,
    StencilIt stencil,
    OutputIt& output,
    Predicate predicate,
    OffsetT num_items)
  {
    using num_selected_out_it_t = OffsetT*;
    using equality_op_t         = cub::NullType;

    cudaError_t status  = cudaSuccess;
    cudaStream_t stream = cuda_cub::stream(policy);

    std::size_t allocation_sizes[2] = {0, sizeof(OffsetT)};
    void* allocations[2]            = {nullptr, nullptr};

    // Query algorithm memory requirements
    status = cub::
      DispatchSelectIf<InputIt, StencilIt, OutputIt, num_selected_out_it_t, Predicate, equality_op_t, OffsetT, SelectionOpt>::
        Dispatch(
          nullptr,
          allocation_sizes[0],
          first,
          stencil,
          output,
          static_cast<num_selected_out_it_t>(nullptr),
          predicate,
          equality_op_t{},
          num_items,
          stream);
    _CUDA_CUB_RET_IF_FAIL(status);

    status = cub::detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    _CUDA_CUB_RET_IF_FAIL(status);

    // Return if we're only querying temporary storage requirements
    if (d_temp_storage == nullptr)
    {
      return status;
    }

    // Return for empty problems
    if (num_items == 0)
    {
      return status;
    }

    // Memory allocation for the number of selected output items
    OffsetT* d_num_selected_out = thrust::detail::aligned_reinterpret_cast<OffsetT*>(allocations[1]);

    // Run algorithm
    status = cub::
      DispatchSelectIf<InputIt, StencilIt, OutputIt, num_selected_out_it_t, Predicate, equality_op_t, OffsetT, SelectionOpt>::
        Dispatch(
          allocations[0],
          allocation_sizes[0],
          first,
          stencil,
          output,
          d_num_selected_out,
          predicate,
          equality_op_t{},
          num_items,
          stream);
    _CUDA_CUB_RET_IF_FAIL(status);

    // Get number of selected items
    status = cuda_cub::synchronize(policy);
    _CUDA_CUB_RET_IF_FAIL(status);
    OffsetT num_selected = get_value(policy, d_num_selected_out);
    ::cuda::std::advance(output, num_selected);
    return status;
  }
};

template <cub::SelectImpl SelectionOpt,
          typename Derived,
          typename InputIt,
          typename StencilIt,
          typename OutputIt,
          typename Predicate>
THRUST_RUNTIME_FUNCTION OutputIt copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt output,
  Predicate predicate)
{
  using size_type = thrust::detail::it_difference_t<InputIt>;

  size_type num_items       = static_cast<size_type>(::cuda::std::distance(first, last));
  cudaError_t status        = cudaSuccess;
  size_t temp_storage_bytes = 0;

  // 64-bit offset-type dispatch
  // Since https://github.com/NVIDIA/cccl/pull/2400, cub::DeviceSelect is using a streaming approach that splits up
  // inputs larger than INT_MAX into partitions of up to `INT_MAX` items each, repeatedly invoking the respective
  // algorithm. With that approach, we can always use i64 offset types for DispatchSelectIf, because there's only very
  // limited performance upside for using i32 offset types. This avoids potentially duplicate kernel compilation.
  using dispatch64_t = DispatchCopyIf<SelectionOpt, Derived, InputIt, StencilIt, OutputIt, Predicate, std::int64_t>;

  // Query temporary storage requirements
  status = dispatch64_t::dispatch(
    policy, nullptr, temp_storage_bytes, first, stencil, output, predicate, static_cast<std::int64_t>(num_items));
  cuda_cub::throw_on_error(status, "copy_if failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, temp_storage_bytes);
  void* temp_storage = static_cast<void*>(tmp.data().get());

  // Run algorithm
  status = dispatch64_t::dispatch(
    policy, temp_storage, temp_storage_bytes, first, stencil, output, predicate, static_cast<std::int64_t>(num_items));
  cuda_cub::throw_on_error(status, "copy_if failed on 2nd step");

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
  THRUST_CDP_DISPATCH((return detail::copy_if<cub::SelectImpl::Select>(
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
  THRUST_CDP_DISPATCH((return detail::copy_if<cub::SelectImpl::Select>(policy, first, last, stencil, result, pred);),
                      (return thrust::copy_if(cvt_to_seq(derived_cast(policy)), first, last, stencil, result, pred);));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/copy.h>
#endif // _CCCL_CUDA_COMPILATION()
