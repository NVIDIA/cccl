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

#ifdef _CCCL_CUDA_COMPILER

#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_scan.cuh>

#  include <thrust/detail/integer_math.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/detail/type_traits.h>
#  include <thrust/distance.h>
#  include <thrust/iterator/iterator_traits.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>

#  include <cstdint>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
namespace detail
{

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n_impl(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt result, ScanOp scan_op)
{
  using AccumT     = typename thrust::iterator_traits<InputIt>::value_type;
  using Dispatch32 = cub::DispatchScan<InputIt, OutputIt, ScanOp, cub::NullType, std::uint32_t, AccumT>;
  using Dispatch64 = cub::DispatchScan<InputIt, OutputIt, ScanOp, cub::NullType, std::uint64_t, AccumT>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status;

  // Negative number of items are normalized to `0`
  if (thrust::detail::is_negative(num_items))
  {
    num_items = 0;
  }

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (nullptr, tmp_size, first, result, scan_op, cub::NullType{}, num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for inclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, tmp_size};
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (tmp.data().get(), tmp_size, first, result, scan_op, cub::NullType{}, num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching inclusive_scan kernel");
    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy), "inclusive_scan failed to synchronize");
  }

  return result + num_items;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename InitValueT, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n_impl(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  InitValueT init,
  ScanOp scan_op)
{
  using InputValueT = cub::detail::InputValue<InitValueT>;
  using AccumT      = typename ::cuda::std::__accumulator_t<ScanOp, cub::detail::value_t<InputIt>, InitValueT>;
  constexpr bool ForceInclusive = true;

  using Dispatch32 =
    cub::DispatchScan<InputIt,
                      OutputIt,
                      ScanOp,
                      InputValueT,
                      std::int32_t,
                      AccumT,
                      cub::DeviceScanPolicy<AccumT, ScanOp>,
                      ForceInclusive>;
  using Dispatch64 =
    cub::DispatchScan<InputIt,
                      OutputIt,
                      ScanOp,
                      InputValueT,
                      std::int64_t,
                      AccumT,
                      cub::DeviceScanPolicy<AccumT, ScanOp>,
                      ForceInclusive>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status;

  // Negative number of items are normalized to `0`
  if (thrust::detail::is_negative(num_items))
  {
    num_items = 0;
  }

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (nullptr, tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for inclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, tmp_size};
    THRUST_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (tmp.data().get(), tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching inclusive_scan kernel");
    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy), "inclusive_scan failed to synchronize");
  }

  return result + num_items;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename InitValueT, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt exclusive_scan_n_impl(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  InitValueT init,
  ScanOp scan_op)
{
  using InputValueT = cub::detail::InputValue<InitValueT>;
  using Dispatch32  = cub::DispatchScan<InputIt, OutputIt, ScanOp, InputValueT, std::uint32_t, InitValueT>;
  using Dispatch64  = cub::DispatchScan<InputIt, OutputIt, ScanOp, InputValueT, std::uint64_t, InitValueT>;

  cudaStream_t stream = thrust::cuda_cub::stream(policy);
  cudaError_t status;

  // Negative number of items are normalized to `0`
  if (thrust::detail::is_negative(num_items))
  {
    num_items = 0;
  }

  // Determine temporary storage requirements:
  size_t tmp_size = 0;
  {
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (nullptr, tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(
      status,
      "after determining tmp storage "
      "requirements for exclusive_scan");
  }

  // Run scan:
  {
    // Allocate temporary storage:
    thrust::detail::temporary_array<std::uint8_t, Derived> tmp{policy, tmp_size};
    THRUST_UNSIGNED_INDEX_TYPE_DISPATCH2(
      status,
      Dispatch32::Dispatch,
      Dispatch64::Dispatch,
      num_items,
      (tmp.data().get(), tmp_size, first, result, scan_op, InputValueT(init), num_items_fixed, stream));
    thrust::cuda_cub::throw_on_error(status, "after dispatching exclusive_scan kernel");
    thrust::cuda_cub::throw_on_error(
      thrust::cuda_cub::synchronize_optional(policy), "exclusive_scan failed to synchronize");
  }

  return result + num_items;
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  THRUST_CDP_DISPATCH(
    (result = thrust::cuda_cub::detail::inclusive_scan_n_impl(policy, first, num_items, result, init, scan_op);),
    (result =
       thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)), first, first + num_items, result, init, scan_op);));
  return result;
}

template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan_n(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt result, ScanOp scan_op)
{
  THRUST_CDP_DISPATCH(
    (result = thrust::cuda_cub::detail::inclusive_scan_n_impl(policy, first, num_items, result, scan_op);),
    (result = thrust::inclusive_scan(cvt_to_seq(derived_cast(policy)), first, first + num_items, result, scan_op);));
  return result;
}

template <typename Derived, typename InputIt, typename OutputIt, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, ScanOp scan_op)
{
  using diff_t           = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::cuda_cub::inclusive_scan_n(policy, first, num_items, result, scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt inclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  using diff_t           = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::cuda_cub::inclusive_scan_n(policy, first, num_items, result, init, scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt>
_CCCL_HOST_DEVICE OutputIt
inclusive_scan(thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  return thrust::cuda_cub::inclusive_scan(policy, first, last, result, thrust::plus<>{});
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt exclusive_scan_n(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  Size num_items,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  THRUST_CDP_DISPATCH(
    (result = thrust::cuda_cub::detail::exclusive_scan_n_impl(policy, first, num_items, result, init, scan_op);),
    (result =
       thrust::exclusive_scan(cvt_to_seq(derived_cast(policy)), first, first + num_items, result, init, scan_op);));
  return result;
}

template <typename Derived, typename InputIt, typename OutputIt, typename T, typename ScanOp>
_CCCL_HOST_DEVICE OutputIt exclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  T init,
  ScanOp scan_op)
{
  using diff_t           = typename thrust::iterator_traits<InputIt>::difference_type;
  diff_t const num_items = thrust::distance(first, last);
  return thrust::cuda_cub::exclusive_scan_n(policy, first, num_items, result, init, scan_op);
}

template <typename Derived, typename InputIt, typename OutputIt, typename T>
_CCCL_HOST_DEVICE OutputIt exclusive_scan(
  thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, T init)
{
  return thrust::cuda_cub::exclusive_scan(policy, first, last, result, init, thrust::plus<>{});
}

template <typename Derived, typename InputIt, typename OutputIt>
_CCCL_HOST_DEVICE OutputIt
exclusive_scan(thrust::cuda_cub::execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  using init_type = typename thrust::iterator_traits<InputIt>::value_type;
  return cuda_cub::exclusive_scan(policy, first, last, result, init_type{});
};

} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/scan.h>

#endif // NVCC
