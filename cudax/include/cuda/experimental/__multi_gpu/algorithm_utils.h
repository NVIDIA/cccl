// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_UTILS_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_reduce.cuh>
#include <cub/device/dispatch/kernels/kernel_reduce.cuh>

#include <thrust/system/cuda/detail/dispatch.h>
#include <thrust/system/cuda/detail/util.h>

#include <cuda/__container/buffer.h>
#include <cuda/__device/device_ref.h>
#include <cuda/__functional/lazy_call_or.h>
#include <cuda/__memory_pool/device_memory_pool.h>
#include <cuda/__memory_resource/get_memory_resource.h>
#include <cuda/__memory_resource/resource.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/__utility/no_init.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__ranges/concepts.h>
#include <cuda/std/__ranges/size.h>
#include <cuda/std/__ranges/zip_view.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__multi_gpu/communicator.h>
#include <cuda/experimental/__multi_gpu/concepts.h>

#include <vector>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental::__detail
{
#define __CUDAX_MULTI_GPU_DISPATCH(logical_device, count, call, arguments)                \
  do                                                                                      \
  {                                                                                       \
    const auto __cur_context = __ensure_current_context{(logical_device).context()};      \
    auto __status            = ::cudaError_t{};                                           \
                                                                                          \
    THRUST_INDEX_TYPE_DISPATCH(__status, call, count, arguments);                         \
    ::thrust::cuda_cub::throw_on_error(__status, /*msg=*/"performing " #call #arguments); \
  } while (0)

#define __CUDAX_MULTI_GPU_DOUBLE_DISPATCH(logical_device, count1, count2, call, arguments) \
  do                                                                                       \
  {                                                                                        \
    const auto __cur_context = __ensure_current_context{(logical_device).context()};       \
    auto __status            = ::cudaError_t{};                                            \
                                                                                           \
    THRUST_DOUBLE_INDEX_TYPE_DISPATCH(__status, call, count1, count2, arguments);          \
    ::thrust::cuda_cub::throw_on_error(__status, /*msg=*/"performing " #call #arguments);  \
  } while (0)

template <class _Range0, class... _Ranges>
[[nodiscard]]
_CCCL_HOST_API constexpr ::cuda::std::size_t __verify_range_sizes(const _Range0& __range0, const _Ranges&... __ranges)
{
  if constexpr (::cuda::std::ranges::sized_range<_Range0>)
  {
    const auto __size = ::cuda::std::ranges::size(__range0);

    (
      [&] {
        if constexpr (::cuda::std::ranges::sized_range<_Ranges>)
        {
          _CCCL_VERIFY(__size == ::cuda::std::ranges::size(__ranges), "Range sizes must match");
        }
      }(),
      ...);
    return __size;
  }
  else if constexpr (sizeof...(_Ranges) > 0)
  {
    return __verify_range_sizes(__ranges...);
  }
  else
  {
    return 0;
  }
}

template <class _Buffer, class _Env>
struct __partial_redop
{
  using __buffer_type = _Buffer;
  using __env_type    = _Env;

  _Buffer __buffer;
  _Env __env;
  ::cuda::stream_ref __stream;
};

template <class _Env>
[[nodiscard]] ::cuda::stream_ref __stream_from_env(_Env&& __env)
{
  return ::cuda::__lazy_call_or(
    ::cuda::get_stream,
    [] {
      return ::cuda::stream_ref{::CUstream{}};
    },
    __env);
}

template <class _Env>
[[nodiscard]] auto __resource_from_env(_Env&& __env, const logical_device& __logical_device)
{
  return ::cuda::__lazy_call_or(
    ::cuda::mr::get_memory_resource,
    [&] {
      return ::cuda::device_default_memory_pool(__logical_device.underlying_device());
    },
    __env);
}

template <class _Buffer, class _Env, class _InputRange, class _Tp, class _BinaryOp>
[[nodiscard]] __partial_redop<_Buffer, _Env>
__local_reduction(const communicator& __comm, _Env __env, _InputRange&& __input_range, _Tp __init, _BinaryOp __op)
{
  const auto& __logical_device = __comm.device();
  auto __stream                = __stream_from_env(__env);
  auto __resource              = __resource_from_env(__env, __logical_device);

  static_assert(::cuda::mr::resource_with<decltype(__resource), ::cuda::mr::device_accessible>,
                "Provided memory resource must be device accessible");

  const auto __num_items = ::cuda::std::ranges::size(__input_range);

  // Allocate enough storage so that we can use the buffer directly in a
  // ncclAllGather/AllReduce call. Those calls require that the receive buffer is of size
  // nranks * sendcount.
  auto __buff = _Buffer{__stream, ::cuda::std::move(__resource), __comm.size(), ::cuda::no_init, __env};

  __CUDAX_MULTI_GPU_DISPATCH(
    __logical_device,
    __num_items,
    ::cub::DeviceReduce::Reduce,
    (::cuda::std::ranges::begin(__input_range),
     // Similarly to above, prepare for the NCCL calls later. In order for those to be in-place,
     // the sendbuff = recvbuff + rank so we need to place our partial result there
     __buff.begin() + __comm.rank(),
     __num_items_fixed,
     ::cuda::std::move(__op),
     ::cub::detail::reduce::empty_problem_init_t<_Tp>{::cuda::std::move(__init)},
     __env));

  return {::cuda::std::move(__buff), ::cuda::std::move(__env), __stream};
}

// Returns a vector of partial reductions. Each entry in the vector is a
// __partial_redop<__buffer_type, __env_type> that holds a buffer of size __comm.size(), where
// each entry in those temporaries is the rank-local reduction value.
//
// The only non-deduced template parameter is _ResultTp which should be the type of temporary
// that should be reduced into.
template <class _ResultTp, class _GroupRange, class _EnvRange, class _InputRange, class _Tp, class _BinaryOp>
[[nodiscard]] auto __partial_reduction(
  _GroupRange&& __comms, _EnvRange&& __envs, _InputRange&& __range_of_input_ranges, _Tp __init, _BinaryOp __op)
{
  using __env_type = ::cuda::std::remove_cvref_t<::cuda::std::ranges::range_reference_t<_EnvRange>>;

  static_assert(::cuda::is_trivially_copyable_v<_ResultTp>,
                "Result values must be trivially copyable to be transportable across devices");

  using __pool_type = decltype(::cuda::device_default_memory_pool(::cuda::std::declval<::cuda::device_ref>()));
  using __resource_type =
    ::cuda::__lazy_call_result_or_t<::cuda::mr::get_memory_resource_t, __pool_type(void), __env_type>;
  using __buffer_type = ::cuda::__buffer_type_for_props<_ResultTp, typename __resource_type::default_queries>;

  ::std::vector<__partial_redop<__buffer_type, __env_type>> __partials;

  __partials.reserve(__verify_range_sizes(__comms, __envs, __range_of_input_ranges));
  // TODO(jfaibussowit): can just be ranges::zip | ranges::transform | ranges::to() (and then
  // we don't need to do the env, and buffer type deduction upfront)
  for (auto&& [__comm, __env, __input_range] :
       ::cuda::std::ranges::views::zip(__comms, __envs, __range_of_input_ranges))
  {
    __partials.emplace_back(__local_reduction<__buffer_type>(__comm, __env, __input_range, __init, __op));
  }

  {
    const auto _ = ::cuda::experimental::__nccl::__auto_nccl_group{};

    for (const auto& [__comm, __local] : ::cuda::std::ranges::views::zip(__comms, __partials))
    {
      auto* const __ptr = __local.__buffer.data();

      static_assert(::cuda::is_trivially_copyable_v<::cuda::std::remove_cvref_t<decltype(*__ptr)>>,
                    "Result values must be trivially copyable to be transportable across devices");

      // in-place, sendbuff == recvbuff + rank
      ::cuda::experimental::__nccl::__ncclAllGather(
        /*__sendbuff=*/__ptr + __comm.rank(),
        /*__recvbuff=*/__ptr,
        /*__sendcount=*/sizeof(*__ptr),
        ::cuda::experimental::__nccl::__ncclUint8,
        __comm.comm(),
        __local.__stream);
    }
  }

  return __partials;
}
} // namespace cuda::experimental::__detail

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_ALGORITHM_UTILS_H
