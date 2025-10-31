//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_CONTINUES_ON
#define __CUDAX_EXECUTION_STREAM_CONTINUES_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/get_stream.h>
#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/__utility/immovable.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/schedule_from.cuh>
#include <cuda/experimental/__execution/stream/adaptor.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __stream
{
//! The customization of schedule_from, when transferring back to the CPU, involves
//! adapting the sender and receiver types.
//!
//! A schedule_from sender such as schedule_from(sndr), where sndr completes on the GPU,
//! needs to synchronize the CUDA stream to ensure that all queued GPU work is finished.
//! Only then can the schedule operation be safely invoked -- from the CPU.
//!
//! To effect this, schedule_from(sndr) is transformed into
//! schedule_from(SYNC-STREAM-ADAPTOR(sndr)), where SYNC-STREAM-ADAPTOR(sndr) is a
//! sender that does the following:
//!
//! 1. In connect (called on host): Connects sndr with a sink receiver that ignores values
//!    passed to it and simply returns. The sink receiver's completion operations are
//!    executed on device when the child sender completes.
//!
//! 2. In start (called on host): Starts the child sender, which launches kernels for the
//!    predecessor operations, and then synchronizes the CUDA stream to ensure all queued
//!    GPU work is finished. Then, it pulls the results from sndr's operation state and
//!    passes them to the receiver on the host.
struct __schedule_from_t
{
  // Transition from the GPU to the CPU domain
  template <class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Values>
    _CCCL_API constexpr void set_value(_Values&&...) noexcept
    {
      // no-op
    }

    _CCCL_API constexpr void set_error(::cuda::std::__ignore_t) noexcept
    {
      // no-op
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      // no-op
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__rcvr_));
    }

    _Rcvr& __rcvr_;
  };

  // This opstate will be stored in host memory.
  template <class _Sndr, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __env_t                 = __fwd_env_t<env_of_t<_Rcvr>>;

    _CCCL_API constexpr explicit __opstate_t(_Sndr&& __sndr, _Rcvr __rcvr)
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __stream_(__get_stream(__sndr, execution::get_env(__rcvr_)))
        , __opstate_(execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t<_Rcvr>{__rcvr_}))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      NV_IF_TARGET(NV_IS_HOST, ({ __host_start(); }), ({ __device_start(); }));
    }

    _CCCL_HOST_API void __host_start() noexcept
    {
      // This launches all predecessor kernels on the given stream
      execution::start(__opstate_);

      // Synchronize the CUDA stream to make sure all predecessor work has completed, and
      // the results are available in __opstate_.
      if (auto __status = ::cudaStreamSynchronize(__stream_.get()); __status != ::cudaSuccess)
      {
        execution::set_error(static_cast<_Rcvr&&>(__rcvr_), cudaError_t(__status));
      }
      else
      {
        // __opstate_ is an instance of __stream::__opstate_t, and it has a __set_results
        // member function that will pass the results to the receiver on the host. __rcvr_
        // is the receiver of the parent default schedule_from operation. That receiver
        // will then start the schedule operation on the host.
        __opstate_.__set_results(__rcvr_);
      }
    }

    [[noreturn]] _CCCL_DEVICE_API void __device_start() noexcept
    {
      _CCCL_ASSERT(false, "internal error: stream::schedule_from opstate started on device");
      ::cuda::std::terminate();

      // We do not want the following to be called, but we need these code paths to be
      // instantiated. Without this, the __device_start function in stream/adaptor.cuh
      // will not be instantiated, and the kernel launch in the adaptor's __host_start
      // function will fail.
      execution::start(__opstate_);
      __opstate_.__set_results(__rcvr_);
    }

    _Rcvr __rcvr_;
    stream_ref __stream_;
    connect_result_t<_Sndr, __rcvr_t<_Rcvr>> __opstate_;
  };

  template <class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t
  {
    using sender_concept = sender_t;

    template <class _Self, class... _Env>
    [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
    {
      return execution::get_child_completion_signatures<_Self, _Sndr, _Env...>();
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Sndr, _Rcvr>
    {
      return __opstate_t<_Sndr, _Rcvr>{static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr)};
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr>
    {
      return __opstate_t<const _Sndr&, _Rcvr>{__sndr_, static_cast<_Rcvr&&>(__rcvr)};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> env_of_t<_Sndr>
    {
      return execution::get_env(__sndr_);
    }

    // The use of __tag_t here instructs the stream_domain not to apply any further
    // transformations to this sender. See stream/domain.cuh.
    /*_CCCL_NO_UNIQUE_ADDRESS*/ __tag_t<schedule_from_t> __tag_;
    /*_CCCL_NO_UNIQUE_ADDRESS*/ ::cuda::std::__ignore_t __ignore_;
    _Sndr __sndr_;
  };

  template <class _Sndr>
  _CCCL_API static constexpr auto __mk_sndr(_Sndr&& __sndr)
  {
    return __sndr_t<_Sndr>{{}, {}, static_cast<_Sndr&&>(__sndr)};
  }

  // This function is called when a schedule_from sender, with a predecessor that completes
  // on the stream scheduler, is being connected. It wraps the child sender so that it
  // synchronizes the stream after launching the child.
  template <class _Sndr>
  [[nodiscard]] _CCCL_API auto operator()(set_value_t, _Sndr&& __sndr, ::cuda::std::__ignore_t) const
  {
    static_assert(sender_for<_Sndr, schedule_from_t>);
    [[maybe_unused]] auto& [__tag, __ign, __child] = __sndr;
    using __child_t                                = ::cuda::std::__copy_cvref_t<_Sndr, decltype(__child)>;

    if constexpr (::cuda::__is_specialization_of_v<decltype(__child), __sndr_t>)
    {
      return static_cast<_Sndr&&>(__sndr);
    }
    else
    {
      return execution::schedule_from(__mk_sndr(static_cast<__child_t&&>(__child)));
    }
  }
};
} // namespace __stream

template <>
struct stream_domain::__apply_t<schedule_from_t> : __stream::__schedule_from_t
{};

template <class _Sndr>
inline constexpr size_t structured_binding_size<__stream::__schedule_from_t::__sndr_t<_Sndr>> = 3;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_CONTINUES_ON
