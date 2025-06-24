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
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/stream/adaptor.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__launch/launch.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// Transition from the GPU to the CPU domain
template <>
struct stream_domain::__apply_t<continues_on_t>
{
  // This opstate will be stored in host memory.
  template <class _Sndr, class _Rcvr>
  struct __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using __env_t                 = __fwd_env_t<env_of_t<_Rcvr>>;

    _CCCL_API constexpr explicit __opstate_t(_Sndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __stream_(__stream)
        , __opstate_(execution::connect(static_cast<_Sndr&&>(__sndr), __ref_rcvr(*this)))
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_HOST_API void start() noexcept
    {
      execution::start(__opstate_);
      if (auto __status = ::cudaStreamSynchronize(__stream_.get()); __status != ::cudaSuccess)
      {
        execution::set_error(static_cast<_Rcvr&&>(__rcvr_), cudaError_t(__status));
      }
      else
      {
        // __opstate_ is an instance of __stream::__opstate_t, and it has a __set_results
        // member function that will pass the results to the receiver on the host.
        __opstate_.__set_results(__rcvr_);
      }
    }

    template <class... _Values>
    _CCCL_API constexpr void set_value(_Values&&...) noexcept
    {
      // no-op
    }

    _CCCL_API constexpr void set_error(_CUDA_VSTD::__ignore_t) noexcept
    {
      // no-op
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      // no-op
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __env_t
    {
      return __fwd_env(execution::get_env(__rcvr_));
    }

    _Rcvr __rcvr_;
    stream_ref __stream_;
    connect_result_t<_Sndr, __rcvr_ref_t<__opstate_t, __env_t>> __opstate_;
  };

  struct __thunk_t
  {};

  template <class _Sndr>
  struct __sndr_t
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
      return __opstate_t<_Sndr, _Rcvr>{static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), __stream_};
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<const _Sndr&, _Rcvr>
    {
      return __opstate_t<const _Sndr&, _Rcvr>{__sndr_, static_cast<_Rcvr&&>(__rcvr), __stream_};
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> env_of_t<_Sndr>
    {
      return execution::get_env(__sndr_);
    }

    _CCCL_NO_UNIQUE_ADDRESS __thunk_t __tag_;
    stream_ref __stream_;
    _Sndr __sndr_;
  };

  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_API auto operator()(_Sndr&& __sndr, const _Env&) const -> decltype(auto)
  {
    auto& [__tag, __sched, __child] = __sndr;
    static_assert(__is_specialization_of_v<decltype(__child), __stream::__sndr_t>);
    using __child_t = _CUDA_VSTD::__copy_cvref_t<_Sndr, decltype(__child)>;

    auto __stream = get_stream(get_env(__child));
    return execution::schedule_from(__sched, __sndr_t<__child_t>{{}, __stream, static_cast<__child_t&&>(__child)});
  }
};

template <class _Sndr>
inline constexpr size_t structured_binding_size<stream_domain::__apply_t<continues_on_t>::__sndr_t<_Sndr>> = 3;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_CONTINUES_ON
