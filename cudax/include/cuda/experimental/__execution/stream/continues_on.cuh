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

#include "cuda/experimental/__execution/visit.cuh"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__stream/get_stream.h>
#include <cuda/std/__utility/forward_like.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/stream/adaptor.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
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
  template <class _CvSndr, class _Rcvr>
  struct __opstate_t
  {
    _CCCL_API explicit __opstate_t(_CvSndr&& __sndr, _Rcvr __rcvr, stream_ref __stream)
        : __stream_(__stream)
        , __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __opstate_(execution::connect(static_cast<_CvSndr&&>(__sndr), __ref_rcvr(__rcvr_)))
    {}

    _CCCL_HOST_API void start() noexcept
    {
      execution::start(__opstate_);
      if (auto __status = ::cudaStreamSynchronize(__stream_.get()); __status != ::cudaSuccess)
      {
        printf("stream continues_on failed to synchronize stream: (%d)\n", __status);
        execution::set_error(static_cast<_Rcvr&&>(__rcvr_), cudaError_t(__status));
      }
    }

    stream_ref __stream_;
    _Rcvr __rcvr_;
    connect_result_t<_CvSndr, __rcvr_ref_t<_Rcvr>> __opstate_;
  };

  struct __tag_t
  {};

  template <class _Sndr>
  struct __sndr_t
  {
    using sender_concept = sender_t;

    [[nodiscard]] _CCCL_API auto get_env() const noexcept -> env_of_t<_Sndr>
    {
      return execution::get_env(__sndr_);
    }

    template <class _Self, class... _Env>
    _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
    {
      return execution::get_child_completion_signatures<_Self, _Sndr, _Env...>();
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) &&
    {
      return __opstate_t<_Sndr, _Rcvr>{static_cast<_Sndr&&>(__sndr_), static_cast<_Rcvr&&>(__rcvr), __stream_};
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const&
    {
      return __opstate_t<const _Sndr&, _Rcvr>{__sndr_, static_cast<_Rcvr&&>(__rcvr), __stream_};
    }

    _CCCL_NO_UNIQUE_ADDRESS __tag_t __tag_;
    stream_ref __stream_;
    _Sndr __sndr_;
  };

  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_API auto operator()(_Sndr&& __sndr, const _Env& __env) const -> decltype(auto)
  {
    auto&& [__tag, __sched, __child] = static_cast<_Sndr&&>(__sndr);
    auto __thunk_sched               = get_delegation_scheduler(__env);
    auto __stream                    = get_stream(get_env(__child));

    // Insert an extra hop through the delegation scheduler (a run_loop being driven by sync_wait).
    auto __thunk    = execution::schedule_from(__thunk_sched, _CUDA_VSTD::forward_like<_Sndr>(__child));
    using __thunk_t = decltype(__thunk);

    return execution::schedule_from(__sched, __sndr_t<__thunk_t>{{}, __stream, static_cast<__thunk_t&&>(__thunk)});
  }
};

template <class _Sndr>
inline constexpr size_t structured_binding_size<stream_domain::__apply_t<continues_on_t>::__sndr_t<_Sndr>> = 3;
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_CONTINUES_ON
