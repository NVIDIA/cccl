//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__EXECUTION_STREAM_CONTINUES_ON
#define __CUDAX__EXECUTION_STREAM_CONTINUES_ON

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

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/stream/domain.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda_runtime_api.h>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
// Transition from the GPU to the CPU domain
template <>
struct stream_domain::__apply_t<continues_on_t>
{
  // To get off of the GPU and onto a CPU scheduler, we first put a thunk in the
  // delegation scheduler (the run_loop within sync_wait). That thunk will then schedule
  // the continuation on the destination scheduler.
  template <class _DestinationSched, class _DelegationSched, class _Rcvr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __thunk_opstate_t
  {
    using __run_rcvr_t    = __rcvr_ref_t<__thunk_opstate_t, env_of_t<_Rcvr>>;
    using __run_opstate_t = connect_result_t<schedule_result_t<_DelegationSched>, __run_rcvr_t>;
    using __dst_opstate_t = connect_result_t<schedule_result_t<_DestinationSched>, __rcvr_ref_t<_Rcvr>>;

    _CCCL_API explicit __thunk_opstate_t(
      stream_ref __stream, _DestinationSched __dst_sched, _DelegationSched __run_sched, _Rcvr __rcvr)
        : __rcvr_(static_cast<_Rcvr&&>(__rcvr))
        , __stream_(__stream)
        , __dst_sched_(static_cast<_DestinationSched&&>(__dst_sched))
        , __opstate_()
    {
      __opstate_.__emplace_from(connect, schedule(__run_sched), execution::__ref_rcvr(*this));
    }

    _CCCL_API void start() noexcept
    {
      // This enqueues a task on the run_loop that sync_wait is driving. That thread will
      // execute the task on the CPU, which will then schedule the continuation on the
      // destination scheduler.
      execution::start(__opstate_.template __get<0>());
    }

    // This will be called from run_loop::run() in sync_wait() on the CPU:
    _CCCL_HOST_API void set_value() noexcept
    {
      _CUDAX_TRY( //
        ({ //
          // synchronize the stream to ensure the results are ready for the CPU to take over
          __stream_.sync();
          auto& __opstate = __opstate_.__emplace_from(connect, schedule(__dst_sched_), execution::__ref_rcvr(__rcvr_));
          execution::start(__opstate);
        }),
        _CUDAX_CATCH(...) //
        ({ //
          execution::set_error(static_cast<_Rcvr&&>(__rcvr_), ::std::current_exception());
        }));
    }

    template <class _Error>
    _CCCL_API void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__rcvr_));
    }

    _CCCL_API auto get_env() const noexcept -> env_of_t<_Rcvr>
    {
      return execution::get_env(__rcvr_);
    }

    _Rcvr __rcvr_;
    stream_ref __stream_;
    _DestinationSched __dst_sched_;
    __variant<__run_opstate_t, __dst_opstate_t> __opstate_;
  };

  struct __thunk_sndr_tag
  {};

  template <class _DestinationSched, class _DelegationSched>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __thunk_sndr_t
  {
    using sender_concept = sender_t;

    template <class _Rcvr>
    using __opstate_t = __thunk_opstate_t<_DestinationSched, _DelegationSched, _Rcvr>;

    struct __state_t
    {
      stream_ref __stream_;
      _DestinationSched __dst_sched_;
      _DelegationSched __run_sched_;
    };

    struct __attrs_t
    {
      [[nodiscard]] _CCCL_API auto query(get_completion_scheduler_t<set_value_t>) const noexcept
      {
        return __state_->__dst_sched_;
      }

      [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t<set_value_t>) noexcept -> default_domain
      {
        return {};
      }

      const __state_t* __state_;
    };

    template <class _Self, class... _Env>
    [[nodiscard]] _CCCL_API static constexpr auto get_completion_signatures() noexcept
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__run_completions) = execution::get_completion_signatures<schedule_result_t<_DelegationSched>, _Env...>())
      {
        _CUDAX_LET_COMPLETIONS(auto(__dst_completions) =
                                 execution::get_completion_signatures<schedule_result_t<_DestinationSched>, _Env...>())
        {
          return __run_completions + __dst_completions;
        }
      }
    }

    template <class _Rcvr>
    [[nodiscard]] _CCCL_API auto connect(_Rcvr __rcvr) const -> __opstate_t<_Rcvr>
    {
      return __opstate_t<_Rcvr>{
        __state_.__stream_, __state_.__dst_sched_, __state_.__run_sched_, static_cast<_Rcvr&&>(__rcvr)};
    }

    [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __attrs_t
    {
      return {&__state_};
    }

    _CCCL_NO_UNIQUE_ADDRESS __thunk_sndr_tag __tag_{};
    __state_t __state_;
  };

  template <class _DestinationSched, class _DelegationSched>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __thunk_sched_t
  {
    using scheduler_concept = scheduler_t;

    [[nodiscard]] _CCCL_API auto schedule() const noexcept
    {
      return __thunk_sndr_t<_DestinationSched, _DelegationSched>{{}, {__stream_, __dst_sched_, __run_sched_}};
    }

    [[nodiscard]] _CCCL_API static constexpr auto query(get_domain_t<set_value_t>) noexcept
    {
      return default_domain{};
    }

    [[nodiscard]] _CCCL_API friend auto operator==(const __thunk_sched_t& __lhs, const __thunk_sched_t& __rhs) noexcept
      -> bool
    {
      return __lhs.__dst_sched_ == __rhs.__dst_sched_ && __lhs.__run_sched_ == __rhs.__run_sched_;
    }

    [[nodiscard]] _CCCL_API friend auto operator!=(const __thunk_sched_t& __lhs, const __thunk_sched_t& __rhs) noexcept
      -> bool
    {
      return !(__lhs == __rhs);
    }

    stream_ref __stream_;
    _DestinationSched __dst_sched_;
    _DelegationSched __run_sched_;
  };

  template <class _DestinationSched, class _DelegationSched>
  _CCCL_HOST_DEVICE __thunk_sched_t(stream_ref, _DestinationSched, _DelegationSched)
    -> __thunk_sched_t<_DestinationSched, _DelegationSched>;

  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_API auto operator()(_Sndr&& __sndr, const _Env& __env) const -> decltype(auto)
  {
    auto&& [__tag, __sched, __child] = static_cast<_Sndr&&>(__sndr);
    auto __run_sched                 = get_delegation_scheduler(__env);
    auto __stream                    = get_stream(get_env(__child));
    return execution::schedule_from(
      __thunk_sched_t{__stream, __sched, __run_sched}, _CUDA_VSTD::forward_like<_Sndr>(__child));
  }
};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX__EXECUTION_STREAM_CONTINUES_ON
