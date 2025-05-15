//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_STARTS_ON
#define __CUDAX_ASYNC_DETAIL_STARTS_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__utility/pod_tuple.h>

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/rcvr_with_env.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _Sch>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __sch_env_t
{
  _Sch __sch_;

  auto query(get_scheduler_t) const noexcept -> _Sch
  {
    return __sch_;
  }
};

struct starts_on_t
{
private:
  template <class _Rcvr, class _Sch, class _CvSndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>>
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;
    using __env_t _CCCL_NODEBUG_ALIAS                 = env_of_t<_Rcvr>;
    using __rcvr_with_sch_t _CCCL_NODEBUG_ALIAS       = __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>>;

    _CCCL_API __opstate_t(_Sch __sch, _Rcvr __rcvr, _CvSndr&& __sndr)
        : __rcvr_with_sch_t{static_cast<_Rcvr&&>(__rcvr), {__sch}}
        , __opstate1_{connect(schedule(this->__env_.__sch_), __rcvr_ref<__opstate_t, __env_t>{*this})}
        , __opstate2_{connect(static_cast<_CvSndr&&>(__sndr), __rcvr_ref<__rcvr_with_sch_t>{*this})}
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      execution::start(__opstate1_);
    }

    _CCCL_API void set_value() noexcept
    {
      execution::start(__opstate2_);
    }

    _CCCL_API auto get_env() const noexcept -> __env_t
    {
      return execution::get_env(this->__base());
    }

    connect_result_t<schedule_result_t<_Sch&>, __rcvr_ref<__opstate_t, __env_t>> __opstate1_;
    connect_result_t<_CvSndr, __rcvr_ref<__rcvr_with_sch_t>> __opstate2_;
  };

public:
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch, class _Sndr>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sch __sch, _Sndr __sndr) const;
};

template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT starts_on_t::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS starts_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;

  template <class _Env>
  using __env_t _CCCL_NODEBUG_ALIAS = env<__sch_env_t<_Sch>, _FWD_ENV_T<_Env>>;

  template <class _Self, class... _Env>
  _CCCL_API static constexpr auto get_completion_signatures()
  {
    using __sch_sndr _CCCL_NODEBUG_ALIAS   = schedule_result_t<_Sch>;
    using __child_sndr _CCCL_NODEBUG_ALIAS = __copy_cvref_t<_Self, _Sndr>;
    _CUDAX_LET_COMPLETIONS(
      auto(__sndr_completions) = execution::get_completion_signatures<__child_sndr, __env_t<_Env>...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__sch_completions) = execution::get_completion_signatures<__sch_sndr, _FWD_ENV_T<_Env>...>())
      {
        return __sndr_completions + transform_completion_signatures(__sch_completions, __swallow_transform());
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sch, _Sndr>
  {
    return __opstate_t<_Rcvr, _Sch, _Sndr>{__sch_, static_cast<_Rcvr&&>(__rcvr), static_cast<_Sndr&&>(__sndr_)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, _Sch, const _Sndr&>
  {
    return __opstate_t<_Rcvr, _Sch, const _Sndr&>{__sch_, static_cast<_Rcvr&&>(__rcvr), __sndr_};
  }

  _CCCL_API auto get_env() const noexcept -> env_of_t<_Sndr>
  {
    return execution::get_env(__sndr_);
  }
};

template <class _Sch, class _Sndr>
_CCCL_TRIVIAL_API constexpr auto starts_on_t::operator()(_Sch __sch, _Sndr __sndr) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS  = __domain_of_t<_Sch>; // see [exec.starts.on]
  using __sndr_t _CCCL_NODEBUG_ALIAS = starts_on_t::__sndr_t<_Sch, _Sndr>;
  return transform_sender(__dom_t{}, __sndr_t{{}, static_cast<_Sch&&>(__sch), static_cast<_Sndr&&>(__sndr)});
}

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<starts_on_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT starts_on_t starts_on{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif
