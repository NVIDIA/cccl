//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_START_ON
#define __CUDAX_ASYNC_DETAIL_START_ON

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/unreachable.h>

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/queries.cuh>
#include <cuda/experimental/__async/sender/rcvr_ref.cuh>
#include <cuda/experimental/__async/sender/rcvr_with_env.cuh>
#include <cuda/experimental/__async/sender/tuple.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>
#include <cuda/experimental/__async/sender/variant.cuh>
#include <cuda/experimental/__async/sender/visit.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
template <class _Sch>
struct __sch_env_t
{
  _Sch __sch_;

  _Sch query(get_scheduler_t) const noexcept
  {
    return __sch_;
  }
};

struct start_on_t
{
private:
  template <class _Rcvr, class _Sch, class _CvSndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t : __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>>
  {
    using operation_state_concept = operation_state_t;
    using __env_t                 = env_of_t<_Rcvr>;
    using __rcvr_with_sch_t       = __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>>;

    _CUDAX_API __opstate_t(_Sch __sch, _Rcvr __rcvr, _CvSndr&& __sndr)
        : __rcvr_with_sch_t{static_cast<_Rcvr&&>(__rcvr), {__sch}}
        , __opstate1_{connect(schedule(this->__env_.__sch_), __rcvr_ref<__opstate_t, __env_t>{*this})}
        , __opstate2_{connect(static_cast<_CvSndr&&>(__sndr), __rcvr_ref<__rcvr_with_sch_t>{*this})}
    {}

    _CUDAX_IMMOVABLE(__opstate_t);

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate1_);
    }

    _CUDAX_API void set_value() noexcept
    {
      __async::start(__opstate2_);
    }

    _CUDAX_API auto get_env() const noexcept -> __env_t
    {
      return __async::get_env(this->__base());
    }

    connect_result_t<schedule_result_t<_Sch&>, __rcvr_ref<__opstate_t, __env_t>> __opstate1_;
    connect_result_t<_CvSndr, __rcvr_ref<__rcvr_with_sch_t>> __opstate2_;
  };

public:
  template <class _Sch, class _Sndr>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sch, class _Sndr>
  _CUDAX_API auto operator()(_Sch __sch, _Sndr __sndr) const noexcept -> __sndr_t<_Sch, _Sndr>;
};

template <class _Sch, class _Sndr>
struct _CCCL_TYPE_VISIBILITY_DEFAULT start_on_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS start_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;

  template <class _Env>
  using __env_t = env<__sch_env_t<_Sch>, _FWD_ENV_T<_Env>>;

  template <class _Self, class... _Env>
  _CUDAX_API static constexpr auto get_completion_signatures()
  {
    using __sch_sndr   = schedule_result_t<_Sch>;
    using __child_sndr = __copy_cvref_t<_Self, _Sndr>;
    _CUDAX_LET_COMPLETIONS(
      auto(__sndr_completions) = __async::get_completion_signatures<__child_sndr, __env_t<_Env>...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__sch_completions) = __async::get_completion_signatures<__sch_sndr, _FWD_ENV_T<_Env>...>())
      {
        return __sndr_completions + transform_completion_signatures(__sch_completions, __swallow_transform());
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sch, _Sndr>
  {
    return __opstate_t<_Rcvr, _Sch, _Sndr>{__sch_, static_cast<_Rcvr&&>(__rcvr), static_cast<_Sndr&&>(__sndr_)};
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, _Sch, const _Sndr&>
  {
    return __opstate_t<_Rcvr, _Sch, const _Sndr&>{__sch_, static_cast<_Rcvr&&>(__rcvr), __sndr_};
  }

  _CUDAX_API env_of_t<_Sndr> get_env() const noexcept
  {
    return __async::get_env(__sndr_);
  }
};

template <class _Sch, class _Sndr>
_CUDAX_API auto start_on_t::operator()(_Sch __sch, _Sndr __sndr) const noexcept -> __sndr_t<_Sch, _Sndr>
{
  return __sndr_t<_Sch, _Sndr>{{}, __sch, __sndr};
}

template <class _Sch, class _Sndr>
inline constexpr size_t structured_binding_size<start_on_t::__sndr_t<_Sch, _Sndr>> = 3;

_CCCL_GLOBAL_CONSTANT start_on_t start_on{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
