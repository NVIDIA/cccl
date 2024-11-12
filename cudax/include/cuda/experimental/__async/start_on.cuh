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

#include <cuda/experimental/__async/completion_signatures.cuh>
#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/rcvr_with_env.cuh>
#include <cuda/experimental/__async/tuple.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__async/variant.cuh>

#include <cuda/experimental/__async/prologue.cuh>

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

_CCCL_GLOBAL_CONSTANT struct start_on_t
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC

  template <class _Rcvr, class _Sch, class _CvSndr>
  struct __opstate_t
  {
    _CUDAX_API friend env_of_t<_Rcvr> get_env(const __opstate_t* __self) noexcept
    {
      return __async::get_env(__self->__env_rcvr_.__rcvr());
    }

    using operation_state_concept = operation_state_t;

    using completion_signatures = //
      transform_completion_signatures<
        completion_signatures_of_t<_CvSndr, __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>>*>,
        transform_completion_signatures<completion_signatures_of_t<schedule_result_t<_Sch>, __opstate_t*>,
                                        __async::completion_signatures<>,
                                        _CUDA_VSTD::__type_always<__async::completion_signatures<>>::__call>>;

    __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>> __env_rcvr_;
    connect_result_t<schedule_result_t<_Sch>, __opstate_t*> __opstate1_;
    connect_result_t<_CvSndr, __rcvr_with_env_t<_Rcvr, __sch_env_t<_Sch>>*> __opstate2_;

    _CUDAX_API __opstate_t(_Sch __sch, _Rcvr __rcvr, _CvSndr&& __sndr)
        : __env_rcvr_{static_cast<_Rcvr&&>(__rcvr), {__sch}}
        , __opstate1_{connect(schedule(__env_rcvr_.__env_.__sch_), this)}
        , __opstate2_{connect(static_cast<_CvSndr&&>(__sndr), &__env_rcvr_)}
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

    template <class _Error>
    _CUDAX_API void set_error(_Error&& __error) noexcept
    {
      __async::set_error(static_cast<_Rcvr&&>(__env_rcvr_.__rcvr()), static_cast<_Error&&>(__error));
    }

    _CUDAX_API void set_stopped() noexcept
    {
      __async::set_stopped(static_cast<_Rcvr&&>(__env_rcvr_.__rcvr()));
    }
  };

  template <class _Sch, class _Sndr>
  struct __sndr_t;

public:
  template <class _Sch, class _Sndr>
  _CUDAX_API auto operator()(_Sch __sch, _Sndr __sndr) const noexcept //
    -> __sndr_t<_Sch, _Sndr>;
} start_on{};

template <class _Sch, class _Sndr>
struct start_on_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS start_on_t __tag_;
  _Sch __sch_;
  _Sndr __sndr_;

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
_CUDAX_API auto start_on_t::operator()(_Sch __sch, _Sndr __sndr) const noexcept -> start_on_t::__sndr_t<_Sch, _Sndr>
{
  return __sndr_t<_Sch, _Sndr>{{}, __sch, __sndr};
}
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
