//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_ASYNC_DETAIL_WRITE_ENV
#define __CUDAX_ASYNC_DETAIL_WRITE_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__async/sender/completion_signatures.cuh>
#include <cuda/experimental/__async/sender/cpos.cuh>
#include <cuda/experimental/__async/sender/env.cuh>
#include <cuda/experimental/__async/sender/exception.cuh>
#include <cuda/experimental/__async/sender/queries.cuh>
#include <cuda/experimental/__async/sender/rcvr_ref.cuh>
#include <cuda/experimental/__async/sender/rcvr_with_env.cuh>
#include <cuda/experimental/__async/sender/utility.cuh>
#include <cuda/experimental/__async/sender/visit.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/sender/prologue.cuh>

namespace cuda::experimental::__async
{
struct write_env_t
{
private:
  template <class _Rcvr, class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    __rcvr_with_env_t<_Rcvr, _Env> __env_rcvr_;
    connect_result_t<_Sndr, __rcvr_ref<__rcvr_with_env_t<_Rcvr, _Env>>> __opstate_;

    _CUDAX_API explicit __opstate_t(_Sndr&& __sndr, _Env __env, _Rcvr __rcvr)
        : __env_rcvr_{static_cast<_Rcvr&&>(__rcvr), static_cast<_Env&&>(__env)}
        , __opstate_(__async::connect(static_cast<_Sndr&&>(__sndr), __rcvr_ref{__env_rcvr_}))
    {}

    _CUDAX_IMMOVABLE(__opstate_t);

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate_);
    }
  };

public:
  template <class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  /// @brief Wraps one sender in another that modifies the execution
  /// environment by merging in the environment specified.
  template <class _Sndr, class _Env>
  _CUDAX_TRIVIAL_API constexpr auto operator()(_Sndr, _Env) const -> __sndr_t<_Sndr, _Env>;
};

template <class _Sndr, class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT write_env_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS write_env_t __tag_;
  _Env __env_;
  _Sndr __sndr_;

  template <class _Self, class... _Env2>
  _CUDAX_API static constexpr auto get_completion_signatures()
  {
    using _Child = __copy_cvref_t<_Self, _Sndr>;
    return __async::get_completion_signatures<_Child, env<const _Env&, _FWD_ENV_T<_Env2>>...>();
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sndr, _Env>
  {
    return __opstate_t<_Rcvr, _Sndr, _Env>{
      static_cast<_Sndr&&>(__sndr_), static_cast<_Env&&>(__env_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, const _Sndr&, _Env>
  {
    return __opstate_t<_Rcvr, const _Sndr&, _Env>{__sndr_, __env_, static_cast<_Rcvr&&>(__rcvr)};
  }

  _CUDAX_API auto get_env() const noexcept -> env_of_t<_Sndr>
  {
    return __async::get_env(__sndr_);
  }
};

template <class _Sndr, class _Env>
_CUDAX_TRIVIAL_API constexpr auto write_env_t::operator()(_Sndr __sndr, _Env __env) const -> __sndr_t<_Sndr, _Env>
{
  return __sndr_t<_Sndr, _Env>{{}, static_cast<_Env&&>(__env), static_cast<_Sndr&&>(__sndr)};
}

template <class _Sndr, class _Env>
inline constexpr size_t structured_binding_size<write_env_t::__sndr_t<_Sndr, _Env>> = 3;

_CCCL_GLOBAL_CONSTANT write_env_t write_env{};
} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/sender/epilogue.cuh>

#endif
