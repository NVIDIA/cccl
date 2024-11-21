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

#include <cuda/experimental/__async/cpos.cuh>
#include <cuda/experimental/__async/env.cuh>
#include <cuda/experimental/__async/exception.cuh>
#include <cuda/experimental/__async/queries.cuh>
#include <cuda/experimental/__async/rcvr_with_env.cuh>
#include <cuda/experimental/__async/utility.cuh>
#include <cuda/experimental/__detail/config.cuh>

#include <cuda/experimental/__async/prologue.cuh>

namespace cuda::experimental::__async
{
struct write_env_t
{
#if !defined(_CCCL_CUDA_COMPILER_NVCC)

private:
#endif // _CCCL_CUDA_COMPILER_NVCC
  template <class _Rcvr, class _Sndr, class _Env>
  struct __opstate_t
  {
    using operation_state_concept = operation_state_t;
    using completion_signatures   = completion_signatures_of_t<_Sndr, __rcvr_with_env_t<_Rcvr, _Env>*>;

    __rcvr_with_env_t<_Rcvr, _Env> __env_rcvr_;
    connect_result_t<_Sndr, __rcvr_with_env_t<_Rcvr, _Env>*> __opstate_;

    _CUDAX_API explicit __opstate_t(_Sndr&& __sndr, _Env __env, _Rcvr __rcvr)
        : __env_rcvr_(static_cast<_Env&&>(__env), static_cast<_Rcvr&&>(__rcvr))
        , __opstate_(__async::connect(static_cast<_Sndr&&>(__sndr), &__env_rcvr_))
    {}

    _CUDAX_IMMOVABLE(__opstate_t);

    _CUDAX_API void start() noexcept
    {
      __async::start(__opstate_);
    }
  };

  template <class _Sndr, class _Env>
  struct __sndr_t;

public:
  /// @brief Wraps one sender in another that modifies the execution
  /// environment by merging in the environment specified.
  template <class _Sndr, class _Env>
  _CUDAX_TRIVIAL_API constexpr auto operator()(_Sndr, _Env) const //
    -> __sndr_t<_Sndr, _Env>;
};

template <class _Sndr, class _Env>
struct write_env_t::__sndr_t
{
  using sender_concept = sender_t;
  _CCCL_NO_UNIQUE_ADDRESS write_env_t __tag_;
  _Env __env_;
  _Sndr __sndr_;

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sndr, _Env>
  {
    return __opstate_t<_Rcvr, _Sndr, _Env>{
      static_cast<_Sndr&&>(__sndr_), static_cast<_Env&&>(__env_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CUDAX_API auto connect(_Rcvr __rcvr) const& //
    -> __opstate_t<_Rcvr, const _Sndr&, _Env>
  {
    return __opstate_t<_Rcvr, const _Sndr&, _Env>{__sndr_, __env_, static_cast<_Rcvr&&>(__rcvr)};
  }

  _CUDAX_API env_of_t<_Sndr> get_env() const noexcept
  {
    return __async::get_env(__sndr_);
  }
};

template <class _Sndr, class _Env>
_CUDAX_TRIVIAL_API constexpr auto write_env_t::operator()(_Sndr __sndr, _Env __env) const //
  -> write_env_t::__sndr_t<_Sndr, _Env>
{
  return write_env_t::__sndr_t<_Sndr, _Env>{{}, static_cast<_Env&&>(__env), static_cast<_Sndr&&>(__sndr)};
}

_CCCL_GLOBAL_CONSTANT write_env_t write_env{};

} // namespace cuda::experimental::__async

#include <cuda/experimental/__async/epilogue.cuh>

#endif
