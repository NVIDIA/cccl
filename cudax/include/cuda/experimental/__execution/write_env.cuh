//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_WRITE_ENV
#define __CUDAX_EXECUTION_WRITE_ENV

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/rcvr_with_env.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT __write_env_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Rcvr, class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept _CCCL_NODEBUG_ALIAS = operation_state_t;

    _CCCL_API explicit __opstate_t(_Sndr&& __sndr, _Env __env, _Rcvr __rcvr)
        : __env_rcvr_{static_cast<_Rcvr&&>(__rcvr), static_cast<_Env&&>(__env)}
        , __opstate_(execution::connect(static_cast<_Sndr&&>(__sndr), __ref_rcvr(__env_rcvr_)))
    {}

    _CCCL_IMMOVABLE_OPSTATE(__opstate_t);

    _CCCL_API void start() noexcept
    {
      execution::start(__opstate_);
    }

    __rcvr_with_env_t<_Rcvr, _Env> __env_rcvr_;
    connect_result_t<_Sndr, __rcvr_ref_t<__rcvr_with_env_t<_Rcvr, _Env>>> __opstate_;
  };

public:
  template <class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  /// @brief Wraps one sender in another that modifies the execution
  /// environment by merging in the environment specified.
  template <class _Sndr, class _Env>
  _CCCL_TRIVIAL_API constexpr auto operator()(_Sndr, _Env) const;
};

template <class _Sndr, class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __write_env_t::__sndr_t
{
  using sender_concept _CCCL_NODEBUG_ALIAS = sender_t;

  template <class _Self, class... _Env2>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    using _Child _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__copy_cvref_t<_Self, _Sndr>;
    return execution::get_completion_signatures<_Child, env<const _Env&, __fwd_env_t<_Env2>>...>();
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sndr, _Env>
  {
    return __opstate_t<_Rcvr, _Sndr, _Env>{
      static_cast<_Sndr&&>(__sndr_), static_cast<_Env&&>(__env_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  _CCCL_API auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, const _Sndr&, _Env>
  {
    return __opstate_t<_Rcvr, const _Sndr&, _Env>{__sndr_, __env_, static_cast<_Rcvr&&>(__rcvr)};
  }

  [[nodiscard]] _CCCL_API auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }

  _CCCL_NO_UNIQUE_ADDRESS __write_env_t __tag_;
  _Env __env_;
  _Sndr __sndr_;
};

template <class _Sndr, class _Env>
_CCCL_TRIVIAL_API constexpr auto __write_env_t::operator()(_Sndr __sndr, _Env __env) const
{
  // The write_env algorithm is not customizable by design; hence, we don't dispatch to
  // transform_sender like we do for other algorithms.
  return __sndr_t<_Sndr, _Env>{{}, static_cast<_Env&&>(__env), static_cast<_Sndr&&>(__sndr)};
}

template <class _Sndr, class _Env>
inline constexpr size_t structured_binding_size<__write_env_t::__sndr_t<_Sndr, _Env>> = 3;

_CCCL_GLOBAL_CONSTANT __write_env_t write_env{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_WRITE_ENV
