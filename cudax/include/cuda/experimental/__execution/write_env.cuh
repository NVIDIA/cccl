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

#include <cuda/__utility/immovable.h>

#include <cuda/experimental/__detail/utility.cuh>
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
struct _CCCL_TYPE_VISIBILITY_DEFAULT write_env_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Rcvr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    _Rcvr __rcvr_;
    _Env __env_;
  };

  template <class _Env, class... _RcvrEnv>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __env_ : env<__env_ref_t<_Env const&>, __fwd_env_t<_RcvrEnv>...>
  {
    using __base_t = env<__env_ref_t<_Env const&>, __fwd_env_t<_RcvrEnv>...>;

    _CCCL_API explicit constexpr __env_(_Env const& env, _RcvrEnv&&... __rcvr_env) noexcept
        : __base_t{__env_ref(env), __fwd_env(static_cast<_RcvrEnv&&>(__rcvr_env))...}
    {}

    using __base_t::query;

    // If _Env has a value for the get_scheduler_t query, then make sure we are not
    // delegating the get_domain_t query to the receiver's environment.
    _CCCL_TEMPLATE(class _Env2 = _Env)
    _CCCL_REQUIRES((!__queryable_with<_Env2, get_domain_t>) )
    [[nodiscard]] _CCCL_API constexpr auto query(get_domain_t) const noexcept
      -> __scheduler_domain_t<__scheduler_of_t<_Env2>, __fwd_env_t<_RcvrEnv>...>
    {
      return {};
    }
  };

  template <class _Env, class... _RcvrEnv>
  [[nodiscard]] _CCCL_API static constexpr auto __mk_env(const _Env& __env, _RcvrEnv&&... __rcvr_env) noexcept
  {
    return __env_{__env, static_cast<_RcvrEnv&&>(__rcvr_env)...};
  }

  template <class _Env, class... _RcvrEnv>
  using __env_t = decltype(__mk_env(::cuda::std::declval<_Env>(), ::cuda::std::declval<_RcvrEnv>()...));

  template <class _Rcvr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Ts>
    _CCCL_API constexpr void set_value(_Ts&&... __ts) noexcept
    {
      execution::set_value(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Ts&&>(__ts)...);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __err) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr_), static_cast<_Error&&>(__err));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr_));
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __env_t<_Env, env_of_t<_Rcvr>>
    {
      return __mk_env(__state_->__env_, execution::get_env(__state_->__rcvr_));
    }

    __state_t<_Rcvr, _Env>* __state_;
  };

  template <class _Rcvr, class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept = operation_state_t;

    _CCCL_API constexpr explicit __opstate_t(_Sndr&& __sndr, _Env __env, _Rcvr __rcvr)
        : __state_{static_cast<_Rcvr&&>(__rcvr), static_cast<_Env&&>(__env)}
        , __opstate_(execution::connect(static_cast<_Sndr&&>(__sndr), __rcvr_t<_Rcvr, _Env>{&__state_}))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate_);
    }

    __state_t<_Rcvr, _Env> __state_;
    connect_result_t<_Sndr, __rcvr_t<_Rcvr, _Env>> __opstate_;
  };

public:
  template <class _Sndr, class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Env>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __closure_t;

  /// @brief Wraps one sender in another that modifies the execution
  /// environment by merging in the environment specified.
  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr, _Env __env) const
  {
    // The write_env algorithm is not customizable by design; hence, we don't dispatch to
    // transform_sender like we do for other algorithms.
    return __sndr_t<_Sndr, _Env>{{}, static_cast<_Env&&>(__env), static_cast<_Sndr&&>(__sndr)};
  }

  /// @brief Returns a closure that can be used with the pipe operator
  /// to modify the execution environment.
  template <class _Env>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Env __env) const
  {
    return __closure_t<_Env>{static_cast<_Env&&>(__env)};
  }
};

template <class _Sndr, class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT write_env_t::__sndr_t
{
  using sender_concept = sender_t;

  template <class _Self, class... _RcvrEnv>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    using _Child _CCCL_NODEBUG_ALIAS = ::cuda::std::__copy_cvref_t<_Self, _Sndr>;
    return execution::get_completion_signatures<_Child, __env_t<_Env, _RcvrEnv...>>();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && -> __opstate_t<_Rcvr, _Sndr, _Env>
  {
    return __opstate_t<_Rcvr, _Sndr, _Env>{
      static_cast<_Sndr&&>(__sndr_), static_cast<_Env&&>(__env_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& -> __opstate_t<_Rcvr, const _Sndr&, _Env>
  {
    return __opstate_t<_Rcvr, const _Sndr&, _Env>{__sndr_, __env_, static_cast<_Rcvr&&>(__rcvr)};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Sndr>>
  {
    return __fwd_env(execution::get_env(__sndr_));
  }

  _CCCL_NO_UNIQUE_ADDRESS write_env_t __tag_;
  _Env __env_;
  _Sndr __sndr_;
};

template <class _Env>
struct _CCCL_TYPE_VISIBILITY_DEFAULT write_env_t::__closure_t
{
  template <class _Sndr>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(_Sndr __sndr) const -> __sndr_t<_Sndr, _Env>
  {
    return __sndr_t<_Sndr, _Env>{{}, static_cast<_Env&&>(__env_), static_cast<_Sndr&&>(__sndr)};
  }

  template <class _Sndr>
  [[nodiscard]] _CCCL_NODEBUG_API friend constexpr auto operator|(_Sndr __sndr, __closure_t __self)
    -> __sndr_t<_Sndr, _Env>
  {
    return __sndr_t<_Sndr, _Env>{{}, static_cast<_Env&&>(__self.__env_), static_cast<_Sndr&&>(__sndr)};
  }

  _Env __env_;
};

template <class _Sndr, class _Env>
inline constexpr size_t structured_binding_size<write_env_t::__sndr_t<_Sndr, _Env>> = 3;

_CCCL_GLOBAL_CONSTANT write_env_t write_env{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_WRITE_ENV
