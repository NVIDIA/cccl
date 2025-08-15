//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_QUERIES
#define __CUDAX_EXECUTION_QUERIES

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_CCCL_SUPPRESS_DEPRECATED_PUSH
#include <cuda/std/__memory/allocator.h>
_CCCL_SUPPRESS_DEPRECATED_POP

#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__utility/rel_ops.h>
#include <cuda/std/__utility/unreachable.h>

#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/meta.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__launch/configuration.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
//////////////////////////////////////////////////////////////////////////////////////////
// get_allocator
_CCCL_GLOBAL_CONSTANT struct get_allocator_t
{
  template <class _Env>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_or_t<_Env, get_allocator_t, _CUDA_VSTD::allocator<void>>
  {
    static_assert(__nothrow_queryable_with_or<_Env, get_allocator_t, true>,
                  "The get_allocator query must be noexcept.");
    // NOT TO SPEC: return a default allocator if the query is not supported.
    return __query_or(__env, *this, _CUDA_VSTD::allocator<void>{});
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_allocator{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_stop_token
_CCCL_GLOBAL_CONSTANT struct get_stop_token_t
{
  template <class _Env>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_or_t<_Env, get_stop_token_t, never_stop_token>
  {
    static_assert(__nothrow_queryable_with_or<_Env, get_stop_token_t, true>,
                  "The get_stop_token query must be noexcept.");
    return __query_or(__env, *this, never_stop_token{});
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_stop_token{};

template <class _Ty>
using stop_token_of_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::decay_t<_CUDA_VSTD::__call_result_t<get_stop_token_t, _Ty>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_scheduler
template <class _Tag>
struct get_completion_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_completion_scheduler_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_completion_scheduler_t>
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

template <class _Tag>
extern _CUDA_VSTD::__undefined<_Tag> get_completion_scheduler;

// Explicitly instantiate these because of variable template weirdness in device code
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_value_t> get_completion_scheduler<set_value_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_error_t> get_completion_scheduler<set_error_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_stopped_t> get_completion_scheduler<set_stopped_t>{};

template <class _Env, class _Tag = set_value_t>
using __completion_scheduler_of_t _CCCL_NODEBUG_ALIAS =
  _CUDA_VSTD::decay_t<_CUDA_VSTD::__call_result_t<get_completion_scheduler_t<_Tag>, _Env>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_scheduler
_CCCL_GLOBAL_CONSTANT struct get_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_scheduler_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_scheduler_t>
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_scheduler{};

template <class _Env>
using __scheduler_of_t _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::decay_t<_CUDA_VSTD::__call_result_t<get_scheduler_t, _Env>>;

//////////////////////////////////////////////////////////////////////////////////////////
// get_delegation_scheduler
_CCCL_GLOBAL_CONSTANT struct get_delegation_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_delegation_scheduler_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_delegation_scheduler_t>
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<decltype(__env.query(*this))>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_delegation_scheduler{};

// This query is not a forwarding query.
_CCCL_GLOBAL_CONSTANT struct get_forward_progress_guarantee_t
{
  template <class _Sch>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Sch& __sch) const noexcept -> forward_progress_guarantee
  {
    static_assert(__nothrow_queryable_with_or<_Sch, get_forward_progress_guarantee_t, true>,
                  "The get_forward_progress_guarantee query must be noexcept.");
    return __query_or(__sch, *this, forward_progress_guarantee::weakly_parallel);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_forward_progress_guarantee{};

// By default, CUDA kernels are launched with a single thread and a single block.
using __single_threaded_config_base_t = decltype(experimental::make_config(grid_dims<1>(), block_dims<1>()));

// We hide the complicated type of the default launch configuration so diagnositics are
// easier to read.
struct __single_threaded_config_t : __single_threaded_config_base_t
{
  _CCCL_HOST_API constexpr __single_threaded_config_t() noexcept
      : __single_threaded_config_base_t{experimental::make_config(grid_dims<1>(), block_dims<1>())}
  {}
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_launch_config: A sender can define this attribute to control the launch configuration
// of the kernel it will launch when executed on a CUDA stream scheduler.
_CCCL_GLOBAL_CONSTANT struct get_launch_config_t
{
  template <class _Env>
  [[nodiscard]] _CCCL_HOST_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_or_t<_Env, get_launch_config_t, __single_threaded_config_t>
  {
    static_assert(__nothrow_queryable_with_or<_Env, get_launch_config_t, true>,
                  "The get_launch_config query must be noexcept.");
    return __query_or(__env, *this, __single_threaded_config_t{});
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_launch_config{};

namespace __completion_behavior
{
enum class _CCCL_TYPE_VISIBILITY_DEFAULT completion_behavior : int
{
  unknown, ///< The completion behavior is unknown.
  asynchronous, ///< The operation's completion will not happen on the calling thread before `start()`
                ///< returns.
  synchronous, ///< The operation's completion happens-before the return of `start()`.
  inline_completion ///< The operation completes synchronously within `start()` on the same thread that called
                    ///< `start()`.
};

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
[[nodiscard]] _CCCL_API constexpr auto operator<=>(completion_behavior a, completion_behavior b) noexcept
  -> _CUDA_VSTD::strong_ordering
{
  return static_cast<int>(a) <=> static_cast<int>(b);
}
#else
[[nodiscard]] _CCCL_API constexpr auto operator<(completion_behavior a, completion_behavior b) noexcept -> bool
{
  return static_cast<int>(a) < static_cast<int>(b);
}
[[nodiscard]] _CCCL_API constexpr auto operator==(completion_behavior a, completion_behavior b) noexcept -> bool
{
  return static_cast<int>(a) == static_cast<int>(b);
}
using namespace _CUDA_VSTD::rel_ops;
#endif
} // namespace __completion_behavior

struct _CCCL_TYPE_VISIBILITY_DEFAULT min_t;

struct completion_behavior
{
private:
  template <__completion_behavior::completion_behavior _CB>
  using __constant_t = _CUDA_VSTD::integral_constant<__completion_behavior::completion_behavior, _CB>;

  struct _CCCL_TYPE_VISIBILITY_DEFAULT unknown_t //
      : __constant_t<__completion_behavior::completion_behavior::unknown>
  {};
  struct _CCCL_TYPE_VISIBILITY_DEFAULT asynchronous_t //
      : __constant_t<__completion_behavior::completion_behavior::asynchronous>
  {};
  struct _CCCL_TYPE_VISIBILITY_DEFAULT synchronous_t //
      : __constant_t<__completion_behavior::completion_behavior::synchronous>
  {};
  struct _CCCL_TYPE_VISIBILITY_DEFAULT inline_completion_t //
      : __constant_t<__completion_behavior::completion_behavior::inline_completion>
  {};

  friend struct min_t;

public:
  static constexpr unknown_t unknown{};
  static constexpr asynchronous_t asynchronous{};
  static constexpr synchronous_t synchronous{};
  static constexpr inline_completion_t inline_completion{};
};

//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_behavior: A sender can define this attribute to describe the sender's
// completion behavior
struct get_completion_behavior_t
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(_CUDA_VSTD::__ignore_t, _CUDA_VSTD::__ignore_t = {}) const noexcept
  {
    return completion_behavior::unknown;
  }

  _CCCL_TEMPLATE(class _Attrs)
  _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_behavior_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, _CUDA_VSTD::__ignore_t = {}) const noexcept
  {
    static_assert(__nothrow_queryable_with<_Attrs, get_completion_behavior_t>,
                  "The get_completion_behavior query must be noexcept.");
    static_assert(_CUDA_VSTD::is_convertible_v<__query_result_t<_Attrs, get_completion_behavior_t>,
                                               __completion_behavior::completion_behavior>,
                  "The get_completion_behavior query must return one of the static member variables in "
                  "execution::completion_behavior.");
    return __attrs.query(*this);
  }

  _CCCL_TEMPLATE(class _Attrs, class _Env)
  _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_behavior_t, const _Env&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
  {
    static_assert(__nothrow_queryable_with<_Attrs, get_completion_behavior_t, const _Env&>,
                  "The get_completion_behavior query must be noexcept.");
    static_assert(_CUDA_VSTD::is_convertible_v<__query_result_t<_Attrs, get_completion_behavior_t, const _Env&>,
                                               __completion_behavior::completion_behavior>,
                  "The get_completion_behavior query must return one of the static member variables in "
                  "execution::completion_behavior.");
    return __attrs.query(*this, __env);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

struct _CCCL_TYPE_VISIBILITY_DEFAULT min_t
{
  template <__completion_behavior::completion_behavior... _CBs>
  [[nodiscard]] _CCCL_API constexpr auto operator()(completion_behavior::__constant_t<_CBs>...) const noexcept
  {
    constexpr auto __behavior = _CUDA_VSTD::min({_CBs...});
    if constexpr (__behavior == completion_behavior::unknown)
    {
      return completion_behavior::unknown;
    }
    else if constexpr (__behavior == completion_behavior::asynchronous)
    {
      return completion_behavior::asynchronous;
    }
    else if constexpr (__behavior == completion_behavior::synchronous)
    {
      return completion_behavior::synchronous;
    }
    else if constexpr (__behavior == completion_behavior::inline_completion)
    {
      return completion_behavior::inline_completion;
    }
    _CCCL_UNREACHABLE();
  }
};

_CCCL_GLOBAL_CONSTANT min_t min{};

template <class _Sndr, class... _Env>
[[nodiscard]] _CCCL_API constexpr auto get_completion_behavior() noexcept
{
  return _CUDA_VSTD::__call_result_t<get_completion_behavior_t, env_of_t<_Sndr>, _Env...>{};
}

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_QUERIES
