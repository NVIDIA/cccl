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

#include <cuda/std/__execution/env.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__utility/exchange.h>
#include <cuda/std/__utility/unreachable.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__execution/completion_behavior.cuh>
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
    -> __query_result_or_t<_Env, get_allocator_t, ::cuda::std::allocator<void>>
  {
    static_assert(__nothrow_queryable_with_or<_Env, get_allocator_t, true>,
                  "The get_allocator query must be noexcept.");
    // NOT TO SPEC: return a default allocator if the query is not supported.
    return __query_or(__env, *this, ::cuda::std::allocator<void>{});
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
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

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_stop_token{};

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
    static_assert(__is_scheduler<__query_result_t<_Env, get_scheduler_t>>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_scheduler{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_previous_scheduler
_CCCL_GLOBAL_CONSTANT struct get_previous_scheduler_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_previous_scheduler_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env& __env) const noexcept
    -> __query_result_t<_Env, get_previous_scheduler_t>
  {
    static_assert(noexcept(__env.query(*this)));
    static_assert(__is_scheduler<__query_result_t<_Env, get_previous_scheduler_t>>);
    return __env.query(*this);
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_previous_scheduler{};

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

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_delegation_scheduler{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_completion_scheduler

//! @brief A query type for asking a sender's attributes for the scheduler on which that
//! sender will complete.
//!
//! @tparam _Tag one of set_value_t, set_error_t, or set_stopped_t
template <class _Tag>
struct get_completion_scheduler_t
{
  // This function object reads the completion scheduler from an attribute object or a
  // scheduler, accounting for the fact that the query member function may or may not
  // accept an environment.
  struct __read_query_t
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class _Attrs, class _GetComplSch = get_completion_scheduler_t>
    [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, cuda::std::__ignore_t = {}) const noexcept
      -> __query_result_t<_Attrs, _GetComplSch>
    {
      static_assert(noexcept(__attrs.query(_GetComplSch{})));
      static_assert(__is_scheduler<decltype(__attrs.query(_GetComplSch{}))>,
                    "The get_completion_scheduler query must return a scheduler type.");
      return __attrs.query(_GetComplSch{});
    }

    _CCCL_EXEC_CHECK_DISABLE
    template <class _Attrs, class _Env, class _GetComplSch = get_completion_scheduler_t>
    [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
      -> __query_result_t<_Attrs, _GetComplSch, const _Env&>
    {
      static_assert(noexcept(__attrs.query(_GetComplSch{}, __env)));
      static_assert(__is_scheduler<decltype(__attrs.query(_GetComplSch{}, __env))>,
                    "The get_completion_scheduler query must return a scheduler type.");
      return __attrs.query(_GetComplSch{}, __env);
    }
  };

private:
  // A scheduler might have a completion scheduler different from itself; for example, an
  // inline_scheduler completes wherever the scheduler's sender is started. So we
  // recursively ask the scheduler for its completion scheduler until we find one whose
  // completion scheduler is equal to itself (or it doesn't have one).
  struct __recurse_query_t
  {
    _CCCL_EXEC_CHECK_DISABLE
    template <class _Self = __recurse_query_t, class _Sch, class... _Env>
    [[nodiscard]]
    _CCCL_API constexpr auto operator()(_Sch __sch, const _Env&... __env) const noexcept
    {
      // When determining where the scheduler's operations will complete, we query
      // for the completion scheduler of the value channel:
      using __read_query_t = typename get_completion_scheduler_t<set_value_t>::__read_query_t;

      if constexpr (__callable<__read_query_t, _Sch, const _Env&...>)
      {
        using __sch2_t = __call_result_t<__read_query_t, _Sch, const _Env&...>;
        if constexpr (__same_as<_Sch, __sch2_t>)
        {
          _Sch __prev = __sch;
          do
          {
            __prev = cuda::std::exchange(__sch, __read_query_t{}(__sch, __env...));
          } while (__prev != __sch);
          return __sch;
        }
        else
        {
          // New scheduler has different type. Recurse!
          return _Self{}(__read_query_t{}(__sch, __env...), __env...);
        }
      }
      else if constexpr (__callable<__read_query_t, env_of_t<schedule_result_t<_Sch>>, const _Env&...>)
      {
        // BUGBUG
        // _CCCL_ASSERT(__sch == __read_query_t{}(get_env(schedule_t{}(__sch)), __env...),
        _CCCL_ASSERT(__sch == __read_query_t{}(get_env(__sch.schedule()), __env...),
                     "the scheduler's sender must have a completion scheduler attribute equal to the scheduler that "
                     "provided it.");
        return __sch;
      }
    }
  };

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_API static constexpr auto __impl(const _Attrs& __attrs, const _Env&... __env) noexcept
  {
    // If __attrs has a completion scheduler, then return it (after checking the scheduler
    // for _its_ completion scheduler):
    if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>)
    {
      return __recurse_query_t{}(__read_query_t{}(__attrs, __env...), __env...);
    }
    // Otherwise, if __attrs indicates that its sender completes inline, then we can ask
    // the environment for the current scheduler and return that (after checking the
    // scheduler for _its_ completion scheduler).
    else if constexpr (__completes_inline<_Attrs, _Env...> && __callable<get_scheduler_t, const _Env&...>)
    {
      return __recurse_query_t{}(get_scheduler(__env...), __detail::__hide_scheduler{__env}...);
    }
    // Otherwise, no completion scheduler can be determined. Return void.
  }

  template <class _Attrs, class... _Env>
  using __result_t = __unless_one_of_t<decltype(__impl(declval<_Attrs>(), declval<_Env>()...)), void>;

public:
  using __tag_t = _Tag;

  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Attrs& __attrs, const _Env&... __env) const noexcept
    -> __result_t<const _Attrs&, const _Env&...>
  {
    return __impl(__attrs, __env...);
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

template <class _Tag>
extern ::cuda::std::__undefined<_Tag> get_completion_scheduler;

// Explicitly instantiate these because of variable template weirdness in device code
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_value_t> get_completion_scheduler<set_value_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_error_t> get_completion_scheduler<set_error_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_scheduler_t<set_stopped_t> get_completion_scheduler<set_stopped_t>{};

//////////////////////////////////////////////////////////////////////////////////////////
// get_forward_progress_guarantee

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

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
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

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
} get_launch_config{};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_QUERIES
