//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_SEQUENCE
#define __CUDAX_EXECUTION_SEQUENCE

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/immovable.h>
#include <cuda/std/__cccl/unreachable.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_callable.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/exception.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/lazy.cuh>
#include <cuda/experimental/__execution/rcvr_ref.cuh>
#include <cuda/experimental/__execution/rcvr_with_env.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/variant.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_API static constexpr auto __mk_env2(_Attrs&& __attrs, _Env&&... __env) noexcept
  {
    if constexpr (__callable<get_completion_scheduler_t<set_value_t>, _Attrs&, _Env&...>)
    {
      return __mk_sch_env(get_completion_scheduler<set_value_t>(__attrs, __env...), __env...);
    }
    else if constexpr (__callable<get_completion_domain_t<set_value_t>, _Attrs&>)
    {
      return prop{get_domain, get_completion_domain<set_value_t>(__attrs)};
    }
    else
    {
      return env{};
    }
  }

  template <class _Attrs, class... _Env>
  using __env2_t = decltype(sequence_t::__mk_env2(declval<_Attrs>(), declval<_Env>()...));

  template <class _Rcvr, class _Env2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_base_t
  {
    __rcvr_with_env_t<_Rcvr, _Env2> __rcvr2_;
    void (*__start2_fn_)(__state_base_t*) noexcept;
  };

  template <class _Rcvr, class _Env2, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t : __state_base_t<_Rcvr, _Env2>
  {
    // TODO: __rcvr2_'s env should be wrapped in __fwd_env_t before it is used to
    // connect __sndr2.
    _CCCL_API constexpr explicit __state_t(_Rcvr&& __rcvr, _Env2 __env, _Sndr2&& __sndr2)
        : __state_base_t<_Rcvr, _Env2>{{static_cast<_Rcvr&&>(__rcvr), __env}, &__start2_fn}
        , __opstate2_(execution::connect(static_cast<_Sndr2&&>(__sndr2), __ref_rcvr(this->__rcvr2_)))
    {}

    _CCCL_API static constexpr void __start2_fn(__state_base_t<_Rcvr, _Env2>* __base) noexcept
    {
      auto* __state = static_cast<__state_t*>(__base);
      execution::start(__state->__opstate2_);
    }

  private:
    connect_result_t<_Sndr2, __rcvr_ref_t<__rcvr_with_env_t<_Rcvr, _Env2>>> __opstate2_;
  };

  template <class _Rcvr, class _Env2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Values>
    _CCCL_API constexpr void set_value(_Values&&...) noexcept
    {
      __state_->__start2_fn_(__state_);
    }

    template <class _Error>
    _CCCL_API constexpr void set_error(_Error&& __error) noexcept
    {
      execution::set_error(static_cast<_Rcvr&&>(__state_->__rcvr2_.__base()), static_cast<_Error&&>(__error));
    }

    _CCCL_API constexpr void set_stopped() noexcept
    {
      execution::set_stopped(static_cast<_Rcvr&&>(__state_->__rcvr2_.__base()));
    }

    [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __fwd_env_t<env_of_t<_Rcvr>>
    {
      return __fwd_env(execution::get_env(__state_->__rcvr2_.__base()));
    }

    __state_base_t<_Rcvr, _Env2>* __state_;
  };

  template <class _Rcvr, class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept      = operation_state_t;
    using __env2_t _CCCL_NODEBUG_ALIAS = sequence_t::__env2_t<env_of_t<_Sndr1>, env_of_t<_Rcvr>>;

    // The moves from lvalues here is intentional:
    _CCCL_API constexpr __opstate_t(_Sndr1& __sndr1, _Sndr2& __sndr2, _Rcvr& __rcvr, __env2_t __env2)
        : __state_(static_cast<_Rcvr&&>(__rcvr), static_cast<__env2_t&&>(__env2), static_cast<_Sndr2&&>(__sndr2))
        , __opstate1_(execution::connect(static_cast<_Sndr1&&>(__sndr1), __rcvr_t<_Rcvr, __env2_t>{&__state_}))
    {}

    _CCCL_API constexpr __opstate_t(_Sndr1&& __sndr1, _Sndr2&& __sndr2, _Rcvr&& __rcvr)
        : __opstate_t(__sndr1, __sndr2, __rcvr, sequence_t::__mk_env2(get_env(__sndr1), get_env(__rcvr)))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API ~__opstate_t() {}

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

  private:
    __state_t<_Rcvr, __env2_t, _Sndr2> __state_;
    connect_result_t<_Sndr1, __rcvr_t<_Rcvr, __env2_t>> __opstate1_;
  };

  template <class _SetTag, class _Attrs1, class _Sndr2, class... _Env>
  [[nodiscard]] _CCCL_API constexpr auto
  __get_completion_scheduler(const _Attrs1& __attrs1, const _Sndr2& __sndr2, _Env&&... __env) noexcept
  {
    using __env2_t _CCCL_NODEBUG_ALIAS = sequence_t::__env2_t<_Attrs1, _Env...>;

    if constexpr (__callable<get_completion_scheduler_t<_SetTag>, env_of_t<_Sndr2>, __join_env_t<__env2_t, _Env...>>)
    {
      // If the second sender has a completion scheduler for the given tag, use it.
      auto __env2 = sequence_t::__mk_env2(__attrs1, __env...);
      return get_completion_scheduler<_SetTag>(
        get_env(__sndr2), __join_env(static_cast<__env2_t&&>(__env2), static_cast<_Env&&>(__env)...));
    }
    else if constexpr (get_completion_signatures<_Sndr2, __join_env_t<__env2_t, _Env...>>().count(_SetTag{}) == 0)
    {
      // If the second sender does not have any _SetTag completions, use the first sender's
      // completion scheduler, if it has one:
      if constexpr (__callable<get_completion_scheduler_t<_SetTag>, _Attrs1, __fwd_env_t<_Env>...>)
      {
        return get_completion_scheduler<_SetTag>(__attrs1, __fwd_env(static_cast<_Env&&>(__env))...);
      }
    }
  }

  template <class _SetTag, class _Attrs1, class _Sndr2, class... _Env>
  [[nodiscard]] _CCCL_API constexpr auto
  __get_completion_domain(const _Attrs1& __attrs1, const _Sndr2& __sndr2, _Env&&... __env) noexcept
  {
    using __env2_t _CCCL_NODEBUG_ALIAS = sequence_t::__env2_t<_Attrs1, _Env...>;

    if constexpr (__callable<get_completion_domain_t<_SetTag>, env_of_t<_Sndr2>, __join_env_t<__env2_t, _Env...>>)
    {
      // If the second sender has a completion domain for the given tag, use it.
      return __call_result_t<get_completion_domain_t<_SetTag>, env_of_t<_Sndr2>, __join_env_t<__env2_t, _Env...>>{};
    }
    else if constexpr (get_completion_signatures<_Sndr2, __join_env_t<__env2_t, _Env...>>().count(_SetTag{}) == 0)
    {
      // If the second sender does not have any _SetTag completions, use the first
      // sender's completion domain, if it has one:
      if constexpr (__callable<get_completion_domain_t<_SetTag>, _Attrs1, __fwd_env_t<_Env>...>)
      {
        return __call_result_t<get_completion_domain_t<_SetTag>, _Attrs1, __fwd_env_t<_Env>...>{};
      }
    }
  }

public:
  template <class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sndr1, class _Sndr2>
  _CCCL_NODEBUG_API constexpr auto operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const;
};

template <class _Sndr1, class _Sndr2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t::__sndr_t
{
  using sender_concept               = sender_t;
  using __env2_t _CCCL_NODEBUG_ALIAS = sequence_t::__env2_t<env_of_t<_Sndr1>>;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__completions1) = get_child_completion_signatures<_Self, _Sndr1, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(
        auto(__completions2) = get_child_completion_signatures<_Self, _Sndr2, __join_env_t<__env2_t, _Env...>>())
      {
        // __swallow_transform to ignore the first sender's value completions
        return __completions2 + transform_completion_signatures(__completions1, __swallow_transform{});
      }
    }

    _CCCL_UNREACHABLE();
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) && //
    -> sequence_t::__opstate_t<_Rcvr, _Sndr1, _Sndr2>
  {
    using __opstate_t = sequence_t::__opstate_t<_Rcvr, _Sndr1, _Sndr2>;
    return __opstate_t{static_cast<_Sndr1&&>(__sndr1_), static_cast<_Sndr2>(__sndr2_), static_cast<_Rcvr&&>(__rcvr)};
  }

  template <class _Rcvr>
  [[nodiscard]] _CCCL_API constexpr auto connect(_Rcvr __rcvr) const& //
    -> sequence_t::__opstate_t<_Rcvr, const _Sndr1&, const _Sndr2&>
  {
    using __opstate_t = sequence_t::__opstate_t<_Rcvr, const _Sndr1&, const _Sndr2&>;
    return __opstate_t{__sndr1_, __sndr2_, static_cast<_Rcvr&&>(__rcvr)};
  }

  struct __attrs_t
  {
    // If the second sender does not have any _SetTag completions, we can look at the
    // first sender for a completion scheduler.
    _CCCL_TEMPLATE(class _SetTag, class... _Env, class _Env2 = sequence_t::__env2_t<env_of_t<_Sndr1>, _Env...>)
    _CCCL_REQUIRES((execution::get_completion_signatures<_Sndr2, __join_env_t<_Env2, _Env...>>().count(_SetTag{}) == 0))
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_scheduler_t<_SetTag>, _Env&&... __env) const noexcept
      -> __call_result_t<get_completion_scheduler_t<_SetTag>, env_of_t<_Sndr1>, __fwd_env_t<_Env>...>
    {
      return get_completion_scheduler<_SetTag>(
        execution::get_env(__self_->__sndr1_), __fwd_env(static_cast<_Env&&>(__env))...);
    }

    // If the second sender does not have any _SetTag completions, we can look at the
    // first sender for a completion domain.
    _CCCL_TEMPLATE(class _SetTag, class... _Env, class _Env2 = sequence_t::__env2_t<env_of_t<_Sndr1>, _Env...>)
    _CCCL_REQUIRES((execution::get_completion_signatures<_Sndr2, __join_env_t<_Env2, _Env...>>().count(_SetTag{}) == 0))
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_domain_t<_SetTag>, _Env&&... __env) const noexcept
      -> decay_t<__call_result_t<get_completion_domain_t<_SetTag>, env_of_t<_Sndr1>, __fwd_env_t<_Env>...>>
    {
      return {};
    }

    template <class... _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t, const _Env&...) const noexcept
    {
      using __env2_t _CCCL_NODEBUG_ALIAS = sequence_t::__env2_t<env_of_t<_Sndr1>, _Env...>;
      return (execution::min) (execution::get_completion_behavior<_Sndr1, __fwd_env_t<_Env>...>(),
                               execution::get_completion_behavior<_Sndr2, __join_env_t<__env2_t, _Env...>>());
    }

    // The following overload will not be considered when _Query is get_domain_override_t
    // because get_domain_override_t is not a forwarding query.
    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query, class... _Args)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<env_of_t<_Sndr2>, _Query, _Args...>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<env_of_t<_Sndr2>, _Query, _Args...>)
        -> __query_result_t<env_of_t<_Sndr2>, _Query, _Args...>
    {
      return execution::get_env(__self_->__sndr2_).query(_Query{}, static_cast<_Args&&>(__args)...);
    }

    __sndr_t const* __self_;
  };

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return {this};
  }

  _CCCL_NO_UNIQUE_ADDRESS sequence_t __tag_;
  _CCCL_NO_UNIQUE_ADDRESS ::cuda::std::__ignore_t __ign_;
  _Sndr1 __sndr1_;
  _Sndr2 __sndr2_;
};

template <class _Sndr1, class _Sndr2>
_CCCL_NODEBUG_API constexpr auto sequence_t::operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const
{
  using __dom_t _CCCL_NODEBUG_ALIAS  = __early_domain_of_t<_Sndr1>;
  using __sndr_t _CCCL_NODEBUG_ALIAS = sequence_t::__sndr_t<_Sndr1, _Sndr2>;
  return transform_sender(__dom_t{}, __sndr_t{{}, {}, static_cast<_Sndr1&&>(__sndr1), static_cast<_Sndr2&&>(__sndr2)});
}

template <class _Sndr1, class _Sndr2>
inline constexpr size_t structured_binding_size<sequence_t::__sndr_t<_Sndr1, _Sndr2>> = 4;

_CCCL_GLOBAL_CONSTANT sequence_t sequence{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SEQUENCE
