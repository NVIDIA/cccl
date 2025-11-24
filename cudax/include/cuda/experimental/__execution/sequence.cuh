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
namespace __detail
{
template <class _Attrs, class... _Env>
[[nodiscard]] _CCCL_API constexpr auto __mk_seq_env_next(const _Attrs& __attrs, const _Env&... __env) noexcept
{
  if constexpr (__callable<get_completion_scheduler_t<set_value_t>, const _Attrs&, const _Env&...>)
  {
    return __mk_sch_env(get_completion_scheduler<set_value_t>(__attrs, __env...), __env...);
  }
  else if constexpr (__callable<get_completion_domain_t<set_value_t>, const _Attrs&, const _Env&...>)
  {
    using __domain_t = __call_result_t<get_completion_domain_t<set_value_t>, const _Attrs&, const _Env&...>;
    return prop{get_domain, __domain_t{}};
  }
  else
  {
    return env{};
  }
}

template <class _Attrs, class... _Env>
using __seq_env_next_t = decltype(__detail::__mk_seq_env_next(declval<_Attrs>(), declval<_Env>()...));

//! @brief Given a completion tag type, an environment, and a pack of attributes objects
//! obtained from a sequence of senders, return the scheduler on which the final sender
//! would complete assuming each sender was started where the previous sender completed.
// template <class _Tag, class _Env, class _Attrs>
// [[nodiscard]] _CCCL_API constexpr auto __seq_compl_sch_for(const _Env& __env, const _Attrs& __attrs) noexcept
// {
//   return __call_or(get_completion_scheduler<_Tag>, __nil{}, __attrs, __env);
// }

// template <class _Tag, class _Env, class _Attrs0, class _Attrs1, class... _Attrs>
// [[nodiscard]] _CCCL_API constexpr auto __seq_compl_sch_for(
//   const _Env& __env, const _Attrs0& __attrs0, const _Attrs1& __attrs1, const _Attrs&... __attrs) noexcept
// {
//   return __seq_compl_sch_for<_Tag>(__detail::__mk_seq_env_next(__attrs0, __env), __attrs1, __attrs...);
//   if constexpr (__callable<get_completion_scheduler_t<set_value_t>, const _Attrs0&, const _Env&>)
//   {
//     return;
//   }
//   auto __env_next = __detail::__mk_seq_env_next(__attrs0, __env);
//   return;
// }
} // namespace __detail

struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t
{
  _CUDAX_SEMI_PRIVATE :
  template <class _Attrs, class... _Env>
  using __env2_t = __join_env_t<__detail::__seq_env_next_t<_Attrs, __fwd_env_t<_Env>...>, __fwd_env_t<_Env>...>;

  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_API static constexpr auto __mk_env2(const _Attrs& __attrs, const _Env&... __env) noexcept
    -> __env2_t<_Attrs, _Env...>
  {
    return __join_env(__detail::__mk_seq_env_next(__attrs, __fwd_env(__env)...), __fwd_env(__env)...);
  }

  template <class _Rcvr, class _Env2, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __state_t
  {
    _CCCL_API constexpr explicit __state_t(_Rcvr&& __rcvr, _Env2 __env, _Sndr2&& __sndr2)
        : __rcvr2_{static_cast<_Rcvr&&>(__rcvr), __env}
        , __opstate2_(execution::connect(static_cast<_Sndr2&&>(__sndr2), __ref_rcvr(__rcvr2_)))
    {}

    __rcvr_with_env_t<_Rcvr, _Env2> __rcvr2_;
    connect_result_t<_Sndr2, __rcvr_ref_t<__rcvr_with_env_t<_Rcvr, _Env2>>> __opstate2_;
  };

  template <class _Rcvr, class _Env2, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_t
  {
    using receiver_concept = receiver_t;

    template <class... _Values>
    _CCCL_API constexpr void set_value(_Values&&...) noexcept
    {
      execution::start(__state_->__opstate2_);
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

    __state_t<_Rcvr, _Env2, _Sndr2>* __state_;
  };

  template <class _Rcvr, class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __opstate_t
  {
    using operation_state_concept      = operation_state_t;
    using __env2_t _CCCL_NODEBUG_ALIAS = __detail::__seq_env_next_t<env_of_t<_Sndr1>, env_of_t<_Rcvr>>;

    // The moves from lvalues here is intentional:
    _CCCL_API constexpr __opstate_t(_Sndr1& __sndr1, _Sndr2& __sndr2, _Rcvr& __rcvr, __env2_t __env2)
        : __state_(static_cast<_Rcvr&&>(__rcvr), static_cast<__env2_t&&>(__env2), static_cast<_Sndr2&&>(__sndr2))
        , __opstate1_(execution::connect(static_cast<_Sndr1&&>(__sndr1), __rcvr_t<_Rcvr, __env2_t, _Sndr2>{&__state_}))
    {}

    _CCCL_API constexpr __opstate_t(_Sndr1&& __sndr1, _Sndr2&& __sndr2, _Rcvr&& __rcvr)
        : __opstate_t(__sndr1, __sndr2, __rcvr, __detail::__mk_seq_env_next(get_env(__sndr1), get_env(__rcvr)))
    {}

    _CCCL_IMMOVABLE(__opstate_t);

    _CCCL_API ~__opstate_t() {}

    _CCCL_API constexpr void start() noexcept
    {
      execution::start(__opstate1_);
    }

  private:
    __state_t<_Rcvr, __env2_t, _Sndr2> __state_;
    connect_result_t<_Sndr1, __rcvr_t<_Rcvr, __env2_t, _Sndr2>> __opstate1_;
  };

public:
  template <class _Sndr1, class _Sndr2>
  struct _CCCL_TYPE_VISIBILITY_DEFAULT __sndr_t;

  template <class _Sndr1, class _Sndr2>
  _CCCL_API constexpr auto operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const;
};

template <class _Sndr1, class _Sndr2>
struct _CCCL_TYPE_VISIBILITY_DEFAULT sequence_t::__sndr_t
{
  using sender_concept = sender_t;
  template <class... _Env>
  using __env2_t _CCCL_NODEBUG_ALIAS = sequence_t::__env2_t<env_of_t<_Sndr1>, _Env...>;

  template <class _Self, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto get_completion_signatures()
  {
    _CUDAX_LET_COMPLETIONS(auto(__completions1) = get_child_completion_signatures<_Self, _Sndr1, _Env...>())
    {
      _CUDAX_LET_COMPLETIONS(auto(__completions2) = get_child_completion_signatures<_Self, _Sndr2, __env2_t<_Env...>>())
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
    // If _Sndr2 has _SetTag completions but does not know its _SetTag completion scheduler,
    // then we cannot know it either. Delete the function to prevent its use.
    _CCCL_TEMPLATE(class _SetTag, class... _Env)
    _CCCL_REQUIRES(__has_completions_for<_Sndr2, _SetTag, __env2_t<_Env...>> _CCCL_AND(
      !__callable<get_completion_scheduler_t<_SetTag>, env_of_t<_Sndr2>, __env2_t<_Env...>>))
    _CCCL_API auto query(get_completion_scheduler_t<_SetTag>, const _Env&...) const = delete;

    // If _Sndr2 has _SetTag completions but does not know its _SetTag completion domain,
    // then we cannot know it either. Delete the function to prevent its use.
    _CCCL_TEMPLATE(class _SetTag, class... _Env)
    _CCCL_REQUIRES(__has_completions_for<_Sndr2, _SetTag, __env2_t<_Env...>> _CCCL_AND(
      !__callable<get_completion_domain_t<_SetTag>, env_of_t<_Sndr2>, __env2_t<_Env...>>))
    _CCCL_API auto query(get_completion_domain_t<_SetTag>, const _Env&...) const = delete;

    template <class... _Env>
    [[nodiscard]] _CCCL_API constexpr auto query(get_completion_behavior_t, const _Env&...) const noexcept
    {
      return (execution::min) (execution::get_completion_behavior<_Sndr1, __fwd_env_t<_Env>...>(),
                               execution::get_completion_behavior<_Sndr2, __env2_t<_Env...>>());
    }

    using __child_attrs_t = __join_env_t<env_of_t<_Sndr2>, env_of_t<_Sndr1>>;

    _CCCL_EXEC_CHECK_DISABLE
    _CCCL_TEMPLATE(class _Query, class... _Args)
    _CCCL_REQUIRES(__forwarding_query<_Query> _CCCL_AND __queryable_with<__child_attrs_t, _Query, _Args...>)
    [[nodiscard]] _CCCL_API constexpr auto query(_Query, _Args&&... __args) const
      noexcept(__nothrow_queryable_with<__child_attrs_t, _Query, _Args...>)
        -> __query_result_t<__child_attrs_t, _Query, _Args...>
    {
      auto&& __env = __join_env(execution::get_env(__self_->__sndr2_), execution::get_env(__self_->__sndr1_));
      return __env.query(_Query{}, static_cast<_Args&&>(__args)...);
    }

    __sndr_t const* __self_;
  };

  [[nodiscard]] _CCCL_API constexpr auto get_env() const noexcept -> __attrs_t
  {
    return {this};
  }

  /*_CCCL_NO_UNIQUE_ADDRESS*/ sequence_t __tag_;
  /*_CCCL_NO_UNIQUE_ADDRESS*/ ::cuda::std::__ignore_t __ign_;
  _Sndr1 __sndr1_;
  _Sndr2 __sndr2_;
};

template <class _Sndr1, class _Sndr2>
_CCCL_API constexpr auto sequence_t::operator()(_Sndr1 __sndr1, _Sndr2 __sndr2) const
{
  using __sndr_t _CCCL_NODEBUG_ALIAS = sequence_t::__sndr_t<_Sndr1, _Sndr2>;
  return __sndr_t{{}, {}, static_cast<_Sndr1&&>(__sndr1), static_cast<_Sndr2&&>(__sndr2)};
}

template <class _Sndr1, class _Sndr2>
inline constexpr size_t structured_binding_size<sequence_t::__sndr_t<_Sndr1, _Sndr2>> = 4;

_CCCL_GLOBAL_CONSTANT sequence_t sequence{};
} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_SEQUENCE
