//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_DOMAIN
#define __CUDAX_EXECUTION_DOMAIN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__type_traits/is_specialization_of.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/compose.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/type_list.h>

#include <cuda/experimental/__detail/type_traits.cuh>
#include <cuda/experimental/__detail/utility.cuh>
#include <cuda/experimental/__execution/completion_behavior.cuh>
#include <cuda/experimental/__execution/fwd.cuh>
#include <cuda/experimental/__execution/utility.cuh>
#include <cuda/experimental/__execution/visit.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
template <class _DomainOrTag, class... _Args>
using __apply_sender_result_t _CCCL_NODEBUG_ALIAS = decltype(_DomainOrTag{}.apply_sender(declval<_Args>()...));

template <class _DomainOrTag, class _Sndr, class... _Env>
using __transform_sender_result_t =
  decltype(_DomainOrTag{}.transform_sender(declval<_Sndr>(), declval<const _Env&>()...));

template <class _Domain>
_CCCL_CONCEPT __domain_like =
  ::cuda::std::is_empty_v<_Domain> && //
  ::cuda::std::is_nothrow_default_constructible_v<_Domain> && //
  ::cuda::std::is_nothrow_copy_constructible_v<_Domain>;

//! @brief A structure that selects the default set of algorithm implementations for
//! senders.
//!
//! This structure defines static member functions to handle operations on senders, such
//! as applying and transforming them. It is designed to work with CUDA's experimental
//! execution framework.
//! @see https://eel.is/c++draft/exec.domain.default
struct _CCCL_TYPE_VISIBILITY_DEFAULT default_domain
{
  //! @brief Applies a sender operation using the specified tag and arguments.
  //!
  //! @tparam _Tag The tag type that defines the operation to be applied.
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Args Variadic template for additional arguments.
  //! @param _Tag The tag instance specifying the operation.
  //! @param __sndr The sender to which the operation is applied.
  //! @param __args Additional arguments for the operation.
  //! @return The result of applying the sender operation.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tag, class _Sndr, class... _Args>
  _CCCL_NODEBUG_API static constexpr auto apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(noexcept(
    _Tag{}.apply_sender(declval<_Sndr>(), declval<_Args>()...))) -> __apply_sender_result_t<_Tag, _Sndr, _Args...>
  {
    return _Tag{}.apply_sender(static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }

  //! @brief Transforms a sender with an optional environment.
  //!
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Env The type of the environment.
  //! @param __sndr The sender to be transformed.
  //! @param __env The environment used for the transformation.
  //! @return The result of transforming the sender with the given environment.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr, class _Env>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env& __env) noexcept(
    noexcept(tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env)))
    -> __transform_sender_result_t<tag_of_t<_Sndr>, _Sndr, _Env>
  {
    return tag_of_t<_Sndr>{}.transform_sender(static_cast<_Sndr&&>(__sndr), __env);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto
  transform_sender(_Sndr&& __sndr) noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    // FUTURE TODO: add a transform for the split sender once we have a split sender
    return static_cast<_Sndr&&>(__sndr);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto
  transform_sender(_Sndr&& __sndr, ::cuda::std::__ignore_t) noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    return static_cast<_Sndr&&>(__sndr);
  }
};

namespace __detail
{
template <class _Env>
struct __hide_scheduler
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Query, class... _As)
  _CCCL_REQUIRES(__not_same_as<_Query, get_scheduler_t> _CCCL_AND __queryable_with<_Env, _Query, _As...>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Query __query, const _As&... __as) const
    noexcept(__nothrow_queryable_with<_Env, _Query, _As...>) -> decltype(auto)
  {
    return __env_.query(__query, __as...);
  }

  const _Env& __env_;
};

template <class _Env>
_CCCL_HOST_DEVICE __hide_scheduler(const _Env&) -> __hide_scheduler<_Env>;

template <class _Sch>
_CCCL_API auto __get_scheduler_domain() -> __call_result_t<get_completion_domain_t<set_value_t>, _Sch>;

template <class _Sch, class _Env>
_CCCL_API auto __get_scheduler_domain()
  -> __call_result_or_t<get_completion_domain_t<set_value_t>, default_domain, _Sch, _Env>;

} // namespace __detail

template <class _Sch, class... _Env>
using __scheduler_domain_t _CCCL_NODEBUG_ALIAS = decay_t<decltype(__detail::__get_scheduler_domain<_Sch, _Env...>())>;

//////////////////////////////////////////////////////////////////////////////////////////
//! @brief A query type for asking an environment for its domain, which is an empty class
//! type that is used in tag dispatching to find a custom implementation of a sender
//! algorithm.
//!
//! * When used to query a receiver's environment, it returns the "current" domain, which
//!   is where `start` will be called on the operation state that results from connecting
//!   the receiver to a sender.
//! * When used to query a scheduler `sch`, it is equivalent to
//!   `get_domain(get_env(schedule(sch)))`.
struct get_domain_t
{
  //! @brief If there is a @c get_domain_t query in @c __env, return it.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_domain_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env&) const noexcept
    -> decay_t<__query_result_t<_Env, get_domain_t>>
  {
    using __domain_t = decay_t<__query_result_t<_Env, get_domain_t>>;
    static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
    return __domain_t{};
  }

  //! @brief If there is not a @c get_domain_t query in @c __env, but there is a
  //! scheduler, return the domain of the scheduler if it has one, and @c default_domain
  //! otherwise.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES((!__queryable_with<_Env, get_domain_t>) _CCCL_AND __callable<get_scheduler_t, const _Env&>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Env&) const noexcept
  {
    using __sch_t      = __scheduler_of_t<const _Env&>;
    using __env_t      = __detail::__hide_scheduler<const _Env&>; // to prevent recursion
    using __cmpl_sch_t = __call_result_or_t<get_completion_scheduler_t<set_value_t>, __sch_t, __sch_t, __env_t>;
    using __domain_t   = __scheduler_domain_t<__cmpl_sch_t, __env_t>;
    static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
    return __domain_t{};
  }

  _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept
  {
    return true;
  }
};

_CCCL_GLOBAL_CONSTANT get_domain_t get_domain{};

//////////////////////////////////////////////////////////////////////////////////////////
//! @brief A query type for asking a sender's attributes for the domain on which that
//! sender will complete. As with @c get_domain, it is used in tag dispatching to find a
//! custom implementation of a sender algorithm.
//!
//! @tparam _Tag one of set_value_t, set_error_t, or set_stopped_t
template <class _Tag>
struct get_completion_domain_t
{
  // This function object reads the completion domain from an attribute object or a
  // scheduler, accounting for the fact that the query member function may or may not
  // accept an environment.
  struct __read_query_t
  {
    _CCCL_TEMPLATE(class _Attrs)
    _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_domain_t>)
    _CCCL_API constexpr auto operator()(const _Attrs& __attrs, cuda::std::__ignore_t = {}) const noexcept
      -> decay_t<__query_result_t<_Attrs, get_completion_domain_t>>;

    _CCCL_TEMPLATE(class _Attrs, class _Env)
    _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_domain_t, const _Env&>)
    _CCCL_API constexpr auto operator()(const _Attrs& __attrs, const _Env& __env) const noexcept
      -> decay_t<__query_result_t<_Attrs, get_completion_domain_t, const _Env&>>;
  };

private:
  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_API static constexpr auto __impl(const _Attrs&, const _Env&...) noexcept
  {
    // If __attrs has a completion domain, then return it:
    if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>)
    {
      using __domain_t = __call_result_t<__read_query_t, const _Attrs&, const _Env&...>;
      static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
      return __domain_t{};
    }
    // Otherwise, if __attrs has a completion scheduler, we can ask that scheduler for its
    // completion domain.
    else if constexpr (__callable<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>)
    {
      using __sch_t        = __call_result_t<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>;
      using __read_query_t = typename get_completion_domain_t<set_value_t>::__read_query_t;

      if constexpr (__callable<__read_query_t, __sch_t, const _Env&...>)
      {
        using __domain_t = __call_result_t<__read_query_t, __sch_t, const _Env&...>;
        static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
        return __domain_t{};
      }
      // Otherwise, if the scheduler's sender indicates that it completes inline, we can ask
      // the environment for its domain.
      else if constexpr (__completes_inline<env_of_t<schedule_result_t<__sch_t>>, _Env...>
                         && __callable<get_domain_t, const _Env&...>)
      {
        return __call_result_t<get_domain_t, const _Env&...>{};
      }
      // Otherwise, if we are asking "late" (with an environment), return the default_domain
      else if constexpr (sizeof...(_Env) != 0)
      {
        return default_domain{};
      }
    }
    // Otherwise, if the attributes indicates that the sender completes inline, we can ask
    // the environment for its domain.
    else if constexpr (__completes_inline<_Attrs, _Env...> && __callable<get_domain_t, const _Env&...>)
    {
      return __call_result_t<get_domain_t, const _Env&...>{};
    }
    // Otherwise, no completion domain can be determined. Return void.
  }

  template <class _Attrs, class... _Env>
  using __result_t = __unless_one_of_t<decltype(__impl(declval<_Attrs>(), declval<_Env>()...)), void>;

public:
  using __tag_t = _Tag;

  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Attrs&, const _Env&...) const noexcept
    -> __result_t<const _Attrs&, const _Env&...>
  {
    return {};
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return true;
  }
};

template <class _Tag>
extern ::cuda::std::__undefined<_Tag> get_completion_domain;

// Explicitly instantiate these because of variable template weirdness in device code
template <>
_CCCL_GLOBAL_CONSTANT get_completion_domain_t<set_value_t> get_completion_domain<set_value_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_domain_t<set_error_t> get_completion_domain<set_error_t>{};
template <>
_CCCL_GLOBAL_CONSTANT get_completion_domain_t<set_stopped_t> get_completion_domain<set_stopped_t>{};

// Used by the schedule_from and continues_on senders
struct get_domain_override_t
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Attrs)
  _CCCL_REQUIRES(__queryable_with<_Attrs, get_domain_override_t>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Attrs&, ::cuda::std::__ignore_t = {}) const noexcept
    -> decay_t<__query_result_t<_Attrs, get_domain_override_t>>
  {
    return {};
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Attrs, class _Env)
  _CCCL_REQUIRES(__queryable_with<_Attrs, get_domain_override_t, const _Env&>)
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto operator()(const _Attrs&, const _Env&) const noexcept
    -> decay_t<__query_result_t<_Attrs, get_domain_override_t, const _Env&>>
  {
    return {};
  }

  [[nodiscard]] _CCCL_NODEBUG_API static constexpr auto query(forwarding_query_t) noexcept -> bool
  {
    return false;
  }
};

_CCCL_GLOBAL_CONSTANT get_domain_override_t get_domain_override{};

namespace __detail
{
template <class _Sndr, class _Env = void, class _Default = default_domain>
_CCCL_NODEBUG_API constexpr auto __get_domain_impl() noexcept
{
  if constexpr (::cuda::std::is_void_v<_Env>)
  {
    // We are asking for the "early" domain, so ask the sender's attributes for a
    // completion domain.
    return __call_result_or_t<get_completion_domain_t<set_value_t>, _Default, env_of_t<_Sndr>, env<>&>{};
  }
  else
  {
    // We are asking for the "late" domain, so ask the environment for a domain unless
    // the sender's attributes have a domain override.
    if constexpr (__callable<get_domain_override_t, env_of_t<_Sndr>, _Env&>)
    {
      return __call_result_t<get_domain_override_t, env_of_t<_Sndr>, _Env&>{};
    }
    else
    {
      return __call_result_or_t<get_domain_t, _Default, _Env&>{};
    }
  }
}
} // namespace __detail

template <class _Sndr, class... _EnvAndDefault>
using __domain_of_t _CCCL_NODEBUG_ALIAS = decltype(__detail::__get_domain_impl<_Sndr, _EnvAndDefault...>());

template <class _Sndr, class _Default = default_domain>
using __early_domain_of_t _CCCL_NODEBUG_ALIAS = __domain_of_t<_Sndr, void, _Default>;

template <class _Sndr, class _Env, class _Default = default_domain>
using __late_domain_of_t _CCCL_NODEBUG_ALIAS = __domain_of_t<_Sndr, _Env, _Default>;

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_DOMAIN
