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

#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__tuple_dir/ignore.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__utility/undefined.h>

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

// _DomainOrTag: eg, default_domain or then_t
// _OpTag: either start_t or set_value_t
template <class _DomainOrTag, class _OpTag, class _Sndr, class... _Env>
using __transform_sender_result_t =
  decltype(declval<_DomainOrTag>().transform_sender(declval<_OpTag>(), declval<_Sndr>(), declval<const _Env&>()...));

template <class _DomainOrTag, class _OpTag, class _Sndr, class... _Env>
_CCCL_CONCEPT __has_transform_sender =
  __is_instantiable_with<__transform_sender_result_t, _DomainOrTag, _OpTag, _Sndr, _Env...>;

template <class _DomainOrTag, class _OpTag, class _Sndr, class... _Env>
_CCCL_CONCEPT __nothrow_transform_sender =
  _CCCL_REQUIRES_EXPR((_DomainOrTag, _OpTag, _Sndr, variadic _Env), __declfn_t<_Sndr> __sndr, const _Env&... __env) //
  ( //
    noexcept(_DomainOrTag{}.transform_sender(_OpTag{}, __sndr(), __env...)) //
  );

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
  _CCCL_API static constexpr auto apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(
    noexcept(_Tag{}.apply_sender(declval<_Sndr>(), declval<_Args>()...))) //
    -> __apply_sender_result_t<_Tag, _Sndr, _Args...>
  {
    return _Tag{}.apply_sender(static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }

  //! @brief Transforms a sender with an environment.
  //!
  //! @tparam _OpTag Either start_t or set_value_t.
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Env The type of the environment.
  //! @param __sndr The sender to be transformed.
  //! @param __env The environment used for the transformation.
  //! @return The result of transforming the sender with the given environment.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OpTag, class _Sndr, class _Env)
  _CCCL_REQUIRES(__has_transform_sender<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>)
  [[nodiscard]] _CCCL_API static constexpr auto transform_sender(_OpTag, _Sndr&& __sndr, const _Env& __env) //
    noexcept(__nothrow_transform_sender<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>)
      -> __transform_sender_result_t<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>
  {
    return tag_of_t<_Sndr>{}.transform_sender(_OpTag{}, static_cast<_Sndr&&>(__sndr), __env);
  }

  //! @overload
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Sndr>
  [[nodiscard]] _CCCL_API static constexpr auto
  transform_sender(::cuda::std::__ignore_t, _Sndr&& __sndr, ::cuda::std::__ignore_t) //
    noexcept(__nothrow_movable<_Sndr>) -> _Sndr
  {
    return static_cast<_Sndr&&>(__sndr);
  }
};

//! @brief Concept that checks whether a domain's sender transform behaves like that of
//! @c default_domain when passed the same arguments. The concept is modeled when either
//! of the following is
template <class _Domain, class _OpTag, class _Sndr, class _Env>
_CCCL_CONCEPT __default_domain_like =
  __same_as<decay_t<__transform_sender_result_t<default_domain, _OpTag, _Sndr, _Env>>,
            decay_t<::cuda::std::__type_call<
              ::cuda::std::__type_try_catch<
                ::cuda::std::__type_quote<__transform_sender_result_t>,
                ::cuda::std::__type_always<__transform_sender_result_t<default_domain, _OpTag, _Sndr, _Env>>>,
              _Domain,
              _OpTag,
              _Sndr,
              _Env>>>;

/**
 * @brief Tag type representing an indeterminate (unspecified) execution domain.
 *
 * This domain tag is used when a sender can complete with a given disposition
 * from multiple execution domains.
 *
 * @tparam _Domains...: the (possibly empty) set of domains that a sender's
 * completion may originate from.
 */
template <class... _Domains>
struct _CCCL_TYPE_VISIBILITY_DEFAULT indeterminate_domain
{
  _CCCL_HIDE_FROM_ABI indeterminate_domain() = default;

  _CCCL_API constexpr indeterminate_domain(::cuda::std::__ignore_t) noexcept {}

  //! @brief Transforms a sender with an optional environment.
  //!
  //! @tparam _OpTag Either start_t or set_value_t.
  //! @tparam _Sndr The type of the sender.
  //! @tparam _Env The type of the environment.
  //! @param __sndr The sender to be transformed.
  //! @param __env The environment used for the transformation.
  //! @return `default_domain{}.transform_sender(_OpTag{}, std::forward<_Sndr>(__sndr), __env)`
  //! @pre Every type in @c _Domains... must behave like @c default_domain when passed the
  //! same arguments. If this check fails, the @c static_assert triggers with: "ERROR:
  //! indeterminate domains: cannot pick an algorithm customization"
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _OpTag, class _Sndr, class _Env)
  _CCCL_REQUIRES(__has_transform_sender<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>)
  [[nodiscard]] _CCCL_API static constexpr auto transform_sender(_OpTag, _Sndr&& __sndr, const _Env& __env) //
    noexcept(__nothrow_transform_sender<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>)
      -> __transform_sender_result_t<tag_of_t<_Sndr>, _OpTag, _Sndr, _Env>
  {
    static_assert((__default_domain_like<_Domains, _OpTag, _Sndr, _Env> && ...),
                  "ERROR: indeterminate domains: cannot pick an algorithm customization");
    return tag_of_t<_Sndr>{}.transform_sender(_OpTag{}, static_cast<_Sndr&&>(__sndr), __env);
  }
};

//! @brief A wrapper around an environment that hides a set of queries.
template <class _Env, class... _Queries>
struct __hide_query
{
  static_assert(__nothrow_movable<_Env>);

  _CCCL_API explicit constexpr __hide_query(_Env&& __env, _Queries...) noexcept
      : __env_{static_cast<_Env&&>(__env)}
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TEMPLATE(class _Query, class... _As)
  _CCCL_REQUIRES(__none_of<_Query, _Queries...> _CCCL_AND __queryable_with<_Env, _Query, _As...>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(_Query __query, const _As&... __as) const
    noexcept(__nothrow_queryable_with<_Env, _Query, _As...>) -> __query_result_t<_Env, _Query, _As...>
  {
    return __env_.query(__query, __as...);
  }

private:
  _Env __env_;
};

template <class _Env>
struct __hide_scheduler : __hide_query<_Env, get_scheduler_t, get_domain_t>
{
  _CCCL_API explicit constexpr __hide_scheduler(_Env&& __env) noexcept
      : __hide_query<_Env, get_scheduler_t, get_domain_t>{static_cast<_Env&&>(__env), {}, {}}
  {}
};

template <class _Env>
_CCCL_HOST_DEVICE __hide_scheduler(_Env&&) -> __hide_scheduler<_Env>;

template <class _Sch, class... _Env>
using __scheduler_domain_t _CCCL_NODEBUG_ALIAS = __call_result_t<get_completion_domain_t<set_value_t>, _Sch, _Env...>;

//////////////////////////////////////////////////////////////////////////////////////////
//! @brief A query type for asking a receiver's environment for its domain, which is an
//! empty class type that is used in tag dispatching to find a custom implementation of a
//! sender algorithm. The result of this query is the "current" domain; that is, the domain
//! where `start` will be called on the operation state that results from connecting the
//! receiver to a sender.
struct get_domain_t
{
  //! @brief If there is a @c get_domain_t query in @c __env, return it.
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES(__queryable_with<_Env, get_domain_t>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env&) const noexcept
    -> decay_t<__query_result_t<_Env, get_domain_t>>
  {
    using __domain_t = decay_t<__query_result_t<_Env, get_domain_t>>;
    static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
    return __domain_t{};
  }

  //! @brief If there is not a @c get_domain_t query in @c __env, but there is a
  //! scheduler, return the domain of the scheduler if it has one, and @c default_domain
  //! otherwise.
  _CCCL_TEMPLATE(class _Env)
  _CCCL_REQUIRES((!__queryable_with<_Env, get_domain_t>) _CCCL_AND __callable<get_scheduler_t, const _Env&>)
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Env&) const noexcept
  {
    using __sch_t      = __scheduler_of_t<const _Env&>;
    using __env_t      = __hide_scheduler<const _Env&>; // to prevent recursion
    using __cmpl_sch_t = __call_result_or_t<get_completion_scheduler_t<set_value_t>, __sch_t, __sch_t, __env_t>;
    using __domain_t   = __scheduler_domain_t<__cmpl_sch_t, __env_t>;
    static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
    return __domain_t{};
  }

  //! @brief Fall back to the default domain if no other domain is found.
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::std::__ignore_t) const noexcept -> default_domain
  {
    return {};
  }

  _CCCL_API static constexpr auto query(forwarding_query_t) noexcept
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
    _CCCL_API constexpr auto operator()(const _Attrs&, cuda::std::__ignore_t = {}) const noexcept
    {
      return decay_t<__query_result_t<_Attrs, get_completion_domain_t>>{};
    }

    _CCCL_TEMPLATE(class _Attrs, class _Env)
    _CCCL_REQUIRES(__queryable_with<_Attrs, get_completion_domain_t, const _Env&>)
    _CCCL_API constexpr auto operator()(const _Attrs&, const _Env&) const noexcept
    {
      return decay_t<__query_result_t<_Attrs, get_completion_domain_t, const _Env&>>{};
    }
  };

private:
  template <class _Sch, class Domain, class... _Env>
  _CCCL_API static constexpr void __check_scheduler_domain() noexcept
  {
    static_assert(__same_as<Domain, __scheduler_domain_t<_Sch, const _Env&...>>,
                  "the sender's completion scheduler's domain does not match the domain returned by the scheduler");
  }

  template <class _Attrs, class... _Env, class _Domain>
  [[nodiscard]] _CCCL_TRIVIAL_API static _CCCL_CONSTEVAL auto __check_domain(_Domain) noexcept
  {
    // Sanity check: if a completion scheduler can be determined, then its domain must match
    // the domain returned by the attributes.
    if constexpr (__callable<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>)
    {
      using __sch_t = decay_t<__call_result_t<get_completion_scheduler_t<_Tag>, const _Attrs&, const _Env&...>>;
      if constexpr (!__same_as<__sch_t, _Attrs>) // prevent infinite recursion
      {
        get_completion_domain_t::__check_scheduler_domain<__sch_t, _Domain, _Env...>();
      }
    }
    return __declfn<_Domain>;
  }

  template <class _Attrs, class... _Env>
  [[nodiscard]] _CCCL_API static _CCCL_CONSTEVAL auto __get_declfn() noexcept
  {
    // If __attrs has a completion domain, then return it:
    if constexpr (__callable<__read_query_t, const _Attrs&, const _Env&...>)
    {
      using __domain_t = __call_result_t<__read_query_t, const _Attrs&, const _Env&...>;
      static_assert(__domain_like<__domain_t>, "Domain types are required to be empty class types");
      return __check_domain<_Attrs, _Env...>(__domain_t{});
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
        return __declfn<__domain_t>;
      }
      // Otherwise, if the scheduler's sender indicates that it completes inline, we can ask
      // the environment for its domain.
      else if constexpr (__completes_inline<env_of_t<schedule_result_t<__sch_t>>, _Env...>
                         && __callable<get_domain_t, const _Env&...>)
      {
        using __domain_t = __call_result_t<get_domain_t, const _Env&...>;
        return __declfn<__domain_t>;
      }
      // Otherwise, if we are asking "late" (with an environment), return the default_domain
      else if constexpr (sizeof...(_Env) != 0)
      {
        return __declfn<default_domain>;
      }
    }
    // Otherwise, if the attributes indicates that the sender completes inline, we can ask
    // the environment for its domain.
    else if constexpr (__completes_inline<_Attrs, _Env...> && __callable<get_domain_t, const _Env&...>)
    {
      using __domain_t = __call_result_t<get_domain_t, const _Env&...>;
      return __declfn<__domain_t>;
    }
    // Otherwise, if we are asking "late" (with an environment), return the default_domain
    else if constexpr (sizeof...(_Env) != 0)
    {
      return __declfn<default_domain>;
    }
    // Otherwise, no completion domain can be determined. Return void.
  }

public:
  template <class _Attrs, class... _Env, auto _DeclFn = __get_declfn<_Attrs, _Env...>()>
  [[nodiscard]] _CCCL_TRIVIAL_API constexpr auto operator()(const _Attrs&, const _Env&...) const noexcept
    -> decltype(_DeclFn())
  {
    return {};
  }

  [[nodiscard]] _CCCL_TRIVIAL_API static constexpr auto query(forwarding_query_t) noexcept -> bool
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

struct __not_a_domain
{
  _CCCL_HIDE_FROM_ABI __not_a_domain() = default;
  template <class _Domain>
  _CCCL_API constexpr __not_a_domain(_Domain&&) noexcept
  {}
};

template <class... _Domains>
using __indeterminate_domain_t =
  ::cuda::std::_If<sizeof...(_Domains) == 1, decltype((_Domains(), ...)), indeterminate_domain<_Domains...>>;

template <class _DomainSet>
using __domain_from_set_t =
  ::cuda::std::__type_apply<::cuda::std::_If<::cuda::std::__type_set_contains_v<_DomainSet, __not_a_domain>,
                                             ::cuda::std::__type_always<__not_a_domain>,
                                             ::cuda::std::__type_quote<__indeterminate_domain_t>>,
                            _DomainSet>;

template <class... _Domains>
using __make_domain_t = __domain_from_set_t<::cuda::std::__make_type_set<_Domains...>>;

// Common domain for a set of domains
template <class... _Domains>
struct __common_domain
{
  using type =
    ::cuda::std::__type_call<::cuda::std::__type_try_catch<::cuda::std::__type_quote<::cuda::std::common_type_t>,
                                                           ::cuda::std::__type_quote<__make_domain_t>>,
                             _Domains...>;
};

template <class... _Domains>
using __common_domain_t = typename __common_domain<_Domains...>::type;

namespace __detail
{
template <class _Tag, class _Sndr, class... _Env>
extern __call_result_or_t<get_completion_domain_t<_Tag>, indeterminate_domain<>, env_of_t<_Sndr>, _Env...>
  __compl_domain_v;

template <class _Tag, class _Sndr>
extern __call_result_or_t<get_completion_domain_t<_Tag>,
                          // If we ask for the completion domain early (without an env)
                          // and it cannot be determined, then:
                          // - if the sender knows it can never complete with _Tag, return
                          //   indeterminate_domain<>
                          // - otherwise, return __not_a_domain (indicating that the
                          //   completion domain may only be knowable later, when an env
                          //   is available)
                          ::cuda::std::_If<__never_completes_with<_Sndr, _Tag>, indeterminate_domain<>, __not_a_domain>,
                          env_of_t<_Sndr>>
  __compl_domain_v<_Tag, _Sndr>;
} // namespace __detail

template <class _Tag, class _Sndr, class... _Env>
using __compl_domain_t = decltype(__detail::__compl_domain_v<_Tag, _Sndr, _Env...>);
} // namespace cuda::experimental::execution

// Specializations of cuda::std::common_type for execution::indeterminate_domain
_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Ds, class _Domain>
struct common_type<::cuda::experimental::execution::indeterminate_domain<_Ds...>, _Domain>
{
  using type = ::cuda::experimental::execution::__make_domain_t<_Ds..., _Domain>;
};

template <class _Domain, class... _Ds>
struct common_type<_Domain, ::cuda::experimental::execution::indeterminate_domain<_Ds...>>
{
  using type = ::cuda::experimental::execution::__make_domain_t<_Ds..., _Domain>;
};

template <class... _As, class... _Bs>
struct common_type<::cuda::experimental::execution::indeterminate_domain<_As...>,
                   ::cuda::experimental::execution::indeterminate_domain<_Bs...>>
{
  using type = ::cuda::experimental::execution::__make_domain_t<_As..., _Bs...>;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_DOMAIN
