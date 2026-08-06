// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/allocator_aware_execution_policy.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/any_system_tag.h>
#include <thrust/system/cpp/detail/execution_policy.h>

#include <hpx/execution.hpp>

THRUST_NAMESPACE_BEGIN
namespace system::hpx
{
// put the canonical tag in the same ns as the backend's entry points
namespace detail
{
// this awkward sequence of definitions arise
// from the desire both for tag to derive
// from execution_policy and for execution_policy
// to convert to tag (when execution_policy is not
// an ancestor of tag)

// forward declaration of tag
struct tag;

// forward declaration of execution_policy
template <typename>
struct execution_policy;

// specialize execution_policy for tag
template <>
struct execution_policy<tag> : thrust::system::cpp::detail::execution_policy<tag>
{
  using tag_type = tag;
};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag>
{};

// allow conversion to tag when it is not a successor
template <typename Derived>
struct execution_policy : thrust::system::cpp::detail::execution_policy<Derived>
{
  using tag_type = tag;
  operator tag() const
  {
    return tag();
  }
};

////////////////////////////////////////////////////////////////////////
// Base execution policy

//! Base class template for HPX-backed Thrust execution policies that carry an executor and parameters.
//!
//! This policy provides `on(...)` and `with(..)` to rebind the underlying HPX executor and executor parameters.
//! Derived policies are expected to be of the form `Derived<Executor, Parameters>`.
template <template <class, class> typename Derived, typename Executor, typename Parameters, typename Category = void>
struct basic_execution_policy
    : execution_policy<Derived<Executor, Parameters>>
    , thrust::detail::allocator_aware_execution_policy<execution_policy>
{
private:
  using decayed_executor_type   = std::decay_t<Executor>;
  using decayed_parameters_type = std::decay_t<Parameters>;
  using derived_type            = Derived<Executor, Parameters>;

  constexpr derived_type& derived() noexcept
  {
    return static_cast<derived_type&>(*this);
  }
  constexpr derived_type const& derived() const noexcept
  {
    return static_cast<derived_type const&>(*this);
  }

public:
  // The type of the executor associated with this execution policy
  using executor_type = decayed_executor_type;

  // The type of the associated executor parameters object which is
  // associated with this execution policy
  using executor_parameters_type = decayed_parameters_type;

  // The category of the execution agents created by this execution
  // policy.
  using execution_category =
    std::conditional_t<std::is_void_v<Category>, ::hpx::traits::executor_execution_category_t<executor_type>, Category>;

  // Rebind the type of executor used by this execution policy. The
  // execution category of Executor shall not be weaker than that of
  // this execution policy
  template <typename Executor_, typename Parameters_>
  struct rebind
  {
    using type = Derived<Executor_, Parameters_>;
  };

  constexpr basic_execution_policy() = default;

  template <typename Executor_, typename Parameters_>
  constexpr basic_execution_policy(Executor_&& exec, Parameters_&& params)
      : exec_(std::forward<Executor_>(exec))
      , params_(std::forward<Parameters_>(params))
  {}

  template <typename Executor_,
            typename Parameters_,
            typename Category_,
            typename = std::enable_if_t<!std::is_same_v<Derived<Executor_, Parameters_>, Derived<Executor, Parameters>>
                                        && std::is_convertible_v<Executor_, Executor>
                                        && std::is_convertible_v<Parameters_, Parameters>>>
  explicit constexpr basic_execution_policy(
    basic_execution_policy<Derived, Executor_, Parameters_, Category_> const& rhs)
      : exec_(rhs.executor())
      , params_(rhs.parameters())
  {}

  template <typename Executor_,
            typename Parameters_,
            typename Category_,
            typename = std::enable_if_t<std::is_convertible_v<Executor_, Executor>
                                        && std::is_convertible_v<Parameters_, Parameters>>>
  basic_execution_policy& operator=(basic_execution_policy<Derived, Executor_, Parameters_, Category_> const& rhs)
  {
    exec_   = rhs.executor();
    params_ = rhs.parameters();
    return *this;
  }

  // Create a new derived execution policy from the given executor
  template <typename Executor_>
  constexpr decltype(auto) on(Executor_&& exec) const
  {
    static_assert(::hpx::traits::is_executor_any_v<std::decay_t<Executor_>>,
                  "::hpx::traits::is_executor_any_v<Executor>");

    return ::hpx::execution::experimental::create_rebound_policy(derived(), std::forward<Executor_>(exec), parameters());
  }

  // Create a new execution policy from the given execution parameters
  template <typename... Parameters_>
  constexpr decltype(auto) with(Parameters_&&... params) const
  {
    return ::hpx::execution::experimental::create_rebound_policy(
      derived(),
      executor(),
      ::hpx::execution::experimental::join_executor_parameters(std::forward<Parameters_>(params)...));
  }

public:
  // Return the associated executor object.
  executor_type& executor() noexcept
  {
    return exec_;
  }

  // Return the associated executor object.
  constexpr executor_type const& executor() const noexcept
  {
    return exec_;
  }

  // Return the associated executor parameters object.
  executor_parameters_type& parameters() noexcept
  {
    return params_;
  }

  // Return the associated executor parameters object.
  constexpr executor_parameters_type const& parameters() const noexcept
  {
    return params_;
  }

private:
  executor_type exec_;
  executor_parameters_type params_;
};

template <typename Derived>
auto to_hpx_execution_policy(const execution_policy<Derived>& exec [[maybe_unused]]) noexcept
{
  if constexpr (::hpx::is_execution_policy_v<Derived>)
  {
    return thrust::detail::derived_cast(exec);
  }
  else
  {
    return ::hpx::execution::par;
  }
}
} // namespace detail

//! \addtogroup execution_policies
//! \{

//! \p thrust::hpx::tag is a type representing Thrust's HPX backend system in C++'s type system.
//!
//! Iterators "tagged" with a type which is convertible to \p hpx::tag assert that they may be "dispatched" to algorithm
//! Implementations in the \p hpx system.
using thrust::system::hpx::detail::tag;

//! \p thrust::hpx::execution_policy is the base class for all Thrust parallel execution policies which are derived from
//! Thrust's HPX backend system
using thrust::system::hpx::detail::execution_policy;

//! \p thrust::hpx::par is the parallel execution policy associated with Thrust's HPX backend system.
//!
//! Instead of relying on implicit algorithm dispatch through iterator system tags, users may directly target Thrust's
//! HPX backend system by providing \p thrust::hpx::par as an algorithm parameter.
//!
//! Explicit dispatch can be useful in avoiding the introudcution of data copies into containers such as
//! \p thrust::hpx::vector (when available).

//! \}
} // namespace system::hpx

// alias items at top-level
namespace hpx
{
using thrust::system::hpx::execution_policy;
using thrust::system::hpx::tag;
} // namespace hpx
THRUST_NAMESPACE_END
