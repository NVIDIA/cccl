/*
 *  Copyright 2008-2025 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

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
namespace system
{
// put the canonical tag in the same ns as the backend's entry points
namespace hpx
{
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
{};

// tag's definition comes before the
// generic definition of execution_policy
struct tag : execution_policy<tag>
{};

// allow conversion to tag when it is not a successor
template <typename Derived>
struct execution_policy : thrust::system::cpp::detail::execution_policy<Derived>
{
  typedef tag tag_type;
  operator tag() const
  {
    return tag();
  }
};

////////////////////////////////////////////////////////////////////////
// Base execution policy
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
  //
  // \tparam Executor  The type of the executor to associate with this
  //                   execution policy.
  //
  // \param exec       [in] The executor to use for the execution of
  //                   the parallel algorithm the returned execution
  //                   policy is used with.
  //
  // \note Requires: ::hpx::traits::is_executor_v<Executor> is true
  //
  // \returns The new execution policy
  //
  template <typename Executor_>
  constexpr decltype(auto) on(Executor_&& exec) const
  {
    static_assert(::hpx::traits::is_executor_any_v<std::decay_t<Executor_>>,
                  "::hpx::traits::is_executor_any_v<Executor>");

    return ::hpx::execution::experimental::create_rebound_policy(derived(), std::forward<Executor_>(exec), parameters());
  }

  // Create a new execution policy from the given execution parameters
  //
  // \tparam Parameters  The type of the executor parameters to
  //                     associate with this execution policy.
  //
  // \param params       [in] The executor parameters to use for the
  //                     execution of the parallel algorithm the
  //                     returned execution policy is used with.
  //
  // \note Requires: all parameters are executor_parameters, different
  //                 parameter types can't be duplicated
  //
  // \returns The new execution policy
  //
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
auto to_hpx_execution_policy(const execution_policy<Derived>& exec) noexcept
{
  if constexpr (::hpx::is_execution_policy_v<Derived>)
  {
    return thrust::detail::derived_cast(exec);
  }
  else
  {
    (void) exec;
    return ::hpx::execution::par;
  }
}

} // namespace detail

// alias execution_policy and tag here
using thrust::system::hpx::detail::execution_policy;
using thrust::system::hpx::detail::tag;

} // namespace hpx
} // namespace system

// alias items at top-level
namespace hpx
{

using thrust::system::hpx::execution_policy;
using thrust::system::hpx::tag;

} // namespace hpx
THRUST_NAMESPACE_END
