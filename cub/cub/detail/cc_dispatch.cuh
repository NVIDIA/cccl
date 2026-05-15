// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

namespace detail
{
// makes a functor that gets the policy for CC from PolicySelector when called
template <typename PolicySelector, int CC>
struct policy_getter : PolicySelector
{
  _CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE constexpr auto operator()() const
  {
    return PolicySelector::operator()(::cuda::compute_capability{CC});
  }
};

// Device-only variant for kernel-side compile-time policy queries.
template <typename PolicySelector, int CC>
struct device_policy_getter : PolicySelector
{
  _CCCL_DEVICE_API _CCCL_FORCEINLINE constexpr auto operator()() const
  {
    return PolicySelector::operator()(::cuda::compute_capability{CC});
  }
};

#if !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
#  if _CCCL_STD_VER < 2020
template <typename CudaCcSeq, typename PolicySelector, size_t... Is>
struct lowest_cc_resolver;

// we keep the compile-time build up of the mapping table outside a template parameterized by a user-provided callable
template <int... CudaCcs, typename PolicySelector, size_t... Is>
struct lowest_cc_resolver<::cuda::std::integer_sequence<int, CudaCcs...>, PolicySelector, Is...>
{
  static_assert(sizeof...(CudaCcs) == sizeof...(Is));

  using policy_t = decltype(PolicySelector{}(::cuda::compute_capability{}));

  static constexpr ::cuda::compute_capability all_ccs[sizeof...(Is)]{::cuda::compute_capability{CudaCcs}...};

  // GCC 7 has issues reusing the constexpr array of tuning policies in find_lowest below (it loses the constexpr-ness)
#    if _CCCL_COMPILER(GCC, >=, 8)
  static constexpr policy_t all_policies[sizeof...(Is)]{PolicySelector{}(all_ccs[Is])...};
#    endif // _CCCL_COMPILER(GCC, <, 8)

  _CCCL_HOST_DEVICE_API static constexpr auto find_lowest(size_t i) -> ::cuda::compute_capability
  {
#    if _CCCL_COMPILER(GCC, >=, 8)
    const auto& policy = all_policies[i];
    while (i > 0 && policy == all_policies[i - 1])
#    else // _CCCL_COMPILER(GCC, >=, 8)
    const auto& policy = PolicySelector{}(all_ccs[i]);
    while (i > 0 && policy == PolicySelector{}(all_ccs[i - 1]))
#    endif // _CCCL_COMPILER(GCC, >=, 8)
    {
      --i;
    }
    return all_ccs[i];
  }

  static constexpr ::cuda::compute_capability lowest_cc_with_same_policy[sizeof...(Is)]{find_lowest(Is)...};
};
#  endif // if _CCCL_STD_VER < 2020

template <typename PolicySelector, typename FunctorT, size_t... Is>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_to_cc_list(
  PolicySelector policy_selector, ::cuda::compute_capability device_cc, FunctorT&& f, ::cuda::std::index_sequence<Is...>)
{
  constexpr auto all_ccs = ::cuda::__target_compute_capabilities();

  _CCCL_ASSERT(((device_cc == all_ccs[Is]) || ...),
               "device_cc must appear in the list of compute capabilities compiled for");

  cudaError_t e = cudaErrorInvalidDeviceFunction;
#  if _CCCL_STD_VER >= 2020
  // In C++20, we just create an integral_constant holding the policy, because policies are structural types in C++20.
  // This causes f to be only instantiated for each distinct policy, since the same policy for different arches results
  // in the same integral_constant type passed to f
  using policy_t = decltype(policy_selector(::cuda::compute_capability{}));
  (...,
   (device_cc == all_ccs[Is] ? (e = f(::cuda::std::integral_constant<policy_t, policy_selector(all_ccs[Is])>{}))
                             : cudaSuccess));
#  else // if _CCCL_STD_VER >= 2020
  // In C++17, we have to collapse architectures with the same policies ourselves, so we instantiate call_for_cc once
  // per policy on the lowest CC which produces the same policy
  using resolver_t =
    lowest_cc_resolver<::cuda::std::integer_sequence<int, all_ccs[Is].get()...>, PolicySelector, Is...>;
  (...,
   (device_cc == all_ccs[Is]
      ? (e = f(policy_getter<PolicySelector, resolver_t::lowest_cc_with_same_policy[Is].get()>{policy_selector}))
      : cudaSuccess));

#  endif // if _CCCL_STD_VER >= 2020
  return e;
}

//! Takes a policy hub and instantiates f with the minimum possible number of nullary functor types that return a policy
//! at compile-time (if possible), and then calls the appropriate instantiation based on a runtime GPU architecture.
//! Depending on the used compiler, C++ standard, and available macros, a different number of instantiations may be
//! produced.
template <typename PolicySelector, typename F>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
dispatch_compute_cap(PolicySelector policy_selector, ::cuda::compute_capability device_cc, F&& f)
{
  // when not using CCCL.C, policy_selector is empty since all information is contained in its type
  static_assert(::cuda::std::is_empty_v<PolicySelector>);
  return dispatch_to_cc_list(
    policy_selector,
    device_cc,
    ::cuda::std::forward<F>(f),
    ::cuda::std::make_index_sequence<::cuda::__target_compute_capabilities().size()>{});
}

#else // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)

// if we are compiling CCCL.C with runtime policies, we cannot query the policy hub at compile time
_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector, typename F>
_CCCL_HOST_DEVICE_API _CCCL_FORCEINLINE cudaError_t
dispatch_compute_cap(PolicySelector policy_selector, ::cuda::compute_capability device_cc, F&& f)
{
  return f([&] {
    return policy_selector(device_cc);
  });
}
#endif // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
} // namespace detail

CUB_NAMESPACE_END
