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

#include <cuda/__device/arch_id.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/array>

CUB_NAMESPACE_BEGIN

namespace detail
{
#if !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)

#  if _CCCL_STD_VER < 2020
template <typename PolicySelector, ::cuda::arch_id LowestArchId>
struct policy_getter_17
{
  PolicySelector policy_selector;

  _CCCL_API _CCCL_FORCEINLINE constexpr auto operator()() const
  {
    return policy_selector(LowestArchId);
  }
};

template <typename PolicySelector, size_t N>
_CCCL_API constexpr auto find_lowest_arch_with_same_policy(
  PolicySelector policy_selector, size_t i, const ::cuda::std::array<::cuda::arch_id, N>& all_arches) -> ::cuda::arch_id
{
  const auto policy = policy_selector(all_arches[i]);
  while (i > 0 && policy_selector(all_arches[i - 1]) == policy)
  {
    --i;
  }
  return all_arches[i];
}

template <int ArchMult, typename CudaArchSeq, typename PolicySelector, size_t... Is>
struct lowest_arch_resolver;

// we keep the compile-time build up of the mapping table outside a template parameterized by a user-provided callable
template <int ArchMult, int... CudaArches, typename PolicySelector, size_t... Is>
struct lowest_arch_resolver<ArchMult, ::cuda::std::integer_sequence<int, CudaArches...>, PolicySelector, Is...>
{
  static_assert(sizeof...(CudaArches) == sizeof...(Is));

  using policy_t = decltype(PolicySelector{}(::cuda::arch_id{}));

  static constexpr ::cuda::arch_id all_arches[sizeof...(Is)] = {::cuda::arch_id{(CudaArches * ArchMult) / 10}...};

  // GCC 7 has issues reusing the constexpr array of tuning policies in find_lowest below (it loses the constexpr-ness)
#    if _CCCL_COMPILER(GCC, >=, 8)
  static constexpr policy_t all_policies[sizeof...(Is)] = {PolicySelector{}(all_arches[Is])...};
#    endif // _CCCL_COMPILER(GCC, <, 8)

  _CCCL_API static constexpr auto find_lowest(size_t i) -> ::cuda::arch_id
  {
#    if _CCCL_COMPILER(GCC, >=, 8)
    const auto& policy = all_policies[i];
    while (i > 0 && policy == all_policies[i - 1])
#    else // _CCCL_COMPILER(GCC, >=, 8)
    const auto& policy = PolicySelector{}(all_arches[i]);
    while (i > 0 && policy == PolicySelector{}(all_arches[i - 1]))
#    endif // _CCCL_COMPILER(GCC, >=, 8)
    {
      --i;
    }
    return all_arches[i];
  }

  static constexpr ::cuda::arch_id lowest_arch_with_same_policy[sizeof...(Is)] = {find_lowest(Is)...};
};
#  endif // if _CCCL_STD_VER < 2020

template <int ArchMult, int... CudaArches, typename PolicySelector, typename FunctorT, size_t... Is>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_to_arch_list(
  PolicySelector policy_selector, ::cuda::arch_id device_arch, FunctorT&& f, ::cuda::std::index_sequence<Is...>)
{
  _CCCL_ASSERT(((device_arch == ::cuda::arch_id{(CudaArches * ArchMult) / 10}) || ...),
               "device_arch must appear in the list of architectures compiled for");

  using policy_t = decltype(policy_selector(::cuda::arch_id{}));

  cudaError_t e = cudaErrorInvalidDeviceFunction;
#  if _CCCL_STD_VER >= 2020
  // In C++20, we just create an integral_constant holding the policy, because policies are structural types in C++20.
  // This causes f to be only instantiated for each distinct policy, since the same policy for different arches results
  // in the same integral_constant type passed to f
  (...,
   (device_arch == ::cuda::arch_id{(CudaArches * ArchMult) / 10}
      ? (e = f(
           ::cuda::std::integral_constant<policy_t, policy_selector(::cuda::arch_id{(CudaArches * ArchMult) / 10})>{}))
      : cudaSuccess));
#  else // if _CCCL_STD_VER >= 2020
  // In C++17, we have to collapse architectures with the same policies ourselves, so we instantiate call_for_arch once
  // per policy on the lowest ArchId which produces the same policy
  using resolver_t =
    lowest_arch_resolver<ArchMult, ::cuda::std::integer_sequence<int, CudaArches...>, PolicySelector, Is...>;
  (...,
   (device_arch == ::cuda::arch_id{(CudaArches * ArchMult) / 10}
      ? (e = f(policy_getter_17<PolicySelector, resolver_t::lowest_arch_with_same_policy[Is]>{policy_selector}))
      : cudaSuccess));

#  endif // if _CCCL_STD_VER >= 2020
  return e;
}

template <typename PolicySelector, typename FunctorT, size_t... Is>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_all_arches_helper(
  PolicySelector policy_selector, ::cuda::arch_id device_arch, FunctorT&& f, ::cuda::std::index_sequence<Is...> seq)
{
  static constexpr auto all_arches = ::cuda::__all_arch_ids();
  return dispatch_to_arch_list<10, static_cast<int>(all_arches[Is])...>(policy_selector, device_arch, f, seq);
}

//! Takes a policy hub and instantiates f with the minimum possible number of nullary functor types that return a policy
//! at compile-time (if possible), and then calls the appropriate instantiation based on a runtime GPU architecture.
//! Depending on the used compiler, C++ standard, and available macros, a different number of instantiations may be
//! produced.
template <typename PolicySelector, typename F>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t
dispatch_arch(PolicySelector policy_selector, ::cuda::arch_id device_arch, F&& f)
{
  // when not using CCCL.C, policy_selector is empty since all information is contained in its type
  static_assert(::cuda::std::is_empty_v<PolicySelector>);

  // if we have __CUDA_ARCH_LIST__ or NV_TARGET_SM_INTEGER_LIST, we only poll the policy hub for those arches.
#  ifdef __CUDA_ARCH_LIST__
  [[maybe_unused]] static constexpr auto arch_seq = ::cuda::std::integer_sequence<int, __CUDA_ARCH_LIST__>{};
  return dispatch_to_arch_list<1, __CUDA_ARCH_LIST__>(
    policy_selector, device_arch, ::cuda::std::forward<F>(f), ::cuda::std::make_index_sequence<arch_seq.size()>{});
#  elif defined(NV_TARGET_SM_INTEGER_LIST)
  [[maybe_unused]] static constexpr auto arch_seq = ::cuda::std::integer_sequence<int, NV_TARGET_SM_INTEGER_LIST>{};
  return dispatch_to_arch_list<10, NV_TARGET_SM_INTEGER_LIST>(
    policy_selector, device_arch, ::cuda::std::forward<F>(f), ::cuda::std::make_index_sequence<arch_seq.size()>{});
#  else
  // some compilers don't tell us what arches we are compiling for, so we test all of them
  return dispatch_all_arches_helper(
    policy_selector,
    device_arch,
    ::cuda::std::forward<F>(f),
    ::cuda::std::make_index_sequence<::cuda::__all_arch_ids().size()>{});
#  endif
}

#else // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)

// if we are compiling CCCL.C with runtime policies, we cannot query the policy hub at compile time
_CCCL_EXEC_CHECK_DISABLE
template <typename PolicySelector, typename F>
_CCCL_API _CCCL_FORCEINLINE cudaError_t dispatch_arch(PolicySelector policy_selector, ::cuda::arch_id device_arch, F&& f)
{
  return f([&] {
    return policy_selector(device_arch);
  });
}
#endif // !defined(CUB_DEFINE_RUNTIME_POLICIES) && !_CCCL_COMPILER(NVRTC)
} // namespace detail

CUB_NAMESPACE_END
