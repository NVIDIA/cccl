// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/arch_dispatch.cuh>

#include <cuda/std/__algorithm/find_if.h>

#include <c2h/catch2_test_helper.h>

#ifdef __CUDA_ARCH_LIST__
#  define CUDA_SM_LIST       __CUDA_ARCH_LIST__
#  define CUDA_SM_LIST_SCALE 1
#elif defined(NV_TARGET_SM_INTEGER_LIST)
#  define CUDA_SM_LIST       NV_TARGET_SM_INTEGER_LIST
#  define CUDA_SM_LIST_SCALE 10
#endif

using cuda::arch_id;

struct a_policy
{
  arch_id value;

  _CCCL_API constexpr bool operator==(const a_policy& other) const noexcept
  {
    return value == other.value;
  }

  _CCCL_API constexpr bool operator!=(const a_policy& other) const noexcept
  {
    return value != other.value;
  }
};

struct policy_selector_all
{
  _CCCL_API constexpr auto operator()(arch_id id) const -> a_policy
  {
    return a_policy{id};
  }
};

#ifdef CUDA_SM_LIST
template <arch_id SelectedPolicyArch, int... ArchList>
void check_arch_is_in_list()
{
  static_assert(((SelectedPolicyArch == arch_id{ArchList * CUDA_SM_LIST_SCALE / 10}) || ...));
}
#endif // CUDA_SM_LIST

struct closure_all
{
  arch_id id;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION auto operator()(PolicyGetter policy_getter) const -> cudaError_t
  {
#ifdef CUDA_SM_LIST
    check_arch_is_in_list<PolicyGetter{}().value, CUDA_SM_LIST>();
#endif // CUDA_SM_LIST
    constexpr a_policy active_policy = policy_getter();
    // since an individual policy is generated per architecture, we can do an exact comparison here
    REQUIRE(active_policy.value == id);
    return cudaSuccess;
  }
};

C2H_TEST("dispatch_arch prunes based on __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST", "[util][dispatch]")
{
#ifdef CUDA_SM_LIST
  for (const int sm_val : {CUDA_SM_LIST})
  {
    const auto id = arch_id{sm_val * CUDA_SM_LIST_SCALE / 10};
#else
  for (const arch_id id : cuda::__all_arch_ids())
  {
#endif
    CHECK(cub::detail::dispatch_arch(policy_selector_all{}, id, closure_all{id}) == cudaSuccess);
  }
}

template <int NumPolicies>
struct check_policy_closure
{
  arch_id id;
  cuda::std::array<arch_id, NumPolicies> policy_ids;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(PolicyGetter policy_getter) const
  {
    constexpr a_policy active_policy = policy_getter();
    CAPTURE(id, policy_ids);
    const auto policy_arch = *cuda::std::find_if(policy_ids.rbegin(), policy_ids.rend(), [&](arch_id policy_ver) {
      return policy_ver <= id;
    });
    REQUIRE(active_policy.value == policy_arch);
    return cudaSuccess;
  }
};

// distinct policies for 60+, 80+ and 100+
struct policy_selector_some
{
  _CCCL_API constexpr auto operator()(arch_id id) const -> a_policy
  {
    if (id >= arch_id::sm_100)
    {
      return a_policy{arch_id::sm_100};
    }
    if (id >= arch_id::sm_80)
    {
      return a_policy{arch_id::sm_80};
    }
    // default is policy 60
    return a_policy{arch_id::sm_60};
  }
};

// only a single policy
struct policy_selector_minimal
{
  _CCCL_API constexpr auto operator()(arch_id) const -> a_policy
  {
    // default is policy 60
    return a_policy{arch_id::sm_60};
  }
};

C2H_TEST("dispatch_arch invokes correct policy", "[util][dispatch]")
{
#ifdef CUDA_SM_LIST
  for (const int sm_val : {CUDA_SM_LIST})
  {
    const auto id = arch_id{sm_val * CUDA_SM_LIST_SCALE / 10};
#else
  for (const arch_id id : cuda::__all_arch_ids())
  {
#endif
    const auto closure_some =
      check_policy_closure<3>{id, cuda::std::array<arch_id, 3>{arch_id::sm_60, arch_id::sm_80, arch_id::sm_100}};
    CHECK(cub::detail::dispatch_arch(policy_selector_some{}, id, closure_some) == cudaSuccess);

    const auto closure_minimal = check_policy_closure<1>{id, cuda::std::array<arch_id, 1>{arch_id::sm_60}};
    CHECK(cub::detail::dispatch_arch(policy_selector_minimal{}, id, closure_minimal) == cudaSuccess);
  }
}

#if _CCCL_HAS_CONCEPTS()
// not comparable
struct bad_policy
{};

struct policy_selector_not_regular
{
  _CCCL_API auto operator()(arch_id) const -> bad_policy
  {
    return bad_policy{};
  }
};

C2H_TEST("policy_selector concept", "[util][dispatch]")
{
  STATIC_REQUIRE(::cub::detail::policy_selector<policy_selector_all, a_policy>);
  STATIC_REQUIRE(::cub::detail::policy_selector<policy_selector_some, a_policy>);
  STATIC_REQUIRE(::cub::detail::policy_selector<policy_selector_minimal, a_policy>);
  STATIC_REQUIRE(!::cub::detail::policy_selector<policy_selector_not_regular, bad_policy>);
  STATIC_REQUIRE(!::cub::detail::policy_selector<policy_selector_all, bad_policy>); // policy mismatch
}
#endif // _CCCL_HAS_CONCEPTS()
