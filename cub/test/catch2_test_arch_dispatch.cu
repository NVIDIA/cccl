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

struct arch_policy
{
  arch_id value;

  _CCCL_API constexpr bool operator==(const arch_policy& other) const noexcept
  {
    return value == other.value;
  }

  _CCCL_API constexpr bool operator!=(const arch_policy& other) const noexcept
  {
    return value != other.value;
  }
};

#ifdef CUDA_SM_LIST
struct arch_policies_all
{
  _CCCL_API constexpr auto operator()(arch_id id) const -> arch_policy
  {
    return arch_policy{id};
  }
};

// check that the selected policy exactly matches one of (scaled) arches we compile for
template <arch_id SelectedPolicyArch, int... ArchList>
struct check
{
  static_assert(((SelectedPolicyArch == arch_id{ArchList * CUDA_SM_LIST_SCALE / 10}) || ...));
  using type = cudaError_t;
};

struct closure_all
{
  arch_id id;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION auto operator()(PolicyGetter policy_getter) const ->
    typename check<PolicyGetter{}().value, CUDA_SM_LIST>::type
  {
    constexpr arch_policy active_policy = policy_getter();
    // since an individual policy is generated per architecture, we can do an exact comparison here
    REQUIRE(active_policy.value == id);
    return cudaSuccess;
  }
};

C2H_TEST("dispatch_arch prunes based on __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST", "[util][dispatch]")
{
  for (const int sm_val : {CUDA_SM_LIST})
  {
    const auto id = arch_id{sm_val * CUDA_SM_LIST_SCALE / 10};
    CHECK(cub::detail::dispatch_arch(arch_policies_all{}, id, closure_all{id}) == cudaSuccess);
  }
}
#endif

template <int NumPolicies>
struct check_policy_closure
{
  arch_id id;
  cuda::std::array<arch_id, NumPolicies> policy_ids;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(PolicyGetter policy_getter) const
  {
    constexpr arch_policy active_policy = policy_getter();
    CAPTURE(id, policy_ids);
    const auto policy_arch = *cuda::std::find_if(policy_ids.rbegin(), policy_ids.rend(), [&](arch_id policy_ver) {
      return policy_ver <= id;
    });
    REQUIRE(active_policy.value == policy_arch);
    return cudaSuccess;
  }
};

// distinct policies for 60+, 80+ and 100+
struct arch_policies_some
{
  _CCCL_API constexpr auto operator()(arch_id id) const -> arch_policy
  {
    if (id >= arch_id::sm_100)
    {
      return arch_policy{arch_id::sm_100};
    }
    if (id >= arch_id::sm_80)
    {
      return arch_policy{arch_id::sm_80};
    }
    // default is policy 60
    return arch_policy{arch_id::sm_60};
  }
};

// only a single policy
struct arch_policies_minimal
{
  _CCCL_API constexpr auto operator()(arch_id) const -> arch_policy
  {
    // default is policy 60
    return arch_policy{arch_id::sm_60};
  }
};

C2H_TEST("dispatch_arch invokes correct policy", "[util][dispatch]")
{
  for (const int sm_val : {CUDA_SM_LIST})
  {
    const auto id = arch_id{sm_val * CUDA_SM_LIST_SCALE / 10};

    const auto closure_some =
      check_policy_closure<3>{id, cuda::std::array<arch_id, 3>{arch_id::sm_60, arch_id::sm_80, arch_id::sm_100}};
    CHECK(cub::detail::dispatch_arch(arch_policies_some{}, id, closure_some) == cudaSuccess);

    const auto closure_minimal = check_policy_closure<1>{id, cuda::std::array<arch_id, 1>{arch_id::sm_60}};
    CHECK(cub::detail::dispatch_arch(arch_policies_minimal{}, id, closure_minimal) == cudaSuccess);
  }
}
