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

using cuda::compute_capability;

struct a_policy
{
  int value;

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
  _CCCL_API constexpr auto operator()(compute_capability cc) const -> a_policy
  {
    return a_policy{cc.get()};
  }
};

#ifdef CUDA_SM_LIST
template <int SelectedPolicyCC, int... ArchList>
void check_arch_is_in_list()
{
  static_assert(
    ((compute_capability{SelectedPolicyCC} == compute_capability{ArchList * CUDA_SM_LIST_SCALE / 10}) || ...));
}
#endif // CUDA_SM_LIST

struct closure_all
{
  int cc;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION auto operator()(PolicyGetter policy_getter) const -> cudaError_t
  {
#ifdef CUDA_SM_LIST
    check_arch_is_in_list<PolicyGetter{}().value, CUDA_SM_LIST>();
#endif // CUDA_SM_LIST
    constexpr a_policy active_policy = policy_getter();
    // since an individual policy is generated per architecture, we can do an exact comparison here
    REQUIRE(active_policy.value == cc);
    return cudaSuccess;
  }
};

C2H_TEST("dispatch_compute_cap prunes based on __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST", "[util][dispatch]")
{
#ifdef CUDA_SM_LIST
  for (const int sm_val : {CUDA_SM_LIST})
  {
    const compute_capability cc{sm_val * CUDA_SM_LIST_SCALE / 10};
#else
  for (const compute_capability cc : cuda::__all_compute_capabilities())
  {
#endif
    CHECK(cub::detail::dispatch_compute_cap(policy_selector_all{}, cc, closure_all{cc.get()}) == cudaSuccess);
  }
}

template <int NumPolicies>
struct check_policy_closure
{
  int cc;
  cuda::std::array<int, NumPolicies> policy_ccs;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION cudaError_t operator()(PolicyGetter policy_getter) const
  {
    constexpr a_policy active_policy = policy_getter();
    CAPTURE(cc, policy_ccs);
    const auto policy_arch = *cuda::std::find_if(policy_ccs.rbegin(), policy_ccs.rend(), [&](auto policy_ver) {
      return policy_ver <= cc;
    });
    REQUIRE(active_policy.value == policy_arch);
    return cudaSuccess;
  }
};

// distinct policies for 60+, 80+ and 100+
struct policy_selector_some
{
  _CCCL_API constexpr auto operator()(compute_capability cc) const -> a_policy
  {
    if (cc >= compute_capability{10, 0})
    {
      return a_policy{100};
    }
    if (cc >= compute_capability{8, 0})
    {
      return a_policy{80};
    }
    // default is policy 60
    return a_policy{60};
  }
};

// only a single policy
struct policy_selector_minimal
{
  _CCCL_API constexpr auto operator()(compute_capability) const -> a_policy
  {
    // default is policy 60
    return a_policy{60};
  }
};

C2H_TEST("dispatch_compute_cap invokes correct policy", "[util][dispatch]")
{
#ifdef CUDA_SM_LIST
  for (const int sm_val : {CUDA_SM_LIST})
  {
    const compute_capability cc{sm_val * CUDA_SM_LIST_SCALE / 10};
#else
  for (const compute_capability cc : cuda::__all_compute_capabilities())
  {
#endif
    const auto closure_some = check_policy_closure<3>{cc.get(), cuda::std::array{60, 80, 100}};
    CHECK(cub::detail::dispatch_compute_cap(policy_selector_some{}, cc, closure_some) == cudaSuccess);

    const auto closure_minimal = check_policy_closure<1>{cc.get(), cuda::std::array{60}};
    CHECK(cub::detail::dispatch_compute_cap(policy_selector_minimal{}, cc, closure_minimal) == cudaSuccess);
  }
}

#if _CCCL_HAS_CONCEPTS()
// not comparable
struct bad_policy
{};

struct policy_selector_not_regular
{
  _CCCL_API auto operator()(compute_capability) const -> bad_policy
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
