// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/cc_dispatch.cuh>

#include <cuda/std/__algorithm/find_if.h>

#include <c2h/catch2_test_helper.h>

using cuda::compute_capability;

struct a_policy
{
  int value;

  _CCCL_HOST_DEVICE_API constexpr bool operator==(const a_policy& other) const noexcept
  {
    return value == other.value;
  }

  _CCCL_HOST_DEVICE_API constexpr bool operator!=(const a_policy& other) const noexcept
  {
    return value != other.value;
  }
};

struct policy_selector_all
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(compute_capability cc) const -> a_policy
  {
    return a_policy{cc.get()};
  }
};

template <int SelectedPolicyCC, cuda::std::size_t N>
void check_cc_is_in_list(cuda::std::array<compute_capability, N> cc_list)
{
  for (const auto cc : cc_list)
  {
    if (compute_capability{SelectedPolicyCC} == cc)
    {
      return;
    }
  }
  FAIL("SelectedPolicyCC is not present in cc_list");
}

struct closure_all
{
  int cc;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION auto operator()(PolicyGetter policy_getter) const -> cudaError_t
  {
    check_cc_is_in_list<PolicyGetter{}().value>(cuda::__target_compute_capabilities());
    constexpr a_policy active_policy = policy_getter();
    // since an individual policy is generated per compute capability, we can do an exact comparison here
    REQUIRE(active_policy.value == cc);
    return cudaSuccess;
  }
};

C2H_TEST("dispatch_compute_cap prunes based on __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST", "[util][dispatch]")
{
  for (const auto cc : cuda::__target_compute_capabilities())
  {
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
    const auto policy_cc = *cuda::std::find_if(policy_ccs.rbegin(), policy_ccs.rend(), [&](auto policy_ver) {
      return policy_ver <= cc;
    });
    REQUIRE(active_policy.value == policy_cc);
    return cudaSuccess;
  }
};

// distinct policies for 60+, 80+ and 100+
struct policy_selector_some
{
  _CCCL_HOST_DEVICE_API constexpr auto operator()(compute_capability cc) const -> a_policy
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
  _CCCL_HOST_DEVICE_API constexpr auto operator()(compute_capability) const -> a_policy
  {
    // default is policy 60
    return a_policy{60};
  }
};

C2H_TEST("dispatch_compute_cap invokes correct policy", "[util][dispatch]")
{
  for (const auto cc : cuda::__target_compute_capabilities())
  {
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
  _CCCL_HOST_DEVICE_API auto operator()(compute_capability) const -> bad_policy
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
