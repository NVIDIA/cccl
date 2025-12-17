// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>

#include <cuda/devices>
#include <cuda/std/__algorithm/find_if.h>
#include <cuda/std/array>

#include "catch2_test_launch_helper.h"
#include <c2h/catch2_test_helper.h>

CUB_NAMESPACE_BEGIN

CUB_DETAIL_KERNEL_ATTRIBUTES void write_ptx_version_kernel(int* d_kernel_cuda_arch)
{
  *d_kernel_cuda_arch = CUB_PTX_ARCH;
}

CUB_RUNTIME_FUNCTION static cudaError_t get_cuda_arch_from_kernel(
  void* d_temp_storage, size_t& temp_storage_bytes, int* d_kernel_cuda_arch, int* ptx_version, cudaStream_t stream = 0)
{
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }
  write_ptx_version_kernel<<<1, 1, 0, stream>>>(d_kernel_cuda_arch);
  return cub::PtxVersion(*ptx_version);
}

CUB_NAMESPACE_END

// %PARAM% TEST_LAUNCH lid 0:1:2
DECLARE_LAUNCH_WRAPPER(cub::get_cuda_arch_from_kernel, get_cuda_arch_from_kernel);

C2H_TEST("CUB correctly identifies the ptx version the kernel was compiled for", "[util][dispatch]")
{
  constexpr std::size_t single_item = 1;
  c2h::device_vector<int> cuda_arch(single_item);

  // Query the arch the kernel was actually compiled for
  int ptx_version = [&]() -> int {
    int* buffer{};
    cudaMallocHost(&buffer, sizeof(*buffer));
    get_cuda_arch_from_kernel(thrust::raw_pointer_cast(cuda_arch.data()), buffer);
    int result = *buffer;
    cudaFreeHost(buffer);
    return result;
  }();

  int kernel_cuda_arch = cuda_arch[0];

  // Host cub::PtxVersion
  int host_ptx_version{};
  cub::PtxVersion(host_ptx_version);

  // Ensure variable was properly populated
  REQUIRE(0 != kernel_cuda_arch);

  // Ensure that the ptx version corresponds to the arch the kernel was compiled for
  REQUIRE(ptx_version == kernel_cuda_arch);
  REQUIRE(host_ptx_version == kernel_cuda_arch);
}

#ifdef __CUDA_ARCH_LIST__
#  define CUDA_SM_LIST       __CUDA_ARCH_LIST__
#  define CUDA_SM_LIST_SCALE 1
#elif defined(NV_TARGET_SM_INTEGER_LIST)
#  define CUDA_SM_LIST       NV_TARGET_SM_INTEGER_LIST
#  define CUDA_SM_LIST_SCALE 10
#endif

#ifdef CUDA_SM_LIST
C2H_TEST("PtxVersion returns a value from __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST", "[util][dispatch]")
{
  int ptx_version = 0;
  REQUIRE(cub::PtxVersion(ptx_version) == cudaSuccess);
  auto arch_list = std::vector<int>{CUDA_SM_LIST};
  for (auto& a : arch_list)
  {
    a *= CUDA_SM_LIST_SCALE;
  }
  REQUIRE(std::find(arch_list.begin(), arch_list.end(), ptx_version) != arch_list.end());
}
#endif

#define GEN_POLICY(cur, prev)                                             \
  struct policy##cur : cub::ChainedPolicy<cur, policy##cur, policy##prev> \
  {                                                                       \
    static constexpr int value = cur;                                     \
  }

#ifdef CUDA_SM_LIST
// We list policies for all virtual architectures that __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST can contain, so the
// actual architectures the tests are compiled for should match to one of those
struct policy_hub_all
{
  // for the list of supported architectures,
  // see libcudacxx/include/nv/target or libcudacxx/include/cuda/__device/arch_id.h
  GEN_POLICY(500, 500);
  GEN_POLICY(520, 500);
  GEN_POLICY(530, 520);
  GEN_POLICY(600, 530);
  GEN_POLICY(610, 600);
  GEN_POLICY(620, 610);
  GEN_POLICY(700, 620);
  GEN_POLICY(720, 700);
  GEN_POLICY(750, 720);
  GEN_POLICY(800, 750);
  GEN_POLICY(860, 800);
  GEN_POLICY(870, 860);
  GEN_POLICY(880, 870);
  GEN_POLICY(890, 880);
  GEN_POLICY(900, 890);
  GEN_POLICY(1000, 900);
  GEN_POLICY(1010, 1000);
  GEN_POLICY(1030, 1010);
  GEN_POLICY(1100, 1030);
  GEN_POLICY(1200, 1100);
  GEN_POLICY(1210, 1200);
  // add more policies here when new architectures emerge
  GEN_POLICY(2000, 1210); // non-existing architecture, just to test pruning

  using max_policy = policy2000;
};

// check that the selected policy exactly matches one of (scaled) arches we compile for
template <int SelectedPolicyArch, int... ArchList>
struct check
{
  static_assert(((SelectedPolicyArch == ArchList * CUDA_SM_LIST_SCALE) || ...));
  using type = cudaError_t;
};

struct closure_all
{
  int ptx_version;

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION auto Invoke() const -> typename check<ActivePolicy::value, CUDA_SM_LIST>::type
  {
    // since policy_hub_all lists all PTX virtual architectures, we can do an exact comparison here
#  if TEST_LAUNCH == 0
    REQUIRE(+ActivePolicy::value == ptx_version);
#  endif // TEST_LAUNCH == 0
    // the returned error code will be checked by the launch helper
    return +ActivePolicy::value == ptx_version ? cudaSuccess : cudaErrorInvalidValue;
  }
};

CUB_RUNTIME_FUNCTION cudaError_t
check_chained_policy_prunes_to_arch_list(void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t = 0)
{
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }
  int ptx_version = 0;
  cub::PtxVersion(ptx_version);
  closure_all c{ptx_version};
  return policy_hub_all::max_policy::Invoke(ptx_version, c);
}

DECLARE_LAUNCH_WRAPPER(check_chained_policy_prunes_to_arch_list, check_wrapper_all);

C2H_TEST("ChainedPolicy prunes based on __CUDA_ARCH_LIST__/NV_TARGET_SM_INTEGER_LIST", "[util][dispatch]")
{
  check_wrapper_all();
}
#endif

template <int NumPolicies>
struct check_policy_closure
{
  int ptx_version;
  cuda::std::array<int, NumPolicies> policies;

  // quick way to get a comparator for find_if below
  __host__ __device__ bool operator()(int policy_ver) const
  {
    return policy_ver <= ptx_version;
  }

  template <typename ActivePolicy>
  CUB_RUNTIME_FUNCTION cudaError_t Invoke() const
  {
#define CHECK_EXPR +ActivePolicy::value == *cuda::std::find_if(policies.rbegin(), policies.rend(), *this)

#if TEST_LAUNCH == 0
    CAPTURE(ptx_version, policies);
    REQUIRE(CHECK_EXPR);
#else // TEST_LAUNCH == 0
    if (!(CHECK_EXPR))
    {
      printf("Check `%s` failed!\n  ptx_version=%d\n  ActivePolicy::value=%d\n  policies=",
             THRUST_PP_STRINGIZE(CHECK_EXPR),
             ptx_version,
             ActivePolicy::value);
      for (int i = 0; i < NumPolicies; i++)
      {
        printf("%d,", policies[i]);
      }
      printf("\n");
    }
#endif // TEST_LAUNCH == 0
    // the returned error code will be checked by the launch helper
    return (CHECK_EXPR) ? cudaSuccess : cudaErrorInvalidValue;
#undef CHECK_EXPR
  }
};

template <typename PolicyHub, int NumPolicies>
CUB_RUNTIME_FUNCTION cudaError_t check_chained_policy_selects_correct_policy(
  void* d_temp_storage, size_t& temp_storage_bytes, cuda::std::array<int, NumPolicies> policies, cudaStream_t = 0)
{
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }
  int ptx_version = 0;
  cub::PtxVersion(ptx_version);
  check_policy_closure<NumPolicies> c{ptx_version, std::move(policies)};
  return PolicyHub::max_policy::Invoke(ptx_version, c);
}

DECLARE_TMPL_LAUNCH_WRAPPER(check_chained_policy_selects_correct_policy,
                            check_wrapper_some,
                            ESCAPE_LIST(typename PolicyHub, int NumPolicies),
                            ESCAPE_LIST(PolicyHub, NumPolicies));

struct policy_hub_some
{
  GEN_POLICY(500, 500);
  GEN_POLICY(700, 500);
  GEN_POLICY(900, 700);
  GEN_POLICY(2000, 900); // non-existing architecture, just to test
  using max_policy = policy2000;
};

struct policy_hub_few
{
  GEN_POLICY(500, 500);
  GEN_POLICY(860, 500);
  GEN_POLICY(2000, 860); // non-existing architecture, just to test
  using max_policy = policy2000;
};

struct policy_hub_minimal
{
  GEN_POLICY(500, 500);
  using max_policy = policy500;
};

C2H_TEST("ChainedPolicy invokes correct policy", "[util][dispatch]")
{
  SECTION("policy_hub_some")
  {
    check_wrapper_some<policy_hub_some, 4>(cuda::std::array<int, 4>{500, 700, 900, 2000});
  }
  SECTION("policy_hub_few")
  {
    check_wrapper_some<policy_hub_few, 3>(cuda::std::array<int, 3>{500, 860, 2000});
  }
  SECTION("policy_hub_minimal")
  {
    check_wrapper_some<policy_hub_minimal, 1>(cuda::std::array<int, 1>{500});
  }
}

__global__ void test_max_potential_dynamic_smem_bytes_kernel()
{
  // use inline PTX so the variable doesn't get optimized out
  asm volatile(".shared .align 1 .b8 static_smem[4096];");
}

#if defined(CUB_RDC_ENABLED)
__global__ void test_max_potential_dynamic_smem_bytes_device(int* result)
{
  // Just compile on device.
  cub::MaxPotentialDynamicSmemBytes(*result, test_max_potential_dynamic_smem_bytes_kernel);
}
#endif // CUB_RDC_ENABLED

C2H_TEST("MaxPotentialDynamicSmemBytes", "[util][launch]")
{
  cuda::device_ref device{0};

  // Calculate the expected max potential dynamic shared memory size.
  const auto max_smem_per_block_optin = device.attribute(cuda::device_attributes::max_shared_memory_per_block_optin);
  const auto reserved_smem_per_block  = device.attribute(cuda::device_attributes::reserved_shared_memory_per_block);
  const auto expected                 = static_cast<int>(max_smem_per_block_optin - reserved_smem_per_block - 4096);

  // 1. Test positive case.
  int dyn_smem_size{};
  REQUIRE(
    cub::MaxPotentialDynamicSmemBytes(dyn_smem_size, test_max_potential_dynamic_smem_bytes_kernel) == cudaSuccess);
  REQUIRE(dyn_smem_size == expected);

  // 2. Test that we return -1 if an error occurs.
  REQUIRE(cub::MaxPotentialDynamicSmemBytes(dyn_smem_size, nullptr) != cudaSuccess);
  REQUIRE(dyn_smem_size == -1);
}
