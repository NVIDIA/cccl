//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

__device__ void test_current()
{
  // 1. Test cuda::device::current_arch_id() signature.
  static_assert(cuda::std::is_same_v<cuda::arch_id, decltype(cuda::device::current_arch_id())>);
  static_assert(noexcept(cuda::device::current_arch_id()));

  // 2. Test cuda::device::current_arch_id() in constexpr context. Unsupported with nvc++ -cuda.
#if !_CCCL_CUDA_COMPILER(NVHPC)
  if constexpr (cuda::device::current_arch_id() == cuda::arch_id{})
  {
    // cuda::arch_id{} is an invalid architecture, so this statement should be unrachable.
    assert(false);
  }
#endif // !_CCCL_CUDA_COMPILER(NVHPC)

  // 3. Test cuda::device::current_arch_id() against the NV_IF_TARGET macros
  [[maybe_unused]] cuda::arch_id arch = cuda::device::current_arch_id();

  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (assert(arch == cuda::arch_id::sm_90a); return;),
    NV_HAS_FEATURE_SM_100a,
    (assert(arch == cuda::arch_id::sm_100a); return;),
    NV_HAS_FEATURE_SM_103a,
    (assert(arch == cuda::arch_id::sm_103a); return;),
    NV_HAS_FEATURE_SM_110a,
    (assert(arch == cuda::arch_id::sm_110a); return;),
    NV_HAS_FEATURE_SM_120a,
    (assert(arch == cuda::arch_id::sm_120a); return;),
    NV_HAS_FEATURE_SM_121a,
    (assert(arch == cuda::arch_id::sm_121a); return;))

  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_60,
    (assert(arch == cuda::arch_id::sm_60); return;),
    NV_IS_EXACTLY_SM_61,
    (assert(arch == cuda::arch_id::sm_61); return;),
    NV_IS_EXACTLY_SM_62,
    (assert(arch == cuda::arch_id::sm_62); return;),
    NV_IS_EXACTLY_SM_70,
    (assert(arch == cuda::arch_id::sm_70); return;),
    NV_IS_EXACTLY_SM_75,
    (assert(arch == cuda::arch_id::sm_75); return;),
    NV_IS_EXACTLY_SM_80,
    (assert(arch == cuda::arch_id::sm_80); return;),
    NV_IS_EXACTLY_SM_86,
    (assert(arch == cuda::arch_id::sm_86); return;),
    NV_IS_EXACTLY_SM_87,
    (assert(arch == cuda::arch_id::sm_87); return;),
    NV_IS_EXACTLY_SM_88,
    (assert(arch == cuda::arch_id::sm_88); return;),
    NV_IS_EXACTLY_SM_89,
    (assert(arch == cuda::arch_id::sm_89); return;))
  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_90,
    (assert(arch == cuda::arch_id::sm_90); return;),
    NV_IS_EXACTLY_SM_100,
    (assert(arch == cuda::arch_id::sm_100); return;),
    NV_IS_EXACTLY_SM_103,
    (assert(arch == cuda::arch_id::sm_103); return;),
    NV_IS_EXACTLY_SM_110,
    (assert(arch == cuda::arch_id::sm_110); return;),
    NV_IS_EXACTLY_SM_120,
    (assert(arch == cuda::arch_id::sm_120); return;),
    NV_IS_EXACTLY_SM_121,
    (assert(arch == cuda::arch_id::sm_121); return;),
    NV_ANY_TARGET,
    (assert(false);) // fail for unknown architecture
  )
}

__host__ __device__ constexpr bool test()
{
  // 1. Test cuda::arch_id enum values.
  static_assert(cuda::std::is_scoped_enum_v<cuda::arch_id>);
  static_assert(cuda::std::is_same_v<cuda::std::underlying_type_t<cuda::arch_id>, int>);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_60) == 60);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_61) == 61);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_62) == 62);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_70) == 70);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_75) == 75);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_80) == 80);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_86) == 86);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_87) == 87);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_88) == 88);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_89) == 89);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_90) == 90);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_100) == 100);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_103) == 103);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_110) == 110);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_120) == 120);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_121) == 121);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_90a) == 90 * 100000);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_100a) == 100 * 100000);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_103a) == 103 * 100000);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_110a) == 110 * 100000);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_120a) == 120 * 100000);
  static_assert(cuda::std::to_underlying(cuda::arch_id::sm_121a) == 121 * 100000);

  // 2. Test cuda::to_arch_id(cuda::compute_capability).
  {
    static_assert(cuda::std::is_same_v<cuda::arch_id, decltype(cuda::to_arch_id(cuda::compute_capability{}))>);
    static_assert(noexcept(cuda::to_arch_id(cuda::compute_capability{})));

    cuda::arch_id id_lowest = cuda::to_arch_id(cuda::compute_capability{60});
    assert(id_lowest == cuda::arch_id::sm_60);
    cuda::arch_id id_highest = cuda::to_arch_id(cuda::compute_capability{120});
    assert(id_highest == cuda::arch_id::sm_120);
  }

  // 3. Test cuda::to_arch_specific_id(cuda::compute_capability).
  {
    static_assert(cuda::std::is_same_v<cuda::arch_id, decltype(cuda::to_arch_specific_id(cuda::compute_capability{}))>);
    static_assert(noexcept(cuda::to_arch_specific_id(cuda::compute_capability{})));

    cuda::arch_id id_lowest = cuda::to_arch_specific_id(cuda::compute_capability{90});
    assert(id_lowest == cuda::arch_id::sm_90a);
    cuda::arch_id id_highest = cuda::to_arch_specific_id(cuda::compute_capability{120});
    assert(id_highest == cuda::arch_id::sm_120a);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  NV_IF_TARGET(NV_IS_DEVICE, (test_current();))
  return 0;
}
