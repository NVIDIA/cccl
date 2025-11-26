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

#include <testing.cuh>

__global__ void kernel()
{
  [[maybe_unused]] const cuda::arch_id arch = cuda::device::current_arch_id();

  if constexpr (cuda::device::current_arch_id() == cuda::arch_id::sm_100)
  {
    return;
  }
}

C2H_CCCLRT_TEST("Architecture id", "[device]")
{
  STATIC_REQUIRE(cuda::std::is_scoped_enum_v<cuda::arch_id>);
  STATIC_REQUIRE(cuda::std::is_same_v<cuda::std::underlying_type_t<cuda::arch_id>, int>);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_60) == 60);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_61) == 61);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_62) == 62);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_70) == 70);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_75) == 75);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_80) == 80);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_86) == 86);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_87) == 87);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_88) == 88);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_89) == 89);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_90) == 90);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_100) == 100);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_103) == 103);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_110) == 110);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_120) == 120);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_121) == 121);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_90a) == 90 * 100000);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_100a) == 100 * 100000);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_103a) == 103 * 100000);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_110a) == 110 * 100000);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_120a) == 120 * 100000);
  STATIC_REQUIRE(cuda::std::to_underlying(cuda::arch_id::sm_121a) == 121 * 100000);

  // cuda::to_arch_id(cuda::compute_capability)
  {
    STATIC_REQUIRE(noexcept(cuda::to_arch_id(cuda::compute_capability{})));
    constexpr cuda::arch_id id_lowest = cuda::to_arch_id(cuda::compute_capability{60});
    CCCLRT_REQUIRE(id_lowest == cuda::arch_id::sm_60);
    constexpr cuda::arch_id id_highest = cuda::to_arch_id(cuda::compute_capability{120});
    CCCLRT_REQUIRE(id_highest == cuda::arch_id::sm_120);
  }

  // cuda::to_arch_specific_id(cuda::compute_capability)
  {
    STATIC_REQUIRE(noexcept(cuda::to_arch_specific_id(cuda::compute_capability{})));
    constexpr cuda::arch_id id_lowest = cuda::to_arch_specific_id(cuda::compute_capability{90});
    CCCLRT_REQUIRE(id_lowest == cuda::arch_id::sm_90a);
    constexpr cuda::arch_id id_highest = cuda::to_arch_specific_id(cuda::compute_capability{120});
    CCCLRT_REQUIRE(id_highest == cuda::arch_id::sm_120a);
  }
}
