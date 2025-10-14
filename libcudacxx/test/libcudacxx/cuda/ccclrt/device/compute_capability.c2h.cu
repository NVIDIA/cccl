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
  [[maybe_unused]] const cuda::compute_capability cc = cuda::device::current_compute_capability();
}

C2H_CCCLRT_TEST("Compute capability", "[device]")
{
  // 1. default constructor
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_default_constructible_v<cuda::compute_capability>);
    constexpr cuda::compute_capability cc;
    CCCLRT_REQUIRE(cc.get() == 0);
  }

  // 2. constructor from compute capability in format 10 * major + minor
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_constructible_v<cuda::compute_capability, int>);
    STATIC_REQUIRE(!cuda::std::is_convertible_v<int, cuda::compute_capability>);
    constexpr cuda::compute_capability cc{148};
    CCCLRT_REQUIRE(cc.get() == 148);
  }

  // 3. constructor from major and minor
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_constructible_v<cuda::compute_capability, int, int>);
    constexpr cuda::compute_capability cc{8, 9};
    CCCLRT_REQUIRE(cc.get() == 89);
  }

  // 4. constructor from cuda::arch_id
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_constructible_v<cuda::compute_capability, cuda::arch_id>);
    STATIC_REQUIRE(!cuda::std::is_convertible_v<cuda::arch_id, cuda::compute_capability>);
    constexpr cuda::compute_capability cc1{cuda::arch_id::sm_100};
    CCCLRT_REQUIRE(cc1.get() == 100);
    constexpr cuda::compute_capability cc2{cuda::arch_id::sm_100a};
    CCCLRT_REQUIRE(cc2.get() == 100);
  }

  // 5. copy constructor
  {
    STATIC_REQUIRE(cuda::std::is_trivially_copy_constructible_v<cuda::compute_capability>);
    constexpr cuda::compute_capability cc1{cuda::arch_id::sm_100};
    constexpr cuda::compute_capability cc2{cc1};
    CCCLRT_REQUIRE(cc1.get() == 100);
    CCCLRT_REQUIRE(cc2.get() == 100);
  }

  // 6. assignment operator
  {
    STATIC_REQUIRE(cuda::std::is_nothrow_copy_assignable_v<cuda::compute_capability>);
    constexpr cuda::compute_capability cc1{cuda::arch_id::sm_100};
    cuda::compute_capability cc2;
    CCCLRT_REQUIRE(cc1.get() == 100);
    CCCLRT_REQUIRE(cc2.get() == 0);

    cc2 = cc1;
    CCCLRT_REQUIRE(cc1.get() == 100);
    CCCLRT_REQUIRE(cc2.get() == 100);
  }

  // 7. get()
  {
    STATIC_REQUIRE(noexcept(cuda::compute_capability{}.get()));
    constexpr cuda::compute_capability cc{cuda::arch_id::sm_100};
    CCCLRT_REQUIRE(cc.get() == 100);
  }

  // 8. major()
  {
    STATIC_REQUIRE(noexcept(cuda::compute_capability{}.major()));
    constexpr cuda::compute_capability cc{cuda::arch_id::sm_100};
    CCCLRT_REQUIRE(cc.major() == 10);
  }

  // 9. minor()
  {
    STATIC_REQUIRE(noexcept(cuda::compute_capability{}.minor()));
    constexpr cuda::compute_capability cc{cuda::arch_id::sm_89};
    CCCLRT_REQUIRE(cc.minor() == 9);
  }

  // 10. operator int()
  {
    STATIC_REQUIRE(noexcept(static_cast<int>(cuda::compute_capability{})));
    STATIC_REQUIRE(!cuda::std::is_convertible_v<cuda::compute_capability, int>);
    constexpr cuda::compute_capability cc{cuda::arch_id::sm_89};
    CCCLRT_REQUIRE(static_cast<int>(cc) == 89);
  }

  // 11. comparison operators
  {
    STATIC_REQUIRE(noexcept(operator==(cuda::compute_capability{}, cuda::compute_capability{})));
    STATIC_REQUIRE(noexcept(operator!=(cuda::compute_capability{}, cuda::compute_capability{})));
    STATIC_REQUIRE(noexcept(operator<(cuda::compute_capability{}, cuda::compute_capability{})));
    STATIC_REQUIRE(noexcept(operator<=(cuda::compute_capability{}, cuda::compute_capability{})));
    STATIC_REQUIRE(noexcept(operator>(cuda::compute_capability{}, cuda::compute_capability{})));
    STATIC_REQUIRE(noexcept(operator>=(cuda::compute_capability{}, cuda::compute_capability{})));

    constexpr cuda::compute_capability cc1{127};
    constexpr cuda::compute_capability cc2{43};

    CCCLRT_REQUIRE(cc1 == cc1);
    CCCLRT_REQUIRE(cc2 == cc2);

    CCCLRT_REQUIRE(cc1 != cc2);
    CCCLRT_REQUIRE(cc2 != cc1);

    CCCLRT_REQUIRE(!(cc1 < cc1));
    CCCLRT_REQUIRE(!(cc1 < cc2));
    CCCLRT_REQUIRE(!(cc2 < cc2));
    CCCLRT_REQUIRE(cc2 < cc1);

    CCCLRT_REQUIRE(cc1 <= cc1);
    CCCLRT_REQUIRE(!(cc1 <= cc2));
    CCCLRT_REQUIRE(cc2 <= cc2);
    CCCLRT_REQUIRE(cc2 <= cc1);

    CCCLRT_REQUIRE(!(cc1 > cc1));
    CCCLRT_REQUIRE(cc1 > cc2);
    CCCLRT_REQUIRE(!(cc2 > cc2));
    CCCLRT_REQUIRE(!(cc2 > cc1));

    CCCLRT_REQUIRE(cc1 >= cc1);
    CCCLRT_REQUIRE(cc1 > cc2);
    CCCLRT_REQUIRE(cc2 >= cc2);
    CCCLRT_REQUIRE(!(cc2 > cc1));
  }
}
