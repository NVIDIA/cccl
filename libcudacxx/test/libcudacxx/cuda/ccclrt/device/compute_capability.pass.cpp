//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_DEFINITIONS: CCCL_IGNORE_DEPRECATED_API

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

__device__ void test_current()
{
  // 1. Test cuda::device::current_compute_capability() signature.
  static_assert(cuda::std::is_same_v<cuda::compute_capability, decltype(cuda::device::current_compute_capability())>);
  static_assert(noexcept(cuda::device::current_compute_capability()));

  // 1. Test cuda::device::current_compute_capability() in constexpr context. Unsupported with nvc++ -cuda.
#if !_CCCL_CUDA_COMPILER(NVHPC)
  if constexpr (cuda::device::current_compute_capability() == cuda::compute_capability{})
  {
    // cuda::current_compute_capability{} is an invalid compute capability, so this statement should be unrachable.
    assert(false);
  }
#endif // !_CCCL_CUDA_COMPILER(NVHPC)

  // 2. Test cuda::device::current_compute_capability() against the NV_IF_TARGET macros
  [[maybe_unused]] cuda::compute_capability cc = cuda::device::current_compute_capability();

  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_60,
    (assert(cc == cuda::compute_capability{60}); return;),
    NV_IS_EXACTLY_SM_61,
    (assert(cc == cuda::compute_capability{61}); return;),
    NV_IS_EXACTLY_SM_62,
    (assert(cc == cuda::compute_capability{62}); return;),
    NV_IS_EXACTLY_SM_70,
    (assert(cc == cuda::compute_capability{70}); return;),
    NV_IS_EXACTLY_SM_75,
    (assert(cc == cuda::compute_capability{75}); return;),
    NV_IS_EXACTLY_SM_80,
    (assert(cc == cuda::compute_capability{80}); return;),
    NV_IS_EXACTLY_SM_86,
    (assert(cc == cuda::compute_capability{86}); return;),
    NV_IS_EXACTLY_SM_87,
    (assert(cc == cuda::compute_capability{87}); return;),
    NV_IS_EXACTLY_SM_88,
    (assert(cc == cuda::compute_capability{88}); return;),
    NV_IS_EXACTLY_SM_89,
    (assert(cc == cuda::compute_capability{89}); return;))
  NV_DISPATCH_TARGET(
    NV_IS_EXACTLY_SM_90,
    (assert(cc == cuda::compute_capability{90}); return;),
    NV_IS_EXACTLY_SM_100,
    (assert(cc == cuda::compute_capability{100}); return;),
    NV_IS_EXACTLY_SM_103,
    (assert(cc == cuda::compute_capability{103}); return;),
    NV_IS_EXACTLY_SM_110,
    (assert(cc == cuda::compute_capability{110}); return;),
    NV_IS_EXACTLY_SM_120,
    (assert(cc == cuda::compute_capability{120}); return;),
    NV_IS_EXACTLY_SM_121,
    (assert(cc == cuda::compute_capability{121}); return;),
    NV_ANY_TARGET,
    (assert(false);) // fail for unknown compute capability
  )
}

__host__ __device__ constexpr bool test()
{
  // 1. Test default constructor.
  {
    static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::compute_capability>);
    cuda::compute_capability cc;
    assert(cc.get() == 0);
  }

  // 2. Test constructor from compute capability in format 10 * major + minor.
  {
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::compute_capability, int>);
    static_assert(!cuda::std::is_convertible_v<int, cuda::compute_capability>);

    cuda::compute_capability cc{148};
    assert(cc.get() == 148);
  }

  // 3. Test constructor from major and minor.
  {
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::compute_capability, int, int>);
    cuda::compute_capability cc{8, 9};
    assert(cc.get() == 89);
  }

  // 4. Test constructor from cuda::arch_id.
  {
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::compute_capability, cuda::arch_id>);
    static_assert(!cuda::std::is_convertible_v<cuda::arch_id, cuda::compute_capability>);

    cuda::compute_capability cc1{cuda::arch_id::sm_100};
    assert(cc1.get() == 100);
    cuda::compute_capability cc2{cuda::arch_id::sm_100a};
    assert(cc2.get() == 100);
  }

  // 5. Test copy constructor.
  {
    static_assert(cuda::std::is_trivially_copy_constructible_v<cuda::compute_capability>);

    const cuda::compute_capability cc1{cuda::arch_id::sm_100};
    cuda::compute_capability cc2{cc1};
    assert(cc1.get() == 100);
    assert(cc2.get() == 100);
  }

  // 6. Test assignment operator.
  {
    static_assert(cuda::std::is_nothrow_copy_assignable_v<cuda::compute_capability>);

    const cuda::compute_capability cc1{cuda::arch_id::sm_100};
    cuda::compute_capability cc2;
    assert(cc1.get() == 100);
    assert(cc2.get() == 0);

    cc2 = cc1;
    assert(cc1.get() == 100);
    assert(cc2.get() == 100);
  }

  // 7. Test get().
  {
    static_assert(cuda::std::is_same_v<int, decltype(cuda::compute_capability{}.get())>);
    static_assert(noexcept(cuda::compute_capability{}.get()));

    const cuda::compute_capability cc{cuda::arch_id::sm_100};
    assert(cc.get() == 100);
  }

  // 8. Test major_cap().
  {
    static_assert(cuda::std::is_same_v<int, decltype(cuda::compute_capability{}.major_cap())>);
    static_assert(noexcept(cuda::compute_capability{}.major_cap()));

    const cuda::compute_capability cc{cuda::arch_id::sm_100};
    assert(cc.major_cap() == 10);

    // Test deprecated major().
    static_assert(cuda::std::is_same_v<int, decltype(cuda::compute_capability{}.major())>);
    static_assert(noexcept(cuda::compute_capability{}.major()));

    assert(cc.major() == cc.major_cap());
  }

  // 9. Test minor_cap().
  {
    static_assert(cuda::std::is_same_v<int, decltype(cuda::compute_capability{}.minor_cap())>);
    static_assert(noexcept(cuda::compute_capability{}.minor_cap()));

    const cuda::compute_capability cc{cuda::arch_id::sm_89};
    assert(cc.minor_cap() == 9);

    // Test deprecated minor().
    static_assert(cuda::std::is_same_v<int, decltype(cuda::compute_capability{}.minor())>);
    static_assert(noexcept(cuda::compute_capability{}.minor()));

    assert(cc.minor() == cc.minor_cap());
  }

  // 10. operator int()
  {
    static_assert(noexcept(static_cast<int>(cuda::compute_capability{})));
    static_assert(!cuda::std::is_convertible_v<cuda::compute_capability, int>);

    const cuda::compute_capability cc{cuda::arch_id::sm_89};
    assert(static_cast<int>(cc) == 89);
  }

  // 11. comparison operators
  {
    static_assert(
      cuda::std::is_same_v<bool, decltype(operator==(cuda::compute_capability{}, cuda::compute_capability{}))>);
    static_assert(
      cuda::std::is_same_v<bool, decltype(operator!=(cuda::compute_capability{}, cuda::compute_capability{}))>);
    static_assert(
      cuda::std::is_same_v<bool, decltype(operator<(cuda::compute_capability{}, cuda::compute_capability{}))>);
    static_assert(
      cuda::std::is_same_v<bool, decltype(operator<=(cuda::compute_capability{}, cuda::compute_capability{}))>);
    static_assert(
      cuda::std::is_same_v<bool, decltype(operator>(cuda::compute_capability{}, cuda::compute_capability{}))>);
    static_assert(
      cuda::std::is_same_v<bool, decltype(operator>=(cuda::compute_capability{}, cuda::compute_capability{}))>);

    static_assert(noexcept(operator==(cuda::compute_capability{}, cuda::compute_capability{})));
    static_assert(noexcept(operator!=(cuda::compute_capability{}, cuda::compute_capability{})));
    static_assert(noexcept(operator<(cuda::compute_capability{}, cuda::compute_capability{})));
    static_assert(noexcept(operator<=(cuda::compute_capability{}, cuda::compute_capability{})));
    static_assert(noexcept(operator>(cuda::compute_capability{}, cuda::compute_capability{})));
    static_assert(noexcept(operator>=(cuda::compute_capability{}, cuda::compute_capability{})));

    const cuda::compute_capability cc1{127};
    const cuda::compute_capability cc2{43};

    assert(cc1 == cc1);
    assert(cc2 == cc2);

    assert(cc1 != cc2);
    assert(cc2 != cc1);

    assert(!(cc1 < cc1));
    assert(!(cc1 < cc2));
    assert(!(cc2 < cc2));
    assert(cc2 < cc1);

    assert(cc1 <= cc1);
    assert(!(cc1 <= cc2));
    assert(cc2 <= cc2);
    assert(cc2 <= cc1);

    assert(!(cc1 > cc1));
    assert(cc1 > cc2);
    assert(!(cc2 > cc2));
    assert(!(cc2 > cc1));

    assert(cc1 >= cc1);
    assert(cc1 > cc2);
    assert(cc2 >= cc2);
    assert(!(cc2 > cc1));
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
