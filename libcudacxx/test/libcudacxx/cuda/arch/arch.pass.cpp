//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/arch>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
  // Default constructor
  {
    static_assert(cuda::std::is_trivially_default_constructible_v<cuda::arch>);

    cuda::arch arch{};
    assert(arch.id() == cuda::arch_id{});
  }

  // Implicit constructor from cuda::arch_id
  {
    static_assert(cuda::std::is_nothrow_constructible_v<cuda::arch, cuda::arch_id>);
    static_assert(cuda::std::is_convertible_v<cuda::arch_id, cuda::arch>);

    cuda::arch arch{cuda::arch_id::sm_100};
    assert(arch.id() == cuda::arch_id::sm_100);
  }

  // id()
  {
    static_assert(cuda::std::is_same_v<cuda::arch_id, decltype(cuda::arch{}.id())>);
    static_assert(noexcept(cuda::arch{}.id()));

    cuda::arch arch{cuda::arch_id::sm_120};
    assert(arch.id() == cuda::arch_id::sm_120);
  }

  // is_arch_specific()
  {
    static_assert(cuda::std::is_same_v<bool, decltype(cuda::arch{}.is_arch_specific())>);
    static_assert(noexcept(cuda::arch{}.is_arch_specific()));

    assert(!cuda::arch{cuda::arch_id::sm_60}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_61}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_62}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_70}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_75}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_80}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_86}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_87}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_88}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_89}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_90}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_100}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_103}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_110}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_120}.is_arch_specific());
    assert(!cuda::arch{cuda::arch_id::sm_121}.is_arch_specific());
    // assert(!cuda::arch{cuda::arch_id::sm_100f}.is_arch_specific());
    // assert(!cuda::arch{cuda::arch_id::sm_103f}.is_arch_specific());
    // assert(!cuda::arch{cuda::arch_id::sm_110f}.is_arch_specific());
    // assert(!cuda::arch{cuda::arch_id::sm_120f}.is_arch_specific());
    // assert(!cuda::arch{cuda::arch_id::sm_121f}.is_arch_specific());
    assert(cuda::arch{cuda::arch_id::sm_90a}.is_arch_specific());
    assert(cuda::arch{cuda::arch_id::sm_100a}.is_arch_specific());
    assert(cuda::arch{cuda::arch_id::sm_103a}.is_arch_specific());
    assert(cuda::arch{cuda::arch_id::sm_110a}.is_arch_specific());
    assert(cuda::arch{cuda::arch_id::sm_120a}.is_arch_specific());
    assert(cuda::arch{cuda::arch_id::sm_121a}.is_arch_specific());
  }

  // is_family_specific()
  {
    static_assert(cuda::std::is_same_v<bool, decltype(cuda::arch{}.is_family_specific())>);
    static_assert(noexcept(cuda::arch{}.is_family_specific()));

    assert(!cuda::arch{cuda::arch_id::sm_60}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_61}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_62}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_70}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_75}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_80}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_86}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_87}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_88}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_89}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_90}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_100}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_103}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_110}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_120}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_121}.is_family_specific());
    // assert(cuda::arch{cuda::arch_id::sm_100f}.is_family_specific());
    // assert(cuda::arch{cuda::arch_id::sm_103f}.is_family_specific());
    // assert(cuda::arch{cuda::arch_id::sm_110f}.is_family_specific());
    // assert(cuda::arch{cuda::arch_id::sm_120f}.is_family_specific());
    // assert(cuda::arch{cuda::arch_id::sm_121f}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_90a}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_100a}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_103a}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_110a}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_120a}.is_family_specific());
    assert(!cuda::arch{cuda::arch_id::sm_121a}.is_family_specific());
  }

  // provides(arch)
  {
    static_assert(cuda::std::is_same_v<bool, decltype(cuda::arch{}.provides(cuda::arch{}))>);
    static_assert(noexcept(cuda::arch{}.provides(cuda::arch{})));

    cuda::arch arch1{cuda::arch_id::sm_103};
    assert(arch1.provides(cuda::arch_id::sm_60));
    assert(arch1.provides(cuda::arch_id::sm_100));
    assert(arch1.provides(cuda::arch_id::sm_103));
    assert(!arch1.provides(cuda::arch_id::sm_110));
    assert(!arch1.provides(cuda::arch_id::sm_121));

    // assert(!arch1.provides(cuda::arch_id::sm_100f));
    // assert(!arch1.provides(cuda::arch_id::sm_103f));
    // assert(!arch1.provides(cuda::arch_id::sm_110f));
    // assert(!arch1.provides(cuda::arch_id::sm_121f));

    assert(!arch1.provides(cuda::arch_id::sm_100a));
    assert(!arch1.provides(cuda::arch_id::sm_103a));
    assert(!arch1.provides(cuda::arch_id::sm_110a));
    assert(!arch1.provides(cuda::arch_id::sm_121a));

    // cuda::arch arch2{cuda::arch_id::sm_103f};
    // assert(arch2.provides(cuda::arch_id::sm_60));
    // assert(arch2.provides(cuda::arch_id::sm_100));
    // assert(arch2.provides(cuda::arch_id::sm_103));
    // assert(!arch2.provides(cuda::arch_id::sm_110));
    // assert(!arch2.provides(cuda::arch_id::sm_121));

    // assert(arch2.provides(cuda::arch_id::sm_100f));
    // assert(arch2.provides(cuda::arch_id::sm_103f));
    // assert(!arch2.provides(cuda::arch_id::sm_110f));
    // assert(!arch2.provides(cuda::arch_id::sm_121f));

    // assert(!arch2.provides(cuda::arch_id::sm_100a));
    // assert(!arch2.provides(cuda::arch_id::sm_103a));
    // assert(!arch2.provides(cuda::arch_id::sm_110a));
    // assert(!arch2.provides(cuda::arch_id::sm_121a));

    cuda::arch arch3{cuda::arch_id::sm_103a};
    assert(arch3.provides(cuda::arch_id::sm_60));
    assert(arch3.provides(cuda::arch_id::sm_100));
    assert(arch3.provides(cuda::arch_id::sm_103));
    assert(!arch3.provides(cuda::arch_id::sm_110));
    assert(!arch3.provides(cuda::arch_id::sm_121));

    // assert(arch3.provides(cuda::arch_id::sm_100f));
    // assert(arch3.provides(cuda::arch_id::sm_103f));
    // assert(!arch3.provides(cuda::arch_id::sm_110f));
    // assert(!arch3.provides(cuda::arch_id::sm_121f));

    assert(!arch3.provides(cuda::arch_id::sm_100a));
    assert(arch3.provides(cuda::arch_id::sm_103a));
    assert(!arch3.provides(cuda::arch_id::sm_110a));
    assert(!arch3.provides(cuda::arch_id::sm_121a));
  }

  // operator==
  {
    static_assert(cuda::std::is_same_v<bool, decltype(cuda::arch{} == cuda::arch{})>);
    static_assert(noexcept(cuda::arch{} == cuda::arch{}));

    cuda::arch_id arch{cuda::arch_id::sm_103};
    assert(arch == arch);
    assert(!(arch == cuda::arch_id::sm_100));
    assert(!(arch == cuda::arch_id::sm_110));
    // assert(!(arch == cuda::arch_id::sm_103f));
    // assert(!(arch == cuda::arch_id::sm_100f));
    // assert(!(arch == cuda::arch_id::sm_110f));
    assert(!(arch == cuda::arch_id::sm_103a));
    assert(!(arch == cuda::arch_id::sm_100a));
    assert(!(arch == cuda::arch_id::sm_110a));
  }

  // operator!=
  {
    static_assert(cuda::std::is_same_v<bool, decltype(cuda::arch{} != cuda::arch{})>);
    static_assert(noexcept(cuda::arch{} != cuda::arch{}));

    cuda::arch_id arch{cuda::arch_id::sm_103};
    assert(!(arch != arch));
    assert(arch != cuda::arch_id::sm_100);
    assert(arch != cuda::arch_id::sm_110);
    // assert(arch != cuda::arch_id::sm_103f);
    // assert(arch != cuda::arch_id::sm_100f);
    // assert(arch != cuda::arch_id::sm_110f);
    assert(arch != cuda::arch_id::sm_103a);
    assert(arch != cuda::arch_id::sm_100a);
    assert(arch != cuda::arch_id::sm_110a);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
