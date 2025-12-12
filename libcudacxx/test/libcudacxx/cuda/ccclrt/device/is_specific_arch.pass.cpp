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
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

__host__ __device__ constexpr bool test()
{
  // 1. Test signature.
  static_assert(cuda::std::is_same_v<bool, decltype(cuda::__is_specific_arch(cuda::arch_id{}))>);
  static_assert(noexcept(cuda::__is_specific_arch(cuda::arch_id{})));

  // 2. Test values.
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_60));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_61));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_62));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_70));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_75));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_80));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_86));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_87));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_88));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_89));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_90));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_100));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_103));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_110));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_120));
  assert(!cuda::__is_specific_arch(cuda::arch_id::sm_121));
  assert(cuda::__is_specific_arch(cuda::arch_id::sm_90a));
  assert(cuda::__is_specific_arch(cuda::arch_id::sm_100a));
  assert(cuda::__is_specific_arch(cuda::arch_id::sm_103a));
  assert(cuda::__is_specific_arch(cuda::arch_id::sm_110a));
  assert(cuda::__is_specific_arch(cuda::arch_id::sm_120a));
  assert(cuda::__is_specific_arch(cuda::arch_id::sm_121a));

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
