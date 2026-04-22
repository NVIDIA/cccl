//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__arch_>
#include <cuda/std/cassert>

__host__ __device__ constexpr bool test()
{
  assert(cuda::arch::sm_60.id() == cuda::arch_id::sm_60);
  assert(cuda::arch::sm_61.id() == cuda::arch_id::sm_61);
  assert(cuda::arch::sm_62.id() == cuda::arch_id::sm_62);
  assert(cuda::arch::sm_70.id() == cuda::arch_id::sm_70);
  assert(cuda::arch::sm_75.id() == cuda::arch_id::sm_75);
  assert(cuda::arch::sm_80.id() == cuda::arch_id::sm_80);
  assert(cuda::arch::sm_86.id() == cuda::arch_id::sm_86);
  assert(cuda::arch::sm_87.id() == cuda::arch_id::sm_87);
  assert(cuda::arch::sm_88.id() == cuda::arch_id::sm_88);
  assert(cuda::arch::sm_89.id() == cuda::arch_id::sm_89);
  assert(cuda::arch::sm_90.id() == cuda::arch_id::sm_90);
  assert(cuda::arch::sm_100.id() == cuda::arch_id::sm_100);
  assert(cuda::arch::sm_103.id() == cuda::arch_id::sm_103);
  assert(cuda::arch::sm_110.id() == cuda::arch_id::sm_110);
  assert(cuda::arch::sm_120.id() == cuda::arch_id::sm_120);
  assert(cuda::arch::sm_121.id() == cuda::arch_id::sm_121);
  // assert(cuda::arch::sm_100f.id() == cuda::arch_id::sm_100f);
  // assert(cuda::arch::sm_103f.id() == cuda::arch_id::sm_103f);
  // assert(cuda::arch::sm_110f.id() == cuda::arch_id::sm_110f);
  // assert(cuda::arch::sm_120f.id() == cuda::arch_id::sm_120f);
  // assert(cuda::arch::sm_121f.id() == cuda::arch_id::sm_121f);
  assert(cuda::arch::sm_90a.id() == cuda::arch_id::sm_90a);
  assert(cuda::arch::sm_100a.id() == cuda::arch_id::sm_100a);
  assert(cuda::arch::sm_103a.id() == cuda::arch_id::sm_103a);
  assert(cuda::arch::sm_110a.id() == cuda::arch_id::sm_110a);
  assert(cuda::arch::sm_120a.id() == cuda::arch_id::sm_120a);
  assert(cuda::arch::sm_121a.id() == cuda::arch_id::sm_121a);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
