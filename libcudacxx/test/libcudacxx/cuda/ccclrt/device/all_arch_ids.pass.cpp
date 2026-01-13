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

__host__ __device__ constexpr bool test()
{
  // 1. Test signature.
  static_assert(cuda::std::__is_cuda_std_array_v<decltype(cuda::__all_arch_ids())>);
  static_assert(noexcept(cuda::__all_arch_ids()));

  // 2. Test that all values are present.
  const auto all_arch_ids = cuda::__all_arch_ids();

  cuda::std::size_t i = 0;
  assert(all_arch_ids[i++] == cuda::arch_id::sm_60);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_61);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_62);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_70);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_75);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_80);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_86);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_87);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_88);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_89);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_90);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_100);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_103);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_110);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_120);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_121);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_90a);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_100a);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_103a);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_110a);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_120a);
  assert(all_arch_ids[i++] == cuda::arch_id::sm_121a);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
