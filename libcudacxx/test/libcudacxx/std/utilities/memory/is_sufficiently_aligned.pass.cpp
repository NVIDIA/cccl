//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<size_t N, class T> bool is_sufficiently_aligned(T* ptr);
//
// Mandates: N is a power of two. N may be smaller than alignof(T).

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "test_macros.h"

struct alignas(32) OverAligned
{
  char data[32];
};

TEST_FUNC bool test()
{
  // alignment equal to or larger than alignof(T)
  auto p = reinterpret_cast<int*>(uintptr_t{12});
  assert(cuda::std::is_sufficiently_aligned<4>(p));
  assert(!cuda::std::is_sufficiently_aligned<8>(p));

  // alignment smaller than alignof(T) is always satisfied by a valid pointer
  auto q = reinterpret_cast<OverAligned*>(uintptr_t{32});
  assert(cuda::std::is_sufficiently_aligned<1>(q));
  assert(cuda::std::is_sufficiently_aligned<16>(q));
  assert(cuda::std::is_sufficiently_aligned<32>(q));
  assert(!cuda::std::is_sufficiently_aligned<64>(q));
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
