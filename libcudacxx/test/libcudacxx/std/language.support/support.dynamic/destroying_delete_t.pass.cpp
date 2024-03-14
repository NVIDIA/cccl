//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// struct destroying_delete_t {
//   explicit destroying_delete_t() = default;
// };
// inline constexpr destroying_delete_t destroying_delete{};

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <cuda/std/__new>
#include <cuda/std/cassert>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR bool A_constructed = false;
STATIC_TEST_GLOBAL_VAR bool A_destroyed = false;
STATIC_TEST_GLOBAL_VAR bool A_destroying_deleted = false;
struct A {
  void* data;
  __host__ __device__ A() { A_constructed = true; }
  __host__ __device__ ~A() { A_destroyed = true; }

  __host__ __device__ static A* New() {
    return ::new (::operator new(sizeof(A))) A();
  }
  __host__ __device__ void operator delete(A* a,
                                           cuda::std::destroying_delete_t) {
    A_destroying_deleted = true;
    ::operator delete(a);
  }
};

__host__ __device__ void test() {
  // Ensure that we call the destroying delete and not the destructor.
  A* ap = A::New();
  assert(A_constructed);
  delete ap;
  assert(!A_destroyed);
  assert(A_destroying_deleted);
}

int main(int, char**) {
  test();
  return 0;
}
