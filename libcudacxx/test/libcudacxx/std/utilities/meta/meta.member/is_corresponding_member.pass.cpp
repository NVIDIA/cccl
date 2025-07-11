//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

struct A
{
  int m1;
  unsigned m2;
  float m3;
  double m4;

  __host__ __device__ void fn() {}
};

struct B
{
  int m1;
  unsigned m2;
  float m3;
  double m4;

  __host__ __device__ void fn() {}
};

struct NonStandard
    : A
    , B
{
  virtual ~NonStandard() = default;

  int m;
};

__host__ __device__ constexpr bool test()
{
#if defined(_CCCL_BUILTIN_IS_CORRESPONDING_MEMBER)
  // 1. Test struct A members to be corresponding with itself
  assert(cuda::std::is_corresponding_member(&A::m1, &A::m1));
  assert(cuda::std::is_corresponding_member(&A::m2, &A::m2));
  assert(cuda::std::is_corresponding_member(&A::m3, &A::m3));
  assert(cuda::std::is_corresponding_member(&A::m4, &A::m4));

  // 2. Test struct A members to be corresponding with struct B members
  assert(cuda::std::is_corresponding_member(&A::m1, &B::m1));
  assert(cuda::std::is_corresponding_member(&A::m2, &B::m2));
  assert(cuda::std::is_corresponding_member(&A::m3, &B::m3));
  assert(cuda::std::is_corresponding_member(&A::m4, &B::m4));

  // 3. Test struct A members not to be corresponding with each other
  assert(!cuda::std::is_corresponding_member(&A::m1, &A::m2));
  assert(!cuda::std::is_corresponding_member(&A::m1, &A::m3));
  assert(!cuda::std::is_corresponding_member(&A::m1, &A::m4));
  assert(!cuda::std::is_corresponding_member(&A::m2, &A::m1));
  assert(!cuda::std::is_corresponding_member(&A::m2, &A::m3));
  assert(!cuda::std::is_corresponding_member(&A::m2, &A::m4));
  assert(!cuda::std::is_corresponding_member(&A::m3, &A::m1));
  assert(!cuda::std::is_corresponding_member(&A::m3, &A::m2));
  assert(!cuda::std::is_corresponding_member(&A::m3, &A::m4));
  assert(!cuda::std::is_corresponding_member(&A::m4, &A::m1));
  assert(!cuda::std::is_corresponding_member(&A::m4, &A::m2));
  assert(!cuda::std::is_corresponding_member(&A::m4, &A::m3));

  // 4. Member functions should not be corresponding
  assert(!cuda::std::is_corresponding_member(&A::fn, &A::fn));

  // 5. If nullptr is passed, it should not be corresponding
  assert(!cuda::std::is_corresponding_member(static_cast<int A::*>(nullptr), static_cast<int A::*>(nullptr)));
  assert(!cuda::std::is_corresponding_member(&A::m1, static_cast<int A::*>(nullptr)));
  assert(!cuda::std::is_corresponding_member(static_cast<int A::*>(nullptr), &A::m1));

  // 6. Non-standard layout types always return false
  assert(!cuda::std::is_corresponding_member(&NonStandard::m, &NonStandard::m));
#endif // _CCCL_BUILTIN_IS_CORRESPONDING_MEMBER

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
