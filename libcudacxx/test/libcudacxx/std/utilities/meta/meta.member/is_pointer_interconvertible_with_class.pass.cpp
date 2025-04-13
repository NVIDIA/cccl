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
  int ma1;
  unsigned ma2;

  __host__ __device__ void fn() {}
};

struct B
{
  int mb1;
  unsigned mb2;

  __host__ __device__ void fn() {}
};

union U
{
  int mu1;
  unsigned mu2;
};

struct NonStandard
    : A
    , B
{
  virtual ~NonStandard() = default;

  int mns1;
};

__host__ __device__ constexpr bool test()
{
#if defined(_CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS)
  // 1. Only the first member of a class is pointer interconvertible with the class itself
  assert(cuda::std::is_pointer_interconvertible_with_class(&A::ma1));
  assert(cuda::std::is_pointer_interconvertible_with_class(&B::mb1));

  // 2. Rest of the members of a class are not pointer interconvertible with the class itself
  assert(!cuda::std::is_pointer_interconvertible_with_class(&A::ma2));
  assert(!cuda::std::is_pointer_interconvertible_with_class(&B::mb2));

  // 3. All union members are pointer interconvertible with the union itself
  assert(cuda::std::is_pointer_interconvertible_with_class(&U::mu1));
  assert(cuda::std::is_pointer_interconvertible_with_class(&U::mu2));

  // 4. Non-standard layout class members are not pointer interconvertible with the class itself
  assert(!cuda::std::is_pointer_interconvertible_with_class(&NonStandard::mns1));

  // 5. Member functions are not pointer interconvertible with the class itself
  assert(!cuda::std::is_pointer_interconvertible_with_class(&A::fn));
  assert(!cuda::std::is_pointer_interconvertible_with_class(&B::fn));

  // 7. is_pointer_interconvertible_with_class always returns false for nullptr
  assert(!cuda::std::is_pointer_interconvertible_with_class(static_cast<int A::*>(nullptr)));
  assert(!cuda::std::is_pointer_interconvertible_with_class(static_cast<int B::*>(nullptr)));
#endif // _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_WITH_CLASS

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
