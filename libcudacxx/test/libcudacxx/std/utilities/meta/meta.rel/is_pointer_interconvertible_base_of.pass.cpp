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

struct BaseA
{};

struct BaseB
{};

struct _CCCL_DECLSPEC_EMPTY_BASES A : BaseA
{};

struct _CCCL_DECLSPEC_EMPTY_BASES B : BaseB
{};

struct _CCCL_DECLSPEC_EMPTY_BASES C
    : A
    , B
{
  float m;
};

struct _CCCL_DECLSPEC_EMPTY_BASES NonStandard
    : BaseA
    , BaseB
{
  virtual ~NonStandard() = default;

  int m;
};

template <class T, class U>
__host__ __device__ constexpr void test_is_pointer_interconvertible_base_of(bool expected)
{
#if defined(_CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF)
  assert((cuda::std::is_pointer_interconvertible_base_of<T, U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<T, const U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<T, volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<T, const volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const T, U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const T, const U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const T, volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const T, const volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<volatile T, U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<volatile T, const U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<volatile T, volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<volatile T, const volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const volatile T, U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const volatile T, const U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const volatile T, volatile U>::value == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of<const volatile T, const volatile U>::value == expected));

  assert((cuda::std::is_pointer_interconvertible_base_of_v<T, U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<T, const U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<T, volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<T, const volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const T, U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const T, const U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const T, volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const T, const volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<volatile T, U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<volatile T, const U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<volatile T, volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<volatile T, const volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const volatile T, U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const volatile T, const U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const volatile T, volatile U> == expected));
  assert((cuda::std::is_pointer_interconvertible_base_of_v<const volatile T, const volatile U> == expected));
#endif // _CCCL_BUILTIN_IS_POINTER_INTERCONVERTIBLE_BASE_OF
}

__host__ __device__ constexpr bool test()
{
  // 1. Structs have pointer-interconvertible
  test_is_pointer_interconvertible_base_of<BaseA, BaseA>(true);
  test_is_pointer_interconvertible_base_of<BaseB, BaseB>(true);
  test_is_pointer_interconvertible_base_of<A, A>(true);
  test_is_pointer_interconvertible_base_of<B, B>(true);
  test_is_pointer_interconvertible_base_of<C, C>(true);

  // 2. Test derived classes to be pointer-interconvertible with base classes
  test_is_pointer_interconvertible_base_of<BaseA, A>(true);
  test_is_pointer_interconvertible_base_of<BaseB, B>(true);
  test_is_pointer_interconvertible_base_of<BaseA, C>(true);
  test_is_pointer_interconvertible_base_of<BaseB, C>(true);
  test_is_pointer_interconvertible_base_of<A, C>(true);
  test_is_pointer_interconvertible_base_of<B, C>(true);

  // 3. Test combinations returning false
  test_is_pointer_interconvertible_base_of<A, B>(false);
  test_is_pointer_interconvertible_base_of<B, A>(false);
  test_is_pointer_interconvertible_base_of<C, NonStandard>(false);
  test_is_pointer_interconvertible_base_of<NonStandard, C>(false);
  test_is_pointer_interconvertible_base_of<int, int>(false);
  test_is_pointer_interconvertible_base_of<int, A>(false);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
