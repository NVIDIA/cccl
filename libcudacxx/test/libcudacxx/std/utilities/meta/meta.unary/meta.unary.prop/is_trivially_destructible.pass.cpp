//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_trivially_destructible

#include <cuda/std/type_traits>

#include "test_macros.h"

// Prevent warning when testing the Abstract test type.
TEST_DIAG_SUPPRESS_CLANG("-Wdelete-non-virtual-dtor")

template <class T>
__host__ __device__ void test_is_trivially_destructible()
{
  static_assert(cuda::std::is_trivially_destructible<T>::value, "");
  static_assert(cuda::std::is_trivially_destructible<const T>::value, "");
  static_assert(cuda::std::is_trivially_destructible<volatile T>::value, "");
  static_assert(cuda::std::is_trivially_destructible<const volatile T>::value, "");
  static_assert(cuda::std::is_trivially_destructible_v<T>, "");
  static_assert(cuda::std::is_trivially_destructible_v<const T>, "");
  static_assert(cuda::std::is_trivially_destructible_v<volatile T>, "");
  static_assert(cuda::std::is_trivially_destructible_v<const volatile T>, "");
}

template <class T>
__host__ __device__ void test_is_not_trivially_destructible()
{
  static_assert(!cuda::std::is_trivially_destructible<T>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<const T>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<volatile T>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<const volatile T>::value, "");
  static_assert(!cuda::std::is_trivially_destructible_v<T>, "");
  static_assert(!cuda::std::is_trivially_destructible_v<const T>, "");
  static_assert(!cuda::std::is_trivially_destructible_v<volatile T>, "");
  static_assert(!cuda::std::is_trivially_destructible_v<const volatile T>, "");
}

struct PublicDestructor
{
public:
  __host__ __device__ ~PublicDestructor() {}
};
struct ProtectedDestructor
{
protected:
  __host__ __device__ ~ProtectedDestructor() {}
};
struct PrivateDestructor
{
private:
  __host__ __device__ ~PrivateDestructor() {}
};

struct VirtualPublicDestructor
{
public:
  __host__ __device__ virtual ~VirtualPublicDestructor() {}
};
struct VirtualProtectedDestructor
{
protected:
  __host__ __device__ virtual ~VirtualProtectedDestructor() {}
};
struct VirtualPrivateDestructor
{
private:
  __host__ __device__ virtual ~VirtualPrivateDestructor() {}
};

struct PurePublicDestructor
{
public:
  __host__ __device__ virtual ~PurePublicDestructor() = 0;
};
struct PureProtectedDestructor
{
protected:
  __host__ __device__ virtual ~PureProtectedDestructor() = 0;
};
struct PurePrivateDestructor
{
private:
  __host__ __device__ virtual ~PurePrivateDestructor() = 0;
};

class Empty
{};

union Union
{};

struct bit_zero
{
  int : 0;
};

class Abstract
{
  __host__ __device__ virtual void foo() = 0;
};

class AbstractDestructor
{
  __host__ __device__ virtual ~AbstractDestructor() = 0;
};

struct A
{
  __host__ __device__ ~A();
};

int main(int, char**)
{
  test_is_not_trivially_destructible<void>();
  test_is_not_trivially_destructible<A>();
  test_is_not_trivially_destructible<char[]>();
  test_is_not_trivially_destructible<VirtualPublicDestructor>();
  test_is_not_trivially_destructible<PurePublicDestructor>();

  test_is_trivially_destructible<Abstract>();
  test_is_trivially_destructible<Union>();
  test_is_trivially_destructible<Empty>();
  test_is_trivially_destructible<int&>();
  test_is_trivially_destructible<int>();
  test_is_trivially_destructible<double>();
  test_is_trivially_destructible<int*>();
  test_is_trivially_destructible<const int*>();
  test_is_trivially_destructible<char[3]>();
  test_is_trivially_destructible<bit_zero>();

  // requires access control sfinae
  test_is_not_trivially_destructible<ProtectedDestructor>();
  test_is_not_trivially_destructible<PrivateDestructor>();
  test_is_not_trivially_destructible<VirtualProtectedDestructor>();
  test_is_not_trivially_destructible<VirtualPrivateDestructor>();
  test_is_not_trivially_destructible<PureProtectedDestructor>();
  test_is_not_trivially_destructible<PurePrivateDestructor>();

#if !_CCCL_IS_IDENTIFIER(_Atomic)
  test_is_trivially_destructible<_Atomic int>();
  test_is_trivially_destructible<_Atomic float>();
  test_is_trivially_destructible<_Atomic int*>();
#endif

  return 0;
}
