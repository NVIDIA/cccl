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
TEST_FUNC void test_is_trivially_destructible()
{
  static_assert(cuda::std::is_trivially_destructible<T>::value);
  static_assert(cuda::std::is_trivially_destructible<const T>::value);
  static_assert(cuda::std::is_trivially_destructible<volatile T>::value);
  static_assert(cuda::std::is_trivially_destructible<const volatile T>::value);
  static_assert(cuda::std::is_trivially_destructible_v<T>);
  static_assert(cuda::std::is_trivially_destructible_v<const T>);
  static_assert(cuda::std::is_trivially_destructible_v<volatile T>);
  static_assert(cuda::std::is_trivially_destructible_v<const volatile T>);
}

template <class T>
TEST_FUNC void test_is_not_trivially_destructible()
{
  static_assert(!cuda::std::is_trivially_destructible<T>::value);
  static_assert(!cuda::std::is_trivially_destructible<const T>::value);
  static_assert(!cuda::std::is_trivially_destructible<volatile T>::value);
  static_assert(!cuda::std::is_trivially_destructible<const volatile T>::value);
  static_assert(!cuda::std::is_trivially_destructible_v<T>);
  static_assert(!cuda::std::is_trivially_destructible_v<const T>);
  static_assert(!cuda::std::is_trivially_destructible_v<volatile T>);
  static_assert(!cuda::std::is_trivially_destructible_v<const volatile T>);
}

struct PublicDestructor
{
public:
  TEST_FUNC ~PublicDestructor() {}
};
struct ProtectedDestructor
{
protected:
  TEST_FUNC ~ProtectedDestructor() {}
};
struct PrivateDestructor
{
private:
  TEST_FUNC ~PrivateDestructor() {}
};

#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
struct VirtualPublicDestructor
{
public:
  TEST_FUNC virtual ~VirtualPublicDestructor() {}
};
struct VirtualProtectedDestructor
{
protected:
  TEST_FUNC virtual ~VirtualProtectedDestructor() {}
};
struct VirtualPrivateDestructor
{
private:
  TEST_FUNC virtual ~VirtualPrivateDestructor() {}
};

struct PurePublicDestructor
{
public:
  TEST_FUNC virtual ~PurePublicDestructor() = 0;
};
struct PureProtectedDestructor
{
protected:
  TEST_FUNC virtual ~PureProtectedDestructor() = 0;
};
struct PurePrivateDestructor
{
private:
  TEST_FUNC virtual ~PurePrivateDestructor() = 0;
};
#endif // !_CCCL_TILE_COMPILATION()

class Empty
{};

union Union
{};

struct bit_zero
{
  int : 0;
};

#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
class Abstract
{
  TEST_FUNC virtual void foo() = 0;
};

class AbstractDestructor
{
  TEST_FUNC virtual ~AbstractDestructor() = 0;
};
#endif // !_CCCL_TILE_COMPILATION()

struct A
{
  TEST_FUNC ~A();
};

int main(int, char**)
{
  test_is_not_trivially_destructible<void>();
  test_is_not_trivially_destructible<A>();
  test_is_not_trivially_destructible<char[]>();
#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
  test_is_not_trivially_destructible<VirtualPublicDestructor>();
  test_is_not_trivially_destructible<PurePublicDestructor>();

  test_is_trivially_destructible<Abstract>();
#endif // !_CCCL_TILE_COMPILATION()
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
#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
  test_is_not_trivially_destructible<VirtualProtectedDestructor>();
  test_is_not_trivially_destructible<VirtualPrivateDestructor>();
  test_is_not_trivially_destructible<PureProtectedDestructor>();
  test_is_not_trivially_destructible<PurePrivateDestructor>();
#endif // !_CCCL_TILE_COMPILATION()

#if !_CCCL_IS_IDENTIFIER(_Atomic)
  test_is_trivially_destructible<_Atomic int>();
  test_is_trivially_destructible<_Atomic float>();
  test_is_trivially_destructible<_Atomic int*>();
#endif

  return 0;
}
