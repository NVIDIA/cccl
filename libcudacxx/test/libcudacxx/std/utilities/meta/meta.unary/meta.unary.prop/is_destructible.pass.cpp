//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_destructible

#include <cuda/std/type_traits>

#include "test_macros.h"

// Prevent warning when testing the Abstract test type.
TEST_DIAG_SUPPRESS_CLANG("-Wdelete-non-virtual-dtor")

template <class T>
TEST_FUNC void test_is_destructible()
{
  static_assert(cuda::std::is_destructible<T>::value);
  static_assert(cuda::std::is_destructible<const T>::value);
  static_assert(cuda::std::is_destructible<volatile T>::value);
  static_assert(cuda::std::is_destructible<const volatile T>::value);
  static_assert(cuda::std::is_destructible_v<T>);
  static_assert(cuda::std::is_destructible_v<const T>);
  static_assert(cuda::std::is_destructible_v<volatile T>);
  static_assert(cuda::std::is_destructible_v<const volatile T>);
}

template <class T>
TEST_FUNC void test_is_not_destructible()
{
  static_assert(!cuda::std::is_destructible<T>::value);
  static_assert(!cuda::std::is_destructible<const T>::value);
  static_assert(!cuda::std::is_destructible<volatile T>::value);
  static_assert(!cuda::std::is_destructible<const volatile T>::value);
  static_assert(!cuda::std::is_destructible_v<T>);
  static_assert(!cuda::std::is_destructible_v<const T>);
  static_assert(!cuda::std::is_destructible_v<volatile T>);
  static_assert(!cuda::std::is_destructible_v<const volatile T>);
}

class Empty
{};

#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
class NotEmpty
{
  TEST_FUNC virtual ~NotEmpty();
};
#endif // !_CCCL_TILE_COMPILATION()

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  TEST_FUNC ~A();
};

using Function = void();

#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
struct PublicAbstract
{
public:
  TEST_FUNC virtual void foo() = 0;
};
struct ProtectedAbstract
{
protected:
  TEST_FUNC virtual void foo() = 0;
};
struct PrivateAbstract
{
private:
  TEST_FUNC virtual void foo() = 0;
};
#endif // !_CCCL_TILE_COMPILATION()

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

struct DeletedPublicDestructor
{
public:
  TEST_FUNC ~DeletedPublicDestructor() = delete;
};
struct DeletedProtectedDestructor
{
protected:
  TEST_FUNC ~DeletedProtectedDestructor() = delete;
};
struct DeletedPrivateDestructor
{
private:
  TEST_FUNC ~DeletedPrivateDestructor() = delete;
};

#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
struct DeletedVirtualPublicDestructor
{
public:
  TEST_FUNC virtual ~DeletedVirtualPublicDestructor() = delete;
};
struct DeletedVirtualProtectedDestructor
{
protected:
  TEST_FUNC virtual ~DeletedVirtualProtectedDestructor() = delete;
};
struct DeletedVirtualPrivateDestructor
{
private:
  TEST_FUNC virtual ~DeletedVirtualPrivateDestructor() = delete;
};
#endif // !_CCCL_TILE_COMPILATION()

int main(int, char**)
{
  test_is_destructible<A>();
  test_is_destructible<int&>();
  test_is_destructible<Union>();
  test_is_destructible<Empty>();
  test_is_destructible<int>();
  test_is_destructible<double>();
  test_is_destructible<int*>();
  test_is_destructible<const int*>();
  test_is_destructible<char[3]>();
  test_is_destructible<bit_zero>();
  test_is_destructible<int[3]>();
  test_is_destructible<PublicDestructor>();
#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
  test_is_destructible<ProtectedAbstract>();
  test_is_destructible<PublicAbstract>();
  test_is_destructible<PrivateAbstract>();
  test_is_destructible<VirtualPublicDestructor>();
  test_is_destructible<PurePublicDestructor>();
#endif // !_CCCL_TILE_COMPILATION()

  test_is_not_destructible<int[]>();
  test_is_not_destructible<void>();
  test_is_not_destructible<Function>();

  // Test access controlled destructors
  test_is_not_destructible<ProtectedDestructor>();
  test_is_not_destructible<PrivateDestructor>();
#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
  test_is_not_destructible<VirtualProtectedDestructor>();
  test_is_not_destructible<VirtualPrivateDestructor>();
  test_is_not_destructible<PureProtectedDestructor>();
  test_is_not_destructible<PurePrivateDestructor>();
#endif // !_CCCL_TILE_COMPILATION()

  // Test deleted constructors
  test_is_not_destructible<DeletedPublicDestructor>();
  test_is_not_destructible<DeletedProtectedDestructor>();
  test_is_not_destructible<DeletedPrivateDestructor>();

#if !_CCCL_TILE_COMPILATION() // error: virtual function is unsupported in tile code
  // test_is_not_destructible<DeletedVirtualPublicDestructor>(); // previously failed due to clang bug #20268
  test_is_not_destructible<DeletedVirtualProtectedDestructor>();
  test_is_not_destructible<DeletedVirtualPrivateDestructor>();

  // Test private destructors
  test_is_not_destructible<NotEmpty>();
#endif // !_CCCL_TILE_COMPILATION()

  return 0;
}
