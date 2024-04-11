//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_destructible

// Prevent warning when testing the Abstract test type.
#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wdelete-non-virtual-dtor"
#endif

#include <cuda/std/type_traits>

#include "test_macros.h"

template <class T>
__host__ __device__ void test_is_destructible()
{
  static_assert(cuda::std::is_destructible<T>::value, "");
  static_assert(cuda::std::is_destructible<const T>::value, "");
  static_assert(cuda::std::is_destructible<volatile T>::value, "");
  static_assert(cuda::std::is_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(cuda::std::is_destructible_v<T>, "");
  static_assert(cuda::std::is_destructible_v<const T>, "");
  static_assert(cuda::std::is_destructible_v<volatile T>, "");
  static_assert(cuda::std::is_destructible_v<const volatile T>, "");
#endif
}

template <class T>
__host__ __device__ void test_is_not_destructible()
{
  static_assert(!cuda::std::is_destructible<T>::value, "");
  static_assert(!cuda::std::is_destructible<const T>::value, "");
  static_assert(!cuda::std::is_destructible<volatile T>::value, "");
  static_assert(!cuda::std::is_destructible<const volatile T>::value, "");
#if TEST_STD_VER > 2011
  static_assert(!cuda::std::is_destructible_v<T>, "");
  static_assert(!cuda::std::is_destructible_v<const T>, "");
  static_assert(!cuda::std::is_destructible_v<volatile T>, "");
  static_assert(!cuda::std::is_destructible_v<const volatile T>, "");
#endif
}

class Empty
{};

class NotEmpty
{
  __host__ __device__ virtual ~NotEmpty();
};

union Union
{};

struct bit_zero
{
  int : 0;
};

struct A
{
  __host__ __device__ ~A();
};

typedef void(Function)();

struct PublicAbstract
{
public:
  __host__ __device__ virtual void foo() = 0;
};
struct ProtectedAbstract
{
protected:
  __host__ __device__ virtual void foo() = 0;
};
struct PrivateAbstract
{
private:
  __host__ __device__ virtual void foo() = 0;
};

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

struct DeletedPublicDestructor
{
public:
  __host__ __device__ ~DeletedPublicDestructor() = delete;
};
struct DeletedProtectedDestructor
{
protected:
  __host__ __device__ ~DeletedProtectedDestructor() = delete;
};
struct DeletedPrivateDestructor
{
private:
  __host__ __device__ ~DeletedPrivateDestructor() = delete;
};

struct DeletedVirtualPublicDestructor
{
public:
  __host__ __device__ virtual ~DeletedVirtualPublicDestructor() = delete;
};
struct DeletedVirtualProtectedDestructor
{
protected:
  __host__ __device__ virtual ~DeletedVirtualProtectedDestructor() = delete;
};
struct DeletedVirtualPrivateDestructor
{
private:
  __host__ __device__ virtual ~DeletedVirtualPrivateDestructor() = delete;
};

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
  test_is_destructible<ProtectedAbstract>();
  test_is_destructible<PublicAbstract>();
  test_is_destructible<PrivateAbstract>();
  test_is_destructible<PublicDestructor>();
  test_is_destructible<VirtualPublicDestructor>();
  test_is_destructible<PurePublicDestructor>();

  test_is_not_destructible<int[]>();
  test_is_not_destructible<void>();
  test_is_not_destructible<Function>();

  // Test access controlled destructors
  test_is_not_destructible<ProtectedDestructor>();
  test_is_not_destructible<PrivateDestructor>();
  test_is_not_destructible<VirtualProtectedDestructor>();
  test_is_not_destructible<VirtualPrivateDestructor>();
  test_is_not_destructible<PureProtectedDestructor>();
  test_is_not_destructible<PurePrivateDestructor>();

  // Test deleted constructors
  test_is_not_destructible<DeletedPublicDestructor>();
  test_is_not_destructible<DeletedProtectedDestructor>();
  test_is_not_destructible<DeletedPrivateDestructor>();
  // test_is_not_destructible<DeletedVirtualPublicDestructor>(); // previously failed due to clang bug #20268
  test_is_not_destructible<DeletedVirtualProtectedDestructor>();
  test_is_not_destructible<DeletedVirtualPrivateDestructor>();

  // Test private destructors
  test_is_not_destructible<NotEmpty>();

  return 0;
}
