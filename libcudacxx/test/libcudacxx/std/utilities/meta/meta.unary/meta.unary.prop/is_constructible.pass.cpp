//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// template <class T, class... Args>
//   struct is_constructible;

// MODULES_DEFINES: _LIBCUDACXX_TESTING_FALLBACK_IS_CONSTRUCTIBLE
#define _LIBCUDACXX_TESTING_FALLBACK_IS_CONSTRUCTIBLE
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(declared_but_not_referenced)

struct A
{
  __host__ __device__ explicit A(int);
  __host__ __device__ A(int, double);
  __host__ __device__ A(int, long, double);

private:
  __host__ __device__ A(char);
};

struct Base
{};
struct Derived : public Base
{};

class Abstract
{
  __host__ __device__ virtual void foo() = 0;
};

class AbstractDestructor
{
  __host__ __device__ virtual ~AbstractDestructor() = 0;
};

struct PrivateDtor
{
  __host__ __device__ PrivateDtor(int) {}

private:
  __host__ __device__ ~PrivateDtor() {}
};

struct S
{
  template <class T>
  __host__ __device__ explicit operator T() const;
};

template <class To>
struct ImplicitTo
{
  __host__ __device__ operator To();
};

template <class To>
struct ExplicitTo
{
  __host__ __device__ explicit operator To();
};

template <class T>
__host__ __device__ void test_is_constructible()
{
  static_assert((cuda::std::is_constructible<T>::value), "");
#if !TEST_COMPILER(MSVC)
  // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
  // use it, so waive it.
  static_assert((cuda::std::__cccl_is_constructible<T>::type::value), "");
#endif // !TEST_COMPILER(MSVC)
  static_assert(cuda::std::is_constructible_v<T>, "");
}

template <class T, class A0>
__host__ __device__ void test_is_constructible()
{
  static_assert((cuda::std::is_constructible<T, A0>::value), "");
#if !TEST_COMPILER(MSVC)
  // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
  // use it, so waive it.
  static_assert((cuda::std::__cccl_is_constructible<T, A0>::type::value), "");
#endif // !TEST_COMPILER(MSVC)
  static_assert((cuda::std::is_constructible_v<T, A0>), "");
}

template <class T, class A0, class A1>
__host__ __device__ void test_is_constructible()
{
  static_assert((cuda::std::is_constructible<T, A0, A1>::value), "");
#if !TEST_COMPILER(MSVC)
  // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
  // use it, so waive it.
  static_assert((cuda::std::__cccl_is_constructible<T, A0, A1>::type::value), "");
#endif // !TEST_COMPILER(MSVC)
  static_assert((cuda::std::is_constructible_v<T, A0, A1>), "");
}

template <class T, class A0, class A1, class A2>
__host__ __device__ void test_is_constructible()
{
  static_assert((cuda::std::is_constructible<T, A0, A1, A2>::value), "");
#if !TEST_COMPILER(MSVC)
  // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
  // use it, so waive it.
  static_assert((cuda::std::__cccl_is_constructible<T, A0, A1, A2>::type::value), "");
#endif // !TEST_COMPILER(MSVC)
  static_assert((cuda::std::is_constructible_v<T, A0, A1, A2>), "");
}

template <class T>
__host__ __device__ void test_is_not_constructible()
{
  static_assert((!cuda::std::is_constructible<T>::value), "");
#if !TEST_COMPILER(MSVC)
  // The fallback SFINAE version doesn't work reliable with MSVC, and we don't
  // use it, so waive it.
  static_assert((!cuda::std::__cccl_is_constructible<T>::type::value), "");
#endif // !TEST_COMPILER(MSVC)
  static_assert((!cuda::std::is_constructible_v<T>), "");
}

template <class T, class A0>
__host__ __device__ void test_is_not_constructible()
{
  static_assert((!cuda::std::is_constructible<T, A0>::value), "");
#if !TEST_COMPILER(MSVC) && !TEST_COMPILER(CLANG) && !TEST_COMPILER(NVRTC) && !TEST_COMPILER(NVHPC)
  // The fallback SFINAE version doesn't work reliable with Clang/MSVC/NVRTC/NVHPC, and we don't
  // use it, so waive it.
  static_assert((!cuda::std::__cccl_is_constructible<T, A0>::type::value), "");
#endif // !TEST_COMPILER(MSVC) && !TEST_COMPILER(CLANG) && !TEST_COMPILER(NVRTC) && !TEST_COMPILER(NVHPC)
  static_assert((!cuda::std::is_constructible_v<T, A0>), "");
}

#if TEST_COMPILER(CLANG)
template <class T = int, class = decltype(static_cast<T&&>(cuda::std::declval<double&>()))>
__host__ __device__ constexpr bool clang_disallows_valid_static_cast_test(int)
{
  return false;
};

__host__ __device__ constexpr bool clang_disallows_valid_static_cast_test(long)
{
  return true;
}

static constexpr bool clang_disallows_valid_static_cast_bug = clang_disallows_valid_static_cast_test(0);
#endif // TEST_COMPILER(CLANG)

int main(int, char**)
{
  typedef Base B;
  typedef Derived D;

  test_is_constructible<int>();
  test_is_constructible<int, const int>();
  test_is_constructible<A, int>();
  test_is_constructible<A, int, double>();
  test_is_constructible<A, int, long, double>();
  test_is_constructible<int&, int&>();

  test_is_not_constructible<A>();
  test_is_not_constructible<A, char>();
  test_is_not_constructible<A, void>();
  test_is_not_constructible<int, void()>();
  test_is_not_constructible<int, void (&)()>();
  test_is_not_constructible<int, void() const>();
  test_is_not_constructible<int&, void>();
  test_is_not_constructible<int&, void()>();
  test_is_not_constructible<int&, void() const>();
  test_is_not_constructible<int&, void (&)()>();

  test_is_not_constructible<void>();
  test_is_not_constructible<const void>(); // LWG 2738
  test_is_not_constructible<volatile void>();
  test_is_not_constructible<const volatile void>();
  test_is_not_constructible<int&>();
  test_is_not_constructible<Abstract>();
  test_is_not_constructible<AbstractDestructor>();
  test_is_constructible<int, S>();
  test_is_not_constructible<int&, S>();

  test_is_constructible<void (&)(), void (&)()>();
  test_is_constructible<void (&)(), void()>();
  test_is_constructible<void (&&)(), void (&&)()>();
  test_is_constructible<void (&&)(), void()>();
  test_is_constructible<void (&&)(), void (&)()>();

  test_is_constructible<int const&, int>();
  test_is_constructible<int const&, int&&>();

#if !TEST_COMPILER(MSVC)
  // This appears to be an MSVC bug, it reproduces with their standard library
  // in 19.20:
  // https://godbolt.org/z/X455LN
  // https://developercommunity.visualstudio.com/content/problem/679848/stdis-constructible-v-incorrectly-returns-false-wi.html
  test_is_constructible<int&&, double&>();
#endif // !TEST_COMPILER(MSVC)
  test_is_constructible<void (&)(), void (&&)()>();

  test_is_not_constructible<int&, int>();
  test_is_not_constructible<int&, int const&>();
  test_is_not_constructible<int&, int&&>();

  test_is_constructible<int&&, int>();
  test_is_constructible<int&&, int&&>();
  test_is_not_constructible<int&&, int&>();
  test_is_not_constructible<int&&, int const&&>();

  test_is_constructible<Base, Derived>();
  test_is_constructible<Base&, Derived&>();
#if !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  test_is_not_constructible<Derived&, Base&>();
#endif // !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  test_is_constructible<Base const&, Derived const&>();
#if !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  test_is_not_constructible<Derived const&, Base const&>();
  test_is_not_constructible<Derived const&, Base>();
#endif // !TEST_COMPILER(GCC) || TEST_STD_VER < 2020

  test_is_constructible<Base&&, Derived>();
  test_is_constructible<Base&&, Derived&&>();
#if !TEST_COMPILER(GCC) || TEST_STD_VER < 2020
  test_is_not_constructible<Derived&&, Base&&>();
  test_is_not_constructible<Derived&&, Base>();
#endif // !TEST_COMPILER(GCC) || TEST_STD_VER < 2020

  // test that T must also be destructible
  test_is_constructible<PrivateDtor&, PrivateDtor&>();
  test_is_not_constructible<PrivateDtor, int>();

  test_is_not_constructible<void() const, void() const>();
  test_is_not_constructible<void() const, void*>();

  test_is_constructible<int&, ImplicitTo<int&>>();
  test_is_constructible<const int&, ImplicitTo<int&&>>();
  test_is_constructible<int&&, ImplicitTo<int&&>>();
  test_is_constructible<const int&, ImplicitTo<int>>();

  test_is_not_constructible<B&&, B&>();
  test_is_not_constructible<B&&, D&>();
  test_is_constructible<B&&, ImplicitTo<D&&>>();
  test_is_constructible<B&&, ImplicitTo<D&&>&>();

  test_is_constructible<const int&, ImplicitTo<int&>&>();
  test_is_constructible<const int&, ImplicitTo<int&>>();
  test_is_constructible<const int&, ExplicitTo<int&>&>();
  test_is_constructible<const int&, ExplicitTo<int&>>();

  test_is_constructible<const int&, ExplicitTo<int&>&>();
  test_is_constructible<const int&, ExplicitTo<int&>>();

  // Binding through reference-compatible type is required to perform
  // direct-initialization as described in [over.match.ref] p. 1 b. 1:
  //
  // But the rvalue to lvalue reference binding isn't allowed according to
  // [over.match.ref] despite Clang accepting it.
  test_is_constructible<int&, ExplicitTo<int&>>();

  // This fails almost everywhere.
  // test_is_constructible<const int&, ExplicitTo<int&&>>();

  // TODO add nvbug tracking
#if !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC)
  static_assert(cuda::std::is_constructible<int&&, ExplicitTo<int&&>>::value, "");
#endif // !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC)

#if TEST_CUDA_COMPILER(CLANG)
  static_assert(clang_disallows_valid_static_cast_bug == false, "");
  static_assert(cuda::std::__cccl_is_constructible<int&&, ExplicitTo<int&&>>::value, "");
#endif // TEST_CUDA_COMPILER(CLANG)

// FIXME Compilers disagree about the validity of these tests.
#if TEST_CUDA_COMPILER(CLANG)
  test_is_constructible<const int&, ExplicitTo<int>>();
  static_assert(
    clang_disallows_valid_static_cast_bug != cuda::std::__cccl_is_constructible<int&&, ExplicitTo<int>>::value, "");
  static_assert(cuda::std::is_constructible<int&&, ExplicitTo<int>>::value, "");
#elif TEST_COMPILER(MSVC) && TEST_CUDA_COMPILER(NVCC)
  // FIXME NVCC and MSVC disagree about the validity of these tests, and give
  //       different answers in host and device code, which is just wonderful.
#elif TEST_COMPILER(CLANG) && TEST_CUDA_COMPILER(NVCC)
  // FIXME NVCC fails the assertion below when used with clang.
#elif TEST_COMPILER(NVRTC) || TEST_COMPILER(NVHPC)
  // FIXME NVRTC also doesn't like these tests.
  // FIXME neither does NVCC+nvhpc.
#elif !TEST_COMPILER(GCC)
  // This fails with gcc, as the intrinsic differs from other compilers
  test_is_not_constructible<const int&, ExplicitTo<int>>();
  test_is_not_constructible<int&&, ExplicitTo<int>>();
#endif

  // Binding through temporary behaves like copy-initialization,
  // see [dcl.init.ref] p. 5, very last sub-bullet:
  test_is_not_constructible<const int&, ExplicitTo<double&&>>();
  test_is_not_constructible<int&&, ExplicitTo<double&&>>();
  test_is_not_constructible<void()>();
  test_is_not_constructible<void() const>();
  test_is_not_constructible<void() volatile>();
  test_is_not_constructible<void()&>();
  test_is_not_constructible<void() &&>();

  return 0;
}
