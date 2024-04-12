//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// type_traits

// is_nothrow_invocable

#include <cuda/std/functional>
#include <cuda/std/type_traits>
#ifdef _LIBCUDACXX_HAS_VECTOR
#  include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR

#include "test_macros.h"

struct Tag
{};

struct Implicit
{
  __host__ __device__ Implicit(int) noexcept {}
};

struct ThrowsImplicit
{
  __host__ __device__ ThrowsImplicit(int) {}
};

struct Explicit
{
  __host__ __device__ explicit Explicit(int) noexcept {}
};

template <bool IsNoexcept, class Ret, class... Args>
struct CallObject
{
  __host__ __device__ Ret operator()(Args&&...) const noexcept(IsNoexcept);
};

struct Sink
{
  template <class... Args>
  __host__ __device__ void operator()(Args&&...) const noexcept
  {}
};

template <class Fn, class... Args>
__host__ __device__ constexpr bool throws_invocable()
{
  return cuda::std::is_invocable<Fn, Args...>::value && !cuda::std::is_nothrow_invocable<Fn, Args...>::value;
}

template <class Ret, class Fn, class... Args>
__host__ __device__ constexpr bool throws_invocable_r()
{
  return cuda::std::is_invocable_r<Ret, Fn, Args...>::value
      && !cuda::std::is_nothrow_invocable_r<Ret, Fn, Args...>::value;
}

__host__ __device__ void test_noexcept_function_pointers()
{
#if !defined(TEST_COMPILER_NVCC) || TEST_STD_VER >= 2017 // nvbug4360046
  struct Dummy
  {
    __host__ __device__ void foo() noexcept {}
    __host__ __device__ static void bar() noexcept {}
  };
  // Check that PMF's and function pointers actually work and that
  // is_nothrow_invocable returns true for noexcept PMF's and function
  // pointers.
  static_assert(cuda::std::is_nothrow_invocable<decltype(&Dummy::foo), Dummy&>::value, "");
  static_assert(cuda::std::is_nothrow_invocable<decltype(&Dummy::bar)>::value, "");
#endif // !defined(TEST_COMPILER_NVCC) || TEST_STD_VER >= 2017
}

int main(int, char**)
{
#if TEST_STD_VER >= 2017
  using AbominableFunc = void(...) const noexcept;
#endif // TEST_STD_VER >= 2017
  //  Non-callable things
  {
    static_assert(!cuda::std::is_nothrow_invocable<void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<const void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<volatile void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<const volatile void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<cuda::std::nullptr_t>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<double>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable<int[]>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int[3]>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable<int*>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<const int*>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int const*>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable<int&>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<const int&>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int&&>::value, "");

#ifdef _LIBCUDACXX_HAS_VECTOR
    static_assert(!cuda::std::is_nothrow_invocable<int, cuda::std::vector<int>>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int, cuda::std::vector<int*>>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int, cuda::std::vector<int**>>::value, "");
#endif // _LIBCUDACXX_HAS_VECTOR

#if TEST_STD_VER >= 2017
    static_assert(!cuda::std::is_nothrow_invocable<AbominableFunc>::value, "");
#endif // TEST_STD_VER >= 2017

    //  with parameters
    static_assert(!cuda::std::is_nothrow_invocable<int, int>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int, double, float>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int, char, float, double>::value, "");
#if TEST_STD_VER >= 2017
    static_assert(!cuda::std::is_nothrow_invocable<Sink, AbominableFunc>::value, "");
#endif // TEST_STD_VER >= 2017
    static_assert(!cuda::std::is_nothrow_invocable<Sink, void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<Sink, const volatile void>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable_r<int, void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, const void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, volatile void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, const volatile void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::nullptr_t>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, double>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable_r<int, int[]>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int[3]>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable_r<int, int*>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, const int*>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int const*>::value, "");

    static_assert(!cuda::std::is_nothrow_invocable_r<int, int&>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, const int&>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int&&>::value, "");

#ifdef _LIBCUDACXX_HAS_VECTOR
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::vector<int>>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::vector<int*>>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::vector<int**>>::value, "");
#endif // _LIBCUDACXX_HAS_VECTOR
#if TEST_STD_VER >= 2017
    static_assert(!cuda::std::is_nothrow_invocable_r<void, AbominableFunc>::value, "");
#endif // TEST_STD_VER >= 2017

    //  with parameters
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int, int>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int, double, float>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int, char, float, double>::value, "");
#if TEST_STD_VER >= 2017
    static_assert(!cuda::std::is_nothrow_invocable_r<void, Sink, AbominableFunc>::value, "");
#endif // TEST_STD_VER >= 2017
    static_assert(!cuda::std::is_nothrow_invocable_r<void, Sink, void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<void, Sink, const volatile void>::value, "");
  }

  {
    // Check that the conversion to the return type is properly checked
    using Fn = CallObject<true, int>;
    static_assert(cuda::std::is_nothrow_invocable_r<Implicit, Fn>::value, "");
    static_assert(cuda::std::is_nothrow_invocable_r<double, Fn>::value, "");
    static_assert(cuda::std::is_nothrow_invocable_r<const volatile void, Fn>::value, "");
#ifndef TEST_COMPILER_ICC
    static_assert(throws_invocable_r<ThrowsImplicit, Fn>(), "");
#endif // TEST_COMPILER_ICC
    static_assert(!cuda::std::is_nothrow_invocable<Fn(), Explicit>(), "");
  }
  {
    // Check that the conversion to the parameters is properly checked
    using Fn = CallObject<true, void, const Implicit&, const ThrowsImplicit&>;
    static_assert(cuda::std::is_nothrow_invocable<Fn, Implicit&, ThrowsImplicit&>::value, "");
    static_assert(cuda::std::is_nothrow_invocable<Fn, int, ThrowsImplicit&>::value, "");
#ifndef TEST_COMPILER_ICC
    static_assert(throws_invocable<Fn, int, int>(), "");
#endif // TEST_COMPILER_ICC
    static_assert(!cuda::std::is_nothrow_invocable<Fn>::value, "");
  }
  {
    // Check that the noexcept-ness of function objects is checked.
    using Fn  = CallObject<true, void>;
    using Fn2 = CallObject<false, void>;
    static_assert(cuda::std::is_nothrow_invocable<Fn>::value, "");
#ifndef TEST_COMPILER_ICC
    static_assert(throws_invocable<Fn2>(), "");
#endif // TEST_COMPILER_ICC
  }
  {
    // Check that PMD derefs are noexcept
    using Fn = int(Tag::*);
    static_assert(cuda::std::is_nothrow_invocable<Fn, Tag&>::value, "");
    static_assert(cuda::std::is_nothrow_invocable_r<Implicit, Fn, Tag&>::value, "");
#ifndef TEST_COMPILER_ICC
    static_assert(throws_invocable_r<ThrowsImplicit, Fn, Tag&>(), "");
#endif // TEST_COMPILER_ICC
  }
#if TEST_STD_VER >= 2017
  {
    // Check that it's fine if the result type is non-moveable.
    struct CantMove
    {
      CantMove()                               = default;
      __host__ __device__ CantMove(CantMove&&) = delete;
    };

    static_assert(!cuda::std::is_move_constructible_v<CantMove>, "");
    static_assert(!cuda::std::is_copy_constructible_v<CantMove>, "");

    using Fn = CantMove() noexcept;

#  if !defined(TEST_COMPILER_MSVC_2017)
    static_assert(cuda::std::is_nothrow_invocable_r<CantMove, Fn>::value, "");
#  endif // !TEST_COMPILER_MSVC_2017
    static_assert(!cuda::std::is_nothrow_invocable_r<CantMove, Fn, int>::value, "");

#  ifndef TEST_COMPILER_MSVC_2017
    static_assert(cuda::std::is_nothrow_invocable_r_v<CantMove, Fn>, "");
    static_assert(!cuda::std::is_nothrow_invocable_r_v<CantMove, Fn, int>, "");
#  endif // TEST_COMPILER_MSVC_2017
  }
#endif // TEST_STD_VER >= 2017
  {
    // Check for is_nothrow_invocable_v
    using Fn = CallObject<true, int>;
    static_assert(cuda::std::is_nothrow_invocable_v<Fn>, "");
    static_assert(!cuda::std::is_nothrow_invocable_v<Fn, int>, "");
  }
  {
    // Check for is_nothrow_invocable_r_v
    using Fn = CallObject<true, int>;
    static_assert(cuda::std::is_nothrow_invocable_r_v<void, Fn>, "");
    static_assert(!cuda::std::is_nothrow_invocable_r_v<int, Fn, int>, "");
  }
  test_noexcept_function_pointers();

  return 0;
}
