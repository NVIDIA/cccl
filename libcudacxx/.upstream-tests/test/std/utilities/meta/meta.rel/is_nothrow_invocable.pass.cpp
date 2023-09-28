//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// type_traits

// is_nothrow_invocable

#include <cuda/std/type_traits>
#ifdef _LIBCUDACXX_HAS_VECTOR
#include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR

#include "test_macros.h"

struct Tag {};

struct Implicit {
  __host__ __device__ Implicit(int) noexcept {}
};

struct ThrowsImplicit {
  __host__ __device__ ThrowsImplicit(int) {}
};

struct Explicit {
  __host__ __device__ explicit Explicit(int) noexcept {}
};

template <bool IsNoexcept, class Ret, class... Args>
struct CallObject {
  __host__ __device__ Ret operator()(Args&&...) const noexcept(IsNoexcept);
};

struct Sink {
  template <class... Args>
  __host__ __device__ void operator()(Args&&...) const noexcept {}
};

template <class Fn, class... Args>
__host__ __device__ constexpr bool throws_invocable() {
  return cuda::std::is_invocable<Fn, Args...>::value &&
         !cuda::std::is_nothrow_invocable<Fn, Args...>::value;
}

template <class Ret, class Fn, class... Args>
__host__ __device__ constexpr bool throws_invocable_r() {
  return cuda::std::is_invocable_r<Ret, Fn, Args...>::value &&
         !cuda::std::is_nothrow_invocable_r<Ret, Fn, Args...>::value;
}

// FIXME(EricWF) Don't test the where noexcept is *not* part of the type system
// once implementations have caught up.
__host__ __device__ void test_noexcept_function_pointers() {
  struct Dummy {
    void foo() noexcept {}
    static void bar() noexcept {}
  };
#if !defined(__cpp_noexcept_function_type)
  {
    // Check that PMF's and function pointers *work*. is_nothrow_invocable will always
    // return false because 'noexcept' is not part of the function type.
    static_assert(throws_invocable<decltype(&Dummy::foo), Dummy&>(), "");
    static_assert(throws_invocable<decltype(&Dummy::bar)>(), "");
  }
#else
  {
    // Check that PMF's and function pointers actually work and that
    // is_nothrow_invocable returns true for noexcept PMF's and function
    // pointers.
    static_assert(
        cuda::std::is_nothrow_invocable<decltype(&Dummy::foo), Dummy&>::value, "");
    static_assert(cuda::std::is_nothrow_invocable<decltype(&Dummy::bar)>::value, "");
  }
#endif
}

int main(int, char**) {
  using AbominableFunc = void(...) const noexcept;
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
    static_assert(!cuda::std::is_nothrow_invocable<int, cuda::std::vector<int> >::value,
                  "");
    static_assert(!cuda::std::is_nothrow_invocable<int, cuda::std::vector<int*> >::value,
                  "");
    static_assert(!cuda::std::is_nothrow_invocable<int, cuda::std::vector<int**> >::value,
                  "");
#endif // _LIBCUDACXX_HAS_VECTOR

    static_assert(!cuda::std::is_nothrow_invocable<AbominableFunc>::value, "");

    //  with parameters
    static_assert(!cuda::std::is_nothrow_invocable<int, int>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int, double, float>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<int, char, float, double>::value,
                  "");
    static_assert(!cuda::std::is_nothrow_invocable<Sink, AbominableFunc>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<Sink, void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable<Sink, const volatile void>::value,
                  "");

    static_assert(!cuda::std::is_nothrow_invocable_r<int, void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, const void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, volatile void>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, const volatile void>::value,
                  "");
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
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::vector<int> >::value,
                  "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::vector<int*> >::value,
                  "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, cuda::std::vector<int**> >::value,
                  "");
#endif // _LIBCUDACXX_HAS_VECTOR
    static_assert(!cuda::std::is_nothrow_invocable_r<void, AbominableFunc>::value,
                  "");

    //  with parameters
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int, int>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<int, int, double, float>::value,
                  "");
    static_assert(
        !cuda::std::is_nothrow_invocable_r<int, int, char, float, double>::value, "");
    static_assert(
        !cuda::std::is_nothrow_invocable_r<void, Sink, AbominableFunc>::value, "");
    static_assert(!cuda::std::is_nothrow_invocable_r<void, Sink, void>::value, "");
    static_assert(
        !cuda::std::is_nothrow_invocable_r<void, Sink, const volatile void>::value,
        "");
  }

  {
    // Check that the conversion to the return type is properly checked
    using Fn = CallObject<true, int>;
    static_assert(cuda::std::is_nothrow_invocable_r<Implicit, Fn>::value, "");
    static_assert(cuda::std::is_nothrow_invocable_r<double, Fn>::value, "");
    static_assert(cuda::std::is_nothrow_invocable_r<const volatile void, Fn>::value,
                  "");
    static_assert(throws_invocable_r<ThrowsImplicit, Fn>(), "");
    static_assert(!cuda::std::is_nothrow_invocable<Fn(), Explicit>(), "");
  }
  {
    // Check that the conversion to the parameters is properly checked
    using Fn = CallObject<true, void, const Implicit&, const ThrowsImplicit&>;
    static_assert(
        cuda::std::is_nothrow_invocable<Fn, Implicit&, ThrowsImplicit&>::value, "");
    static_assert(cuda::std::is_nothrow_invocable<Fn, int, ThrowsImplicit&>::value,
                  "");
    static_assert(throws_invocable<Fn, int, int>(), "");
    static_assert(!cuda::std::is_nothrow_invocable<Fn>::value, "");
  }
  {
    // Check that the noexcept-ness of function objects is checked.
    using Fn = CallObject<true, void>;
    using Fn2 = CallObject<false, void>;
    static_assert(cuda::std::is_nothrow_invocable<Fn>::value, "");
    static_assert(throws_invocable<Fn2>(), "");
  }
  {
    // Check that PMD derefs are noexcept
    using Fn = int(Tag::*);
    static_assert(cuda::std::is_nothrow_invocable<Fn, Tag&>::value, "");
    static_assert(cuda::std::is_nothrow_invocable_r<Implicit, Fn, Tag&>::value, "");
    static_assert(throws_invocable_r<ThrowsImplicit, Fn, Tag&>(), "");
  }
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
