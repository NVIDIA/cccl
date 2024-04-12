//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_macros.h"

// typedef decltype(nullptr) nullptr_t;

struct A
{
  __host__ __device__ A(cuda::std::nullptr_t) {}
};

template <class T>
__host__ __device__ void test_conversions()
{
  {
    T p = 0;
    assert(p == nullptr);
    (void) p; // GCC spuriously claims that p is unused when T is nullptr_t, probably due to optimizations?
  }
  {
    T p = nullptr;
    assert(p == nullptr);
    assert(nullptr == p);
    assert(!(p != nullptr));
    assert(!(nullptr != p));
    (void) p; // GCC spuriously claims that p is unused when T is nullptr_t, probably due to optimizations?
  }
}

template <class T>
struct Voider
{
  typedef void type;
};
template <class T, class = void>
struct has_less : cuda::std::false_type
{};

template <class T>
struct has_less<T, typename Voider<decltype(cuda::std::declval<T>() < nullptr)>::type> : cuda::std::true_type
{};

template <class T>
__host__ __device__ void test_comparisons()
{
  T p = nullptr;
  assert(p == nullptr);
  assert(!(p != nullptr));
  assert(nullptr == p);
  assert(!(nullptr != p));
  (void) p; // GCC spuriously claims that p is unused, probably due to optimizations?
}

#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wnull-conversion"
#endif
__host__ __device__ void test_nullptr_conversions()
{
// GCC does not accept this due to CWG Defect #1423
// http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_defects.html#1423
#if defined(TEST_COMPILER_CLANG) && !defined(TEST_COMPILER_NVCC) && !defined(TEST_COMPILER_CLANG_CUDA)
  {
    bool b = nullptr;
    assert(!b);
  }
#endif
  {
    bool b(nullptr);
    assert(!b);
  }
}
#if defined(__clang__)
#  pragma clang diagnostic pop
#endif

int main(int, char**)
{
  static_assert(sizeof(cuda::std::nullptr_t) == sizeof(void*), "sizeof(cuda::std::nullptr_t) == sizeof(void*)");

  {
    test_conversions<cuda::std::nullptr_t>();
    test_conversions<void*>();
    test_conversions<A*>();
    test_conversions<void (*)()>();
    test_conversions<void (A::*)()>();
    test_conversions<int A::*>();
  }
  {
#ifdef _LIBCUDACXX_HAS_NO_NULLPTR
    static_assert(!has_less<cuda::std::nullptr_t>::value, "");
    // FIXME: our C++03 nullptr emulation still allows for comparisons
    // with other pointer types by way of the conversion operator.
    // static_assert(!has_less<void*>::value, "");
#else
    // TODO Enable this assertion when all compilers implement core DR 583.
    // static_assert(!has_less<cuda::std::nullptr_t>::value, "");
#endif
    test_comparisons<cuda::std::nullptr_t>();
    test_comparisons<void*>();
    test_comparisons<A*>();
    test_comparisons<void (*)()>();
  }
  test_nullptr_conversions();

  return 0;
}
