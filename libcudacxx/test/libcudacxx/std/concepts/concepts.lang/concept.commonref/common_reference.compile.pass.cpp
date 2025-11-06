//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template<class From, class To>
// concept common_reference_with;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::common_reference_with;

template <class T, class U>
__host__ __device__ __host__ __device__ constexpr bool CheckCommonReferenceWith() noexcept
{
  static_assert(common_reference_with<T, U&>, "");
  static_assert(common_reference_with<T, const U&>, "");
  static_assert(common_reference_with<T, volatile U&>, "");
  static_assert(common_reference_with<T, const volatile U&>, "");
  static_assert(common_reference_with<T, U&&>, "");
  static_assert(common_reference_with<T, const U&&>, "");
  static_assert(common_reference_with<T, volatile U&&>, "");
  static_assert(common_reference_with<T, const volatile U&&>, "");
  static_assert(common_reference_with<T&, U&&>, "");
  static_assert(common_reference_with<T&, const U&&>, "");
  static_assert(common_reference_with<T&, volatile U&&>, "");
  static_assert(common_reference_with<T&, const volatile U&&>, "");
  static_assert(common_reference_with<const T&, U&&>, "");
  static_assert(common_reference_with<const T&, const U&&>, "");
  static_assert(common_reference_with<const T&, volatile U&&>, "");
  static_assert(common_reference_with<const T&, const volatile U&&>, "");
  static_assert(common_reference_with<volatile T&, U&&>, "");
  static_assert(common_reference_with<volatile T&, const U&&>, "");
  static_assert(common_reference_with<volatile T&, volatile U&&>, "");
  static_assert(common_reference_with<volatile T&, const volatile U&&>, "");
  static_assert(common_reference_with<const volatile T&, U&&>, "");
  static_assert(common_reference_with<const volatile T&, const U&&>, "");
  static_assert(common_reference_with<const volatile T&, volatile U&&>, "");
  static_assert(common_reference_with<const volatile T&, const volatile U&&>, "");

  return common_reference_with<T, U>;
}

namespace BuiltinTypes
{
// fundamental types
static_assert(common_reference_with<void, void>, "");
static_assert(CheckCommonReferenceWith<int, int>(), "");
static_assert(CheckCommonReferenceWith<int, long>(), "");
static_assert(CheckCommonReferenceWith<int, unsigned char>(), "");
#if _CCCL_HAS_INT128()
static_assert(CheckCommonReferenceWith<int, __int128_t>(), "");
#endif
static_assert(CheckCommonReferenceWith<int, double>(), "");

// arrays
static_assert(CheckCommonReferenceWith<int[5], int[5]>(), "");

// pointers (common with void*)
static_assert(CheckCommonReferenceWith<int*, void*>(), "");
static_assert(CheckCommonReferenceWith<int*, const void*>(), "");
static_assert(CheckCommonReferenceWith<int*, volatile void*>(), "");
static_assert(CheckCommonReferenceWith<int*, const volatile void*>(), "");
static_assert(CheckCommonReferenceWith<const int*, void*>(), "");
static_assert(CheckCommonReferenceWith<const int*, const void*>(), "");
static_assert(CheckCommonReferenceWith<const int*, volatile void*>(), "");
static_assert(CheckCommonReferenceWith<const int*, const volatile void*>(), "");
static_assert(CheckCommonReferenceWith<volatile int*, void*>(), "");
static_assert(CheckCommonReferenceWith<volatile int*, const void*>(), "");
static_assert(CheckCommonReferenceWith<volatile int*, volatile void*>(), "");
static_assert(CheckCommonReferenceWith<volatile int*, const volatile void*>(), "");
static_assert(CheckCommonReferenceWith<const volatile int*, void*>(), "");
static_assert(CheckCommonReferenceWith<const volatile int*, const void*>(), "");
static_assert(CheckCommonReferenceWith<const volatile int*, volatile void*>(), "");
static_assert(CheckCommonReferenceWith<const volatile int*, const volatile void*>(), "");

static_assert(CheckCommonReferenceWith<int (*)(), int (*)()>(), "");
#if !TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (*)(), int (*)() noexcept>(), "");
#endif // TEST_COMPILER(NVHPC)
struct S
{};
static_assert(CheckCommonReferenceWith<int S::*, int S::*>(), "");
static_assert(CheckCommonReferenceWith<int S::*, const int S::*>(), "");
static_assert(CheckCommonReferenceWith<int (S::*)(), int (S::*)()>(), "");
#if !TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)(), int (S::*)() noexcept>(), "");
#endif // TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)() const, int (S::*)() const>(), "");
#if !TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)() const, int (S::*)() const noexcept>(), "");
#endif // TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)() volatile, int (S::*)() volatile>(), "");
#if !TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)() volatile, int (S::*)() volatile noexcept>(), "");
#endif // TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)() const volatile, int (S::*)() const volatile>(), "");
#if !TEST_COMPILER(NVHPC)
static_assert(CheckCommonReferenceWith<int (S::*)() const volatile, int (S::*)() const volatile noexcept>(), "");
#endif // TEST_COMPILER(NVHPC)

// nonsense
static_assert(!common_reference_with<double, float*>, "");
static_assert(!common_reference_with<int, int[5]>, "");
static_assert(!common_reference_with<int*, long*>, "");
static_assert(!common_reference_with<int*, unsigned int*>, "");
static_assert(!common_reference_with<int (*)(), int (*)(int)>, "");
static_assert(!common_reference_with<int S::*, float S::*>, "");
static_assert(!common_reference_with<int (S::*)(), int (S::*)() const>, "");
static_assert(!common_reference_with<int (S::*)(), int (S::*)() volatile>, "");
static_assert(!common_reference_with<int (S::*)(), int (S::*)() const volatile>, "");
static_assert(!common_reference_with<int (S::*)() const, int (S::*)() volatile>, "");
static_assert(!common_reference_with<int (S::*)() const, int (S::*)() const volatile>, "");
static_assert(!common_reference_with<int (S::*)() volatile, int (S::*)() const volatile>, "");
} // namespace BuiltinTypes

namespace NoDefaultCommonReference
{
class T
{};

static_assert(!common_reference_with<T, int>, "");
static_assert(!common_reference_with<int, T>, "");
static_assert(!common_reference_with<T, int[10]>, "");
static_assert(!common_reference_with<T[10], int>, "");
static_assert(!common_reference_with<T*, int*>, "");
static_assert(!common_reference_with<T*, const int*>, "");
static_assert(!common_reference_with<T*, volatile int*>, "");
static_assert(!common_reference_with<T*, const volatile int*>, "");
static_assert(!common_reference_with<const T*, int*>, "");
static_assert(!common_reference_with<volatile T*, int*>, "");
static_assert(!common_reference_with<const volatile T*, int*>, "");
static_assert(!common_reference_with<const T*, const int*>, "");
static_assert(!common_reference_with<const T*, volatile int*>, "");
static_assert(!common_reference_with<const T*, const volatile int*>, "");
static_assert(!common_reference_with<const T*, const int*>, "");
static_assert(!common_reference_with<volatile T*, const int*>, "");
static_assert(!common_reference_with<const volatile T*, const int*>, "");
static_assert(!common_reference_with<volatile T*, const int*>, "");
static_assert(!common_reference_with<volatile T*, volatile int*>, "");
static_assert(!common_reference_with<volatile T*, const volatile int*>, "");
static_assert(!common_reference_with<const T*, volatile int*>, "");
static_assert(!common_reference_with<volatile T*, volatile int*>, "");
static_assert(!common_reference_with<const volatile T*, volatile int*>, "");
static_assert(!common_reference_with<const volatile T*, const int*>, "");
static_assert(!common_reference_with<const volatile T*, volatile int*>, "");
static_assert(!common_reference_with<const volatile T*, const volatile int*>, "");
static_assert(!common_reference_with<const T*, const volatile int*>, "");
static_assert(!common_reference_with<volatile T*, const volatile int*>, "");
static_assert(!common_reference_with<const volatile T*, const volatile int*>, "");
static_assert(!common_reference_with<T&, int&>, "");
static_assert(!common_reference_with<T&, const int&>, "");
static_assert(!common_reference_with<T&, volatile int&>, "");
static_assert(!common_reference_with<T&, const volatile int&>, "");
static_assert(!common_reference_with<const T&, int&>, "");
static_assert(!common_reference_with<volatile T&, int&>, "");
static_assert(!common_reference_with<const volatile T&, int&>, "");
static_assert(!common_reference_with<const T&, const int&>, "");
static_assert(!common_reference_with<const T&, volatile int&>, "");
static_assert(!common_reference_with<const T&, const volatile int&>, "");
static_assert(!common_reference_with<const T&, const int&>, "");
static_assert(!common_reference_with<volatile T&, const int&>, "");
static_assert(!common_reference_with<const volatile T&, const int&>, "");
static_assert(!common_reference_with<volatile T&, const int&>, "");
static_assert(!common_reference_with<volatile T&, volatile int&>, "");
static_assert(!common_reference_with<volatile T&, const volatile int&>, "");
static_assert(!common_reference_with<const T&, volatile int&>, "");
static_assert(!common_reference_with<volatile T&, volatile int&>, "");
static_assert(!common_reference_with<const volatile T&, volatile int&>, "");
static_assert(!common_reference_with<const volatile T&, const int&>, "");
static_assert(!common_reference_with<const volatile T&, volatile int&>, "");
static_assert(!common_reference_with<const volatile T&, const volatile int&>, "");
static_assert(!common_reference_with<const T&, const volatile int&>, "");
static_assert(!common_reference_with<volatile T&, const volatile int&>, "");
static_assert(!common_reference_with<const volatile T&, const volatile int&>, "");
static_assert(!common_reference_with<T&, int&&>, "");
static_assert(!common_reference_with<T&, const int&&>, "");
static_assert(!common_reference_with<T&, volatile int&&>, "");
static_assert(!common_reference_with<T&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&, int&&>, "");
static_assert(!common_reference_with<volatile T&, int&&>, "");
static_assert(!common_reference_with<const volatile T&, int&&>, "");
static_assert(!common_reference_with<const T&, const int&&>, "");
static_assert(!common_reference_with<const T&, volatile int&&>, "");
static_assert(!common_reference_with<const T&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&, const int&&>, "");
static_assert(!common_reference_with<volatile T&, const int&&>, "");
static_assert(!common_reference_with<const volatile T&, const int&&>, "");
static_assert(!common_reference_with<volatile T&, const int&&>, "");
static_assert(!common_reference_with<volatile T&, volatile int&&>, "");
static_assert(!common_reference_with<volatile T&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&, volatile int&&>, "");
static_assert(!common_reference_with<volatile T&, volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&, volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&, const int&&>, "");
static_assert(!common_reference_with<const volatile T&, volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&, const volatile int&&>, "");
static_assert(!common_reference_with<volatile T&, const volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&, const volatile int&&>, "");
static_assert(!common_reference_with<T&&, int&>, "");
static_assert(!common_reference_with<T&&, const int&>, "");
static_assert(!common_reference_with<T&&, volatile int&>, "");
static_assert(!common_reference_with<T&&, const volatile int&>, "");
static_assert(!common_reference_with<const T&&, int&>, "");
static_assert(!common_reference_with<volatile T&&, int&>, "");
static_assert(!common_reference_with<const volatile T&&, int&>, "");
static_assert(!common_reference_with<const T&&, const int&>, "");
static_assert(!common_reference_with<const T&&, volatile int&>, "");
static_assert(!common_reference_with<const T&&, const volatile int&>, "");
static_assert(!common_reference_with<const T&&, const int&>, "");
static_assert(!common_reference_with<volatile T&&, const int&>, "");
static_assert(!common_reference_with<const volatile T&&, const int&>, "");
static_assert(!common_reference_with<volatile T&&, const int&>, "");
static_assert(!common_reference_with<volatile T&&, volatile int&>, "");
static_assert(!common_reference_with<volatile T&&, const volatile int&>, "");
static_assert(!common_reference_with<const T&&, volatile int&>, "");
static_assert(!common_reference_with<volatile T&&, volatile int&>, "");
static_assert(!common_reference_with<const volatile T&&, volatile int&>, "");
static_assert(!common_reference_with<const volatile T&&, const int&>, "");
static_assert(!common_reference_with<const volatile T&&, volatile int&>, "");
static_assert(!common_reference_with<const volatile T&&, const volatile int&>, "");
static_assert(!common_reference_with<const T&&, const volatile int&>, "");
static_assert(!common_reference_with<volatile T&&, const volatile int&>, "");
static_assert(!common_reference_with<const volatile T&&, const volatile int&>, "");
static_assert(!common_reference_with<T&&, int&&>, "");
static_assert(!common_reference_with<T&&, const int&&>, "");
static_assert(!common_reference_with<T&&, volatile int&&>, "");
static_assert(!common_reference_with<T&&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&&, int&&>, "");
static_assert(!common_reference_with<volatile T&&, int&&>, "");
static_assert(!common_reference_with<const volatile T&&, int&&>, "");
static_assert(!common_reference_with<const T&&, const int&&>, "");
static_assert(!common_reference_with<const T&&, volatile int&&>, "");
static_assert(!common_reference_with<const T&&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&&, const int&&>, "");
static_assert(!common_reference_with<volatile T&&, const int&&>, "");
static_assert(!common_reference_with<const volatile T&&, const int&&>, "");
static_assert(!common_reference_with<volatile T&&, const int&&>, "");
static_assert(!common_reference_with<volatile T&&, volatile int&&>, "");
static_assert(!common_reference_with<volatile T&&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&&, volatile int&&>, "");
static_assert(!common_reference_with<volatile T&&, volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&&, volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&&, const int&&>, "");
static_assert(!common_reference_with<const volatile T&&, volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&&, const volatile int&&>, "");
static_assert(!common_reference_with<const T&&, const volatile int&&>, "");
static_assert(!common_reference_with<volatile T&&, const volatile int&&>, "");
static_assert(!common_reference_with<const volatile T&&, const volatile int&&>, "");
} // namespace NoDefaultCommonReference

struct s2
{};
struct convertible_with_const_s2
{
  __host__ __device__ __host__ __device__ operator s2 const&() const;
};
static_assert(common_reference_with<convertible_with_const_s2 const&, s2 const&>, "");

struct convertible_with_volatile_s2
{
  __host__ __device__ operator s2 volatile&() volatile;
};
static_assert(common_reference_with<convertible_with_volatile_s2 volatile&, s2 volatile&>, "");

struct BadBasicCommonReference
{
  // This test is ill-formed, NDR. If it ever blows up in our faces: that's a good thing.
  // In the meantime, the test should be included. If compiler support is added, then an include guard
  // should be placed so the test doesn't get deleted.
  __host__ __device__ operator int() const;
  __host__ __device__ operator int&();
};
static_assert(cuda::std::convertible_to<BadBasicCommonReference, int>, "");
static_assert(cuda::std::convertible_to<BadBasicCommonReference, int&>, "");

namespace cuda::std
{
template <template <class> class X, template <class> class Y>
struct basic_common_reference<BadBasicCommonReference, int, X, Y>
{
  using type = BadBasicCommonReference&;
};

template <template <class> class X, template <class> class Y>
struct basic_common_reference<int, BadBasicCommonReference, X, Y>
{
  using type = int&;
};
} // namespace cuda::std

static_assert(!common_reference_with<BadBasicCommonReference, int>, "");

#if TEST_STD_VER > 2017
struct StructNotConvertibleToCommonReference
{
  __host__ __device__ explicit(false) StructNotConvertibleToCommonReference(int);
};
static_assert(cuda::std::convertible_to<int, StructNotConvertibleToCommonReference>, "");

namespace cuda::std
{
template <template <class> class X, template <class> class Y>
struct basic_common_reference<StructNotConvertibleToCommonReference, int, X, Y>
{
  using type = int&;
};

template <template <class> class X, template <class> class Y>
struct basic_common_reference<int, StructNotConvertibleToCommonReference, X, Y>
{
  using type = int&;
};
} // namespace cuda::std
static_assert(!common_reference_with<StructNotConvertibleToCommonReference, int>, "");
#endif // TEST_STD_VER > 2017

struct IntNotConvertibleToCommonReference
{
  __host__ __device__ operator int&() const;
};

namespace cuda::std
{
template <template <class> class X, template <class> class Y>
struct basic_common_reference<IntNotConvertibleToCommonReference, int, X, Y>
{
  using type = int&;
};

template <template <class> class X, template <class> class Y>
struct basic_common_reference<int, IntNotConvertibleToCommonReference, X, Y>
{
  using type = int&;
};
} // namespace cuda::std

static_assert(!common_reference_with<IntNotConvertibleToCommonReference, int>, "");

#if TEST_STD_VER > 2017
struct HasCommonReference
{
  __host__ __device__ explicit(false) HasCommonReference(int);
  __host__ __device__ operator int&() const;
};

namespace cuda::std
{
template <template <class> class X, template <class> class Y>
struct basic_common_reference<HasCommonReference, int, X, Y>
{
  using type = int&;
};

template <template <class> class X, template <class> class Y>
struct basic_common_reference<int, HasCommonReference, X, Y>
{
  using type = int&;
};
} // namespace cuda::std

static_assert(!common_reference_with<HasCommonReference, int>, "");
static_assert(common_reference_with<HasCommonReference, int&>, "");
#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
