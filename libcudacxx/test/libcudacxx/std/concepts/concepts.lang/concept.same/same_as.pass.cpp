//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class T, class U>
// concept same_as;

#include <cuda/std/concepts>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::same_as;

struct S1
{};
struct S2
{
  int i;

  __host__ __device__ int& f();
  __host__ __device__ double g(int x) const;
};
struct S3
{
  int& r;
};
struct S4
{
  int&& r;
};
struct S5
{
  int* p;
};

#ifdef TEST_COMPILER_CLANG_CUDA
#  pragma clang diagnostic ignored "-Wunused-private-field"
#endif // TEST_COMPILER_CLANG_CUDA
class C1
{};
class C2
{
  /* [[maybe_unused]] */ int i;
};

class C3
{
public:
  int i;
};

template <class T1, class T2 = T1>
class C4
{
  int t1;
  int t2;
};

template <class T1, class T2 = T1>
class C5
{
  /* [[maybe_unused]] */ T1 t1;

public:
  T2 t2;
};

template <class T1, class T2 = T1>
class C6
{
public:
  /* [[maybe_unused]] */ T1 t1;
  /* [[maybe_unused]] */ T2 t2;
};

template <class T>
struct identity
{
  using type = T;
};

template <template <typename> class Modifier = identity>
__host__ __device__ void CheckSameAs()
{
  static_assert(same_as<typename Modifier<int>::type, typename Modifier<int>::type>, "");
  static_assert(same_as<typename Modifier<S1>::type, typename Modifier<S1>::type>, "");
  static_assert(same_as<typename Modifier<S2>::type, typename Modifier<S2>::type>, "");
  static_assert(same_as<typename Modifier<S3>::type, typename Modifier<S3>::type>, "");
  static_assert(same_as<typename Modifier<S4>::type, typename Modifier<S4>::type>, "");
  static_assert(same_as<typename Modifier<S5>::type, typename Modifier<S5>::type>, "");
  static_assert(same_as<typename Modifier<C1>::type, typename Modifier<C1>::type>, "");
  static_assert(same_as<typename Modifier<C2>::type, typename Modifier<C2>::type>, "");
  static_assert(same_as<typename Modifier<C3>::type, typename Modifier<C3>::type>, "");
  static_assert(same_as<typename Modifier<C4<int>>::type, typename Modifier<C4<int>>::type>, "");
  static_assert(same_as<typename Modifier<C4<int&>>::type, typename Modifier<C4<int&>>::type>, "");
  static_assert(same_as<typename Modifier<C4<int&&>>::type, typename Modifier<C4<int&&>>::type>, "");
  static_assert(same_as<typename Modifier<C5<int>>::type, typename Modifier<C5<int>>::type>, "");
  static_assert(same_as<typename Modifier<C5<int&>>::type, typename Modifier<C5<int&>>::type>, "");
  static_assert(same_as<typename Modifier<C5<int&&>>::type, typename Modifier<C5<int&&>>::type>, "");
  static_assert(same_as<typename Modifier<C6<int>>::type, typename Modifier<C6<int>>::type>, "");
  static_assert(same_as<typename Modifier<C6<int&>>::type, typename Modifier<C6<int&>>::type>, "");
  static_assert(same_as<typename Modifier<C6<int&&>>::type, typename Modifier<C6<int&&>>::type>, "");

  static_assert(same_as<typename Modifier<void>::type, typename Modifier<void>::type>, "");
}

template <template <typename> class Modifier1, template <typename> class Modifier2>
__host__ __device__ void CheckNotSameAs()
{
  static_assert(!same_as<typename Modifier1<int>::type, typename Modifier2<int>::type>, "");
  static_assert(!same_as<typename Modifier1<S1>::type, typename Modifier2<S1>::type>, "");
  static_assert(!same_as<typename Modifier1<S2>::type, typename Modifier2<S2>::type>, "");
  static_assert(!same_as<typename Modifier1<S3>::type, typename Modifier2<S3>::type>, "");
  static_assert(!same_as<typename Modifier1<S4>::type, typename Modifier2<S4>::type>, "");
  static_assert(!same_as<typename Modifier1<S5>::type, typename Modifier2<S5>::type>, "");
  static_assert(!same_as<typename Modifier1<C1>::type, typename Modifier2<C1>::type>, "");
  static_assert(!same_as<typename Modifier1<C2>::type, typename Modifier2<C2>::type>, "");
  static_assert(!same_as<typename Modifier1<C3>::type, typename Modifier2<C3>::type>, "");
  static_assert(!same_as<typename Modifier1<C4<int>>::type, typename Modifier2<C4<int>>::type>, "");
  static_assert(!same_as<typename Modifier1<C4<int&>>::type, typename Modifier2<C4<int&>>::type>, "");
  static_assert(!same_as<typename Modifier1<C4<int&&>>::type, typename Modifier2<C4<int&&>>::type>, "");
  static_assert(!same_as<typename Modifier1<C5<int>>::type, typename Modifier2<C5<int>>::type>, "");
  static_assert(!same_as<typename Modifier1<C5<int&>>::type, typename Modifier2<C5<int&>>::type>, "");
  static_assert(!same_as<typename Modifier1<C5<int&&>>::type, typename Modifier2<C5<int&&>>::type>, "");
  static_assert(!same_as<typename Modifier1<C6<int>>::type, typename Modifier2<C6<int>>::type>, "");
  static_assert(!same_as<typename Modifier1<C6<int&>>::type, typename Modifier2<C6<int&>>::type>, "");
  static_assert(!same_as<typename Modifier1<C6<int&&>>::type, typename Modifier2<C6<int&&>>::type>, "");
}

#if TEST_STD_VER > 2017
// Checks subsumption works as intended
_LIBCUDACXX_TEMPLATE(class T, class U)
_LIBCUDACXX_REQUIRES(same_as<T, U>)
__host__ __device__ void SubsumptionTest();

// clang-format off
_LIBCUDACXX_TEMPLATE(class T, class U)
  _LIBCUDACXX_REQUIRES( same_as<T, U> && true)
__host__ __device__ int SubsumptionTest();
// clang-format on

static_assert(same_as<int, decltype(SubsumptionTest<int, int>())>, "");
static_assert(same_as<int, decltype(SubsumptionTest<void, void>())>, "");
static_assert(same_as<int, decltype(SubsumptionTest<int (*)(), int (*)()>())>, "");
static_assert(same_as<int, decltype(SubsumptionTest<double (&)(int), double (&)(int)>())>, "");
static_assert(same_as<int, decltype(SubsumptionTest<int S2::*, int S2::*>())>, "");
static_assert(same_as<int, decltype(SubsumptionTest<int& (S2::*) (), int& (S2::*) ()>())>, "");
#endif

int main(int, char**)
{
  { // Checks same_as<T, T> is true
    CheckSameAs();

    // Checks same_as<T&, T&> is true
    CheckSameAs<cuda::std::add_lvalue_reference>();

    // Checks same_as<T&&, T&&> is true
    CheckSameAs<cuda::std::add_rvalue_reference>();

    // Checks same_as<const T, const T> is true
    CheckSameAs<cuda::std::add_const>();

    // Checks same_as<volatile T, volatile T> is true
    CheckSameAs<cuda::std::add_volatile>();

    // Checks same_as<const volatile T, const volatile T> is true
    CheckSameAs<cuda::std::add_cv>();

    // Checks same_as<T*, T*> is true
    CheckSameAs<cuda::std::add_pointer>();

    // Checks concrete types are identical
    static_assert(same_as<void, void>, "");

    using Void = void;
    static_assert(same_as<void, Void>, "");

    static_assert(same_as<int[1], int[1]>, "");
    static_assert(same_as<int[2], int[2]>, "");

    static_assert(same_as<int (*)(), int (*)()>, "");
    static_assert(same_as<void (&)(), void (&)()>, "");
    static_assert(same_as<S1& (*) (S1), S1& (*) (S1)>, "");
    static_assert(same_as<C1& (&) (S1, int), C1& (&) (S1, int)>, "");

    static_assert(same_as<int S2::*, int S2::*>, "");
    static_assert(same_as<double S2::*, double S2::*>, "");

    static_assert(same_as<int& (S2::*) (), int& (S2::*) ()>, "");
    static_assert(same_as<double& (S2::*) (int), double& (S2::*) (int)>, "");
  }

  { // Checks that `T` and `T&` are distinct types
    CheckNotSameAs<identity, cuda::std::add_lvalue_reference>();
    CheckNotSameAs<cuda::std::add_lvalue_reference, identity>();

    // Checks that `T` and `T&&` are distinct types
    CheckNotSameAs<identity, cuda::std::add_rvalue_reference>();
    CheckNotSameAs<cuda::std::add_rvalue_reference, identity>();

    // Checks that `T` and `const T` are distinct types
    CheckNotSameAs<identity, cuda::std::add_const>();
    CheckNotSameAs<cuda::std::add_const, identity>();

    // Checks that `T` and `volatile T` are distinct types
    CheckNotSameAs<identity, cuda::std::add_volatile>();
    CheckNotSameAs<cuda::std::add_volatile, identity>();

    // Checks that `T` and `const volatile T` are distinct types
    CheckNotSameAs<identity, cuda::std::add_cv>();
    CheckNotSameAs<cuda::std::add_cv, identity>();

    // Checks that `const T` and `volatile T` are distinct types
    CheckNotSameAs<cuda::std::add_const, cuda::std::add_volatile>();
    CheckNotSameAs<cuda::std::add_volatile, cuda::std::add_const>();

    // Checks that `const T` and `const volatile T` are distinct types
    CheckNotSameAs<cuda::std::add_const, cuda::std::add_cv>();
    CheckNotSameAs<cuda::std::add_cv, cuda::std::add_const>();

    // Checks that `volatile T` and `const volatile T` are distinct types
    CheckNotSameAs<cuda::std::add_volatile, cuda::std::add_cv>();
    CheckNotSameAs<cuda::std::add_cv, cuda::std::add_volatile>();

    // Checks `T&` and `T&&` are distinct types
    CheckNotSameAs<cuda::std::add_lvalue_reference, cuda::std::add_rvalue_reference>();
    CheckNotSameAs<cuda::std::add_rvalue_reference, cuda::std::add_lvalue_reference>();
  }

  { // Checks different type names are distinct types
    static_assert(!same_as<S1, C1>, "");
    static_assert(!same_as<C4<int>, C5<int>>, "");
    static_assert(!same_as<C4<int>, C5<int>>, "");
    static_assert(!same_as<C5<int, double>, C5<double, int>>, "");

    static_assert(!same_as<int&, const int&>, "");
    static_assert(!same_as<int&, volatile int&>, "");
    static_assert(!same_as<int&, const volatile int&>, "");

    static_assert(!same_as<int&&, const int&>, "");
    static_assert(!same_as<int&&, volatile int&>, "");
    static_assert(!same_as<int&&, const volatile int&>, "");

    static_assert(!same_as<int&, const int&&>, "");
    static_assert(!same_as<int&, volatile int&&>, "");
    static_assert(!same_as<int&, const volatile int&&>, "");

    static_assert(!same_as<int&&, const int&&>, "");
    static_assert(!same_as<int&&, volatile int&&>, "");
    static_assert(!same_as<int&&, const volatile int&&>, "");

    static_assert(!same_as<void, int>, "");

    static_assert(!same_as<int[1], int[2]>, "");
    static_assert(!same_as<double[1], int[2]>, "");

    static_assert(!same_as<int* (*) (), const int* (*) ()>, "");
    static_assert(!same_as<void (&)(), void (&)(S1)>, "");
    static_assert(!same_as<S1 (*)(S1), S1& (*) (S1)>, "");
    static_assert(!same_as<C3 (&)(int), C1& (&) (S1, int)>, "");

    static_assert(!same_as<int S2::*, double S2::*>, "");

    static_assert(!same_as<int& (S2::*) (), double& (S2::*) (int)>, "");
  }

  return 0;
}
