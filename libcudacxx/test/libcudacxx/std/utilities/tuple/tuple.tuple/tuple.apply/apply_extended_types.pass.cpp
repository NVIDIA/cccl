//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: gcc-6

// <cuda/std/tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Testing extended function types. The extended function types are those
// named by INVOKE but that are not actual callable objects. These include
// bullets 1-4 of invoke.

#include <cuda/std/tuple>

// Array tests are disabled
// #include <cuda/std/array>

#include <cuda/std/cassert>
#include <cuda/std/utility>

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"
#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR int count = 0;

struct A_int_0
{
  __host__ __device__ A_int_0()
      : obj1(0)
  {}
  __host__ __device__ A_int_0(int x)
      : obj1(x)
  {}
  __host__ __device__ int mem1()
  {
    return ++count;
  }
  __host__ __device__ int mem2() const
  {
    return ++count;
  }
  int const obj1;
};

struct A_int_1
{
  __host__ __device__ A_int_1() {}
  __host__ __device__ A_int_1(int) {}
  __host__ __device__ int mem1(int x)
  {
    return count += x;
  }
  __host__ __device__ int mem2(int x) const
  {
    return count += x;
  }
};

struct A_int_2
{
  __host__ __device__ A_int_2() {}
  __host__ __device__ A_int_2(int) {}
  __host__ __device__ int mem1(int x, int y)
  {
    return count += (x + y);
  }
  __host__ __device__ int mem2(int x, int y) const
  {
    return count += (x + y);
  }
};

template <class A>
struct A_wrap
{
  __host__ __device__ A_wrap() {}
  __host__ __device__ A_wrap(int x)
      : m_a(x)
  {}
  __host__ __device__ A& operator*()
  {
    return m_a;
  }
  __host__ __device__ A const& operator*() const
  {
    return m_a;
  }
  A m_a;
};

typedef A_wrap<A_int_0> A_wrap_0;
typedef A_wrap<A_int_1> A_wrap_1;
typedef A_wrap<A_int_2> A_wrap_2;

template <class A>
struct A_base : public A
{
  __host__ __device__ A_base()
      : A()
  {}
  __host__ __device__ A_base(int x)
      : A(x)
  {}
};

typedef A_base<A_int_0> A_base_0;
typedef A_base<A_int_1> A_base_1;
typedef A_base<A_int_2> A_base_2;

template <class Tuple,
          class ConstTuple,
          class TuplePtr,
          class ConstTuplePtr,
          class TupleWrap,
          class ConstTupleWrap,
          class TupleBase,
          class ConstTupleBase>
__host__ __device__ void test_ext_int_0()
{
  count = 0;
  typedef A_int_0 T;
  typedef A_wrap_0 Wrap;
  typedef A_base_0 Base;

  typedef int (T::*mem1_t)();
  mem1_t mem1 = &T::mem1;

  typedef int (T::*mem2_t)() const;
  mem2_t mem2 = &T::mem2;

  typedef int const T::*obj1_t;
  obj1_t obj1 = &T::obj1;

  // member function w/ref
  {
    T a;
    Tuple t{a};
    assert(1 == cuda::std::apply(mem1, t));
    assert(count == 1);
  }
  count = 0;
  // member function w/pointer
  {
    T a;
    TuplePtr t{&a};
    assert(1 == cuda::std::apply(mem1, t));
    assert(count == 1);
  }
  count = 0;
  // member function w/base
  {
    Base a;
    TupleBase t{a};
    assert(1 == cuda::std::apply(mem1, t));
    assert(count == 1);
  }
  count = 0;
  // member function w/wrap
  {
    Wrap a;
    TupleWrap t{a};
    assert(1 == cuda::std::apply(mem1, t));
    assert(count == 1);
  }
  count = 0;
  // const member function w/ref
  {
    T const a;
    ConstTuple t{a};
    assert(1 == cuda::std::apply(mem2, t));
    assert(count == 1);
  }
  count = 0;
  // const member function w/pointer
  {
    T const a;
    ConstTuplePtr t{&a};
    assert(1 == cuda::std::apply(mem2, t));
    assert(count == 1);
  }
  count = 0;
  // const member function w/base
  {
    Base const a;
    ConstTupleBase t{a};
    assert(1 == cuda::std::apply(mem2, t));
    assert(count == 1);
  }
  count = 0;
  // const member function w/wrapper
  {
    Wrap const a;
    ConstTupleWrap t{a};
    assert(1 == cuda::std::apply(mem2, t));
    assert(1 == count);
  }
  // member object w/ref
  {
    T a{42};
    Tuple t{a};
    assert(42 == cuda::std::apply(obj1, t));
  }
  // member object w/pointer
  {
    T a{42};
    TuplePtr t{&a};
    assert(42 == cuda::std::apply(obj1, t));
  }
  // member object w/base
  {
    Base a{42};
    TupleBase t{a};
    assert(42 == cuda::std::apply(obj1, t));
  }
  // member object w/wrapper
  {
    Wrap a{42};
    TupleWrap t{a};
    assert(42 == cuda::std::apply(obj1, t));
  }
}

template <class Tuple,
          class ConstTuple,
          class TuplePtr,
          class ConstTuplePtr,
          class TupleWrap,
          class ConstTupleWrap,
          class TupleBase,
          class ConstTupleBase>
__host__ __device__ void test_ext_int_1()
{
  count = 0;
  typedef A_int_1 T;
  typedef A_wrap_1 Wrap;
  typedef A_base_1 Base;

  typedef int (T::*mem1_t)(int);
  mem1_t mem1 = &T::mem1;

  typedef int (T::*mem2_t)(int) const;
  mem2_t mem2 = &T::mem2;

  // member function w/ref
  {
    T a;
    Tuple t{a, 2};
    assert(2 == cuda::std::apply(mem1, t));
    assert(count == 2);
  }
  count = 0;
  // member function w/pointer
  {
    T a;
    TuplePtr t{&a, 3};
    assert(3 == cuda::std::apply(mem1, t));
    assert(count == 3);
  }
  count = 0;
  // member function w/base
  {
    Base a;
    TupleBase t{a, 4};
    assert(4 == cuda::std::apply(mem1, t));
    assert(count == 4);
  }
  count = 0;
  // member function w/wrap
  {
    Wrap a;
    TupleWrap t{a, 5};
    assert(5 == cuda::std::apply(mem1, t));
    assert(count == 5);
  }
  count = 0;
  // const member function w/ref
  {
    T const a;
    ConstTuple t{a, 6};
    assert(6 == cuda::std::apply(mem2, t));
    assert(count == 6);
  }
  count = 0;
  // const member function w/pointer
  {
    T const a;
    ConstTuplePtr t{&a, 7};
    assert(7 == cuda::std::apply(mem2, t));
    assert(count == 7);
  }
  count = 0;
  // const member function w/base
  {
    Base const a;
    ConstTupleBase t{a, 8};
    assert(8 == cuda::std::apply(mem2, t));
    assert(count == 8);
  }
  count = 0;
  // const member function w/wrapper
  {
    Wrap const a;
    ConstTupleWrap t{a, 9};
    assert(9 == cuda::std::apply(mem2, t));
    assert(9 == count);
  }
}

template <class Tuple,
          class ConstTuple,
          class TuplePtr,
          class ConstTuplePtr,
          class TupleWrap,
          class ConstTupleWrap,
          class TupleBase,
          class ConstTupleBase>
__host__ __device__ void test_ext_int_2()
{
  count = 0;
  typedef A_int_2 T;
  typedef A_wrap_2 Wrap;
  typedef A_base_2 Base;

  typedef int (T::*mem1_t)(int, int);
  mem1_t mem1 = &T::mem1;

  typedef int (T::*mem2_t)(int, int) const;
  mem2_t mem2 = &T::mem2;

  // member function w/ref
  {
    T a;
    Tuple t{a, 1, 1};
    assert(2 == cuda::std::apply(mem1, t));
    assert(count == 2);
  }
  count = 0;
  // member function w/pointer
  {
    T a;
    TuplePtr t{&a, 1, 2};
    assert(3 == cuda::std::apply(mem1, t));
    assert(count == 3);
  }
  count = 0;
  // member function w/base
  {
    Base a;
    TupleBase t{a, 2, 2};
    assert(4 == cuda::std::apply(mem1, t));
    assert(count == 4);
  }
  count = 0;
  // member function w/wrap
  {
    Wrap a;
    TupleWrap t{a, 2, 3};
    assert(5 == cuda::std::apply(mem1, t));
    assert(count == 5);
  }
  count = 0;
  // const member function w/ref
  {
    T const a;
    ConstTuple t{a, 3, 3};
    assert(6 == cuda::std::apply(mem2, t));
    assert(count == 6);
  }
  count = 0;
  // const member function w/pointer
  {
    T const a;
    ConstTuplePtr t{&a, 3, 4};
    assert(7 == cuda::std::apply(mem2, t));
    assert(count == 7);
  }
  count = 0;
  // const member function w/base
  {
    Base const a;
    ConstTupleBase t{a, 4, 4};
    assert(8 == cuda::std::apply(mem2, t));
    assert(count == 8);
  }
  count = 0;
  // const member function w/wrapper
  {
    Wrap const a;
    ConstTupleWrap t{a, 4, 5};
    assert(9 == cuda::std::apply(mem2, t));
    assert(9 == count);
  }
}

int main(int, char**)
{
  {
    test_ext_int_0<cuda::std::tuple<A_int_0&>,
                   cuda::std::tuple<A_int_0 const&>,
                   cuda::std::tuple<A_int_0*>,
                   cuda::std::tuple<A_int_0 const*>,
                   cuda::std::tuple<A_wrap_0&>,
                   cuda::std::tuple<A_wrap_0 const&>,
                   cuda::std::tuple<A_base_0&>,
                   cuda::std::tuple<A_base_0 const&>>();
    test_ext_int_0<cuda::std::tuple<A_int_0>,
                   cuda::std::tuple<A_int_0 const>,
                   cuda::std::tuple<A_int_0*>,
                   cuda::std::tuple<A_int_0 const*>,
                   cuda::std::tuple<A_wrap_0>,
                   cuda::std::tuple<A_wrap_0 const>,
                   cuda::std::tuple<A_base_0>,
                   cuda::std::tuple<A_base_0 const>>();
    /*
            test_ext_int_0<
                cuda::std::array<A_int_0, 1>, cuda::std::array<A_int_0 const, 1>
              , cuda::std::array<A_int_0*, 1>, cuda::std::array<A_int_0 const*, 1>
              , cuda::std::array<A_wrap_0, 1>, cuda::std::array<A_wrap_0 const, 1>
              , cuda::std::array<A_base_0, 1>, cuda::std::array<A_base_0 const, 1>
              >();
    */
  }
  {
    test_ext_int_1<cuda::std::tuple<A_int_1&, int>,
                   cuda::std::tuple<A_int_1 const&, int>,
                   cuda::std::tuple<A_int_1*, int>,
                   cuda::std::tuple<A_int_1 const*, int>,
                   cuda::std::tuple<A_wrap_1&, int>,
                   cuda::std::tuple<A_wrap_1 const&, int>,
                   cuda::std::tuple<A_base_1&, int>,
                   cuda::std::tuple<A_base_1 const&, int>>();
    test_ext_int_1<cuda::std::tuple<A_int_1, int>,
                   cuda::std::tuple<A_int_1 const, int>,
                   cuda::std::tuple<A_int_1*, int>,
                   cuda::std::tuple<A_int_1 const*, int>,
                   cuda::std::tuple<A_wrap_1, int>,
                   cuda::std::tuple<A_wrap_1 const, int>,
                   cuda::std::tuple<A_base_1, int>,
                   cuda::std::tuple<A_base_1 const, int>>();
    test_ext_int_1<cuda::std::pair<A_int_1&, int>,
                   cuda::std::pair<A_int_1 const&, int>,
                   cuda::std::pair<A_int_1*, int>,
                   cuda::std::pair<A_int_1 const*, int>,
                   cuda::std::pair<A_wrap_1&, int>,
                   cuda::std::pair<A_wrap_1 const&, int>,
                   cuda::std::pair<A_base_1&, int>,
                   cuda::std::pair<A_base_1 const&, int>>();
    test_ext_int_1<cuda::std::pair<A_int_1, int>,
                   cuda::std::pair<A_int_1 const, int>,
                   cuda::std::pair<A_int_1*, int>,
                   cuda::std::pair<A_int_1 const*, int>,
                   cuda::std::pair<A_wrap_1, int>,
                   cuda::std::pair<A_wrap_1 const, int>,
                   cuda::std::pair<A_base_1, int>,
                   cuda::std::pair<A_base_1 const, int>>();
  }
  {
    test_ext_int_2<cuda::std::tuple<A_int_2&, int, int>,
                   cuda::std::tuple<A_int_2 const&, int, int>,
                   cuda::std::tuple<A_int_2*, int, int>,
                   cuda::std::tuple<A_int_2 const*, int, int>,
                   cuda::std::tuple<A_wrap_2&, int, int>,
                   cuda::std::tuple<A_wrap_2 const&, int, int>,
                   cuda::std::tuple<A_base_2&, int, int>,
                   cuda::std::tuple<A_base_2 const&, int, int>>();
    test_ext_int_2<cuda::std::tuple<A_int_2, int, int>,
                   cuda::std::tuple<A_int_2 const, int, int>,
                   cuda::std::tuple<A_int_2*, int, int>,
                   cuda::std::tuple<A_int_2 const*, int, int>,
                   cuda::std::tuple<A_wrap_2, int, int>,
                   cuda::std::tuple<A_wrap_2 const, int, int>,
                   cuda::std::tuple<A_base_2, int, int>,
                   cuda::std::tuple<A_base_2 const, int, int>>();
  }

  return 0;
}
