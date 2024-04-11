//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   explicit(see-below) tuple(allocator_arg_t, const Alloc& a);

// NOTE: this constructor does not currently support tags derived from
// allocator_arg_t because libc++ has to deduce the parameter as a template
// argument. See PR27684 (https://bugs.llvm.org/show_bug.cgi?id=27684)

#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "DefaultOnly.h"
#include "test_macros.h"

template <class T = void>
struct NonDefaultConstructible
{
  __host__ __device__ constexpr NonDefaultConstructible()
  {
    static_assert(!cuda::std::is_same<T, T>::value, "Default Ctor instantiated");
  }

  __host__ __device__ explicit constexpr NonDefaultConstructible(int) {}
};

struct DerivedFromAllocArgT : cuda::std::allocator_arg_t
{};

int main(int, char**)
{
  DefaultOnly::count()                 = 0;
  alloc_first::allocator_constructed() = false;
  alloc_last::allocator_constructed()  = false;
  {
    cuda::std::tuple<> t(cuda::std::allocator_arg, A1<int>());
    unused(t);
  }
  {
    cuda::std::tuple<int> t(cuda::std::allocator_arg, A1<int>());
    assert(cuda::std::get<0>(t) == 0);
  }
  {
    cuda::std::tuple<DefaultOnly> t(cuda::std::allocator_arg, A1<int>());
    assert(cuda::std::get<0>(t) == DefaultOnly());
  }
  {
    assert(!alloc_first::allocator_constructed());
    cuda::std::tuple<alloc_first> t(cuda::std::allocator_arg, A1<int>(5));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t) == alloc_first());
  }
  {
    assert(!alloc_last::allocator_constructed());
    cuda::std::tuple<alloc_last> t(cuda::std::allocator_arg, A1<int>(5));
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t) == alloc_last());
  }
  {
    alloc_first::allocator_constructed() = false;
    cuda::std::tuple<DefaultOnly, alloc_first> t(cuda::std::allocator_arg, A1<int>(5));
    assert(cuda::std::get<0>(t) == DefaultOnly());
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first());
  }
  {
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    cuda::std::tuple<DefaultOnly, alloc_first, alloc_last> t(cuda::std::allocator_arg, A1<int>(5));
    assert(cuda::std::get<0>(t) == DefaultOnly());
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<2>(t) == alloc_last());
  }
  {
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    cuda::std::tuple<DefaultOnly, alloc_first, alloc_last> t(cuda::std::allocator_arg, A2<int>(5));
    assert(cuda::std::get<0>(t) == DefaultOnly());
    assert(!alloc_first::allocator_constructed());
    assert(cuda::std::get<1>(t) == alloc_first());
    assert(!alloc_last::allocator_constructed());
    assert(cuda::std::get<2>(t) == alloc_last());
  }
  /*
  {
      // Test that the uses-allocator default constructor does not evaluate
      // its SFINAE when it otherwise shouldn't be selected. Do this by
      // using 'NonDefaultConstructible' which will cause a compile error
      // if cuda::std::is_default_constructible is evaluated on it.
      using T = NonDefaultConstructible<>;
      T v(42);
      cuda::std::tuple<T, T> t(v, v);
      unused(t);
      cuda::std::tuple<T, T> t2(42, 42);
      unused(t2);
  }
  */
  return 0;
}
