//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, tuple<UTypes...>&&);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/tuple>

#include "../alloc_first.h"
#include "../alloc_last.h"
#include "allocators.h"
#include "test_macros.h"

#if !_CCCL_TILE_COMPILATION() // virtual functions are unsupported in tile code
struct B
{
  int id_;

  TEST_FUNC explicit B(int i)
      : id_(i)
  {}

  TEST_FUNC virtual ~B() {}
};

struct D : B
{
  TEST_FUNC explicit D(int i)
      : B(i)
  {}
};
#endif // !_CCCL_TILE_COMPILATION()

struct Explicit
{
  int value;
  TEST_FUNC explicit Explicit(int x)
      : value(x)
  {}
};

struct Implicit
{
  int value;
  TEST_FUNC Implicit(int x)
      : value(x)
  {}
};

int main(int, char**)
{
  {
    using T0 = cuda::std::tuple<int>;
    using T1 = cuda::std::tuple<alloc_first>;
    T0 t0(2);
    alloc_first::allocator_constructed() = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 2);
  }

#if !_CCCL_TILE_COMPILATION() // virtual functions are unsupported in tile code
  {
    using T0 = cuda::std::tuple<cuda::std::unique_ptr<D>>;
    using T1 = cuda::std::tuple<cuda::std::unique_ptr<B>>;
    T0 t0(cuda::std::unique_ptr<D>(new D(3)));
    T1 t1(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(cuda::std::get<0>(t1)->id_ == 3);
  }
  {
    using T0 = cuda::std::tuple<int, cuda::std::unique_ptr<D>>;
    using T1 = cuda::std::tuple<alloc_first, cuda::std::unique_ptr<B>>;
    T0 t0(2, cuda::std::unique_ptr<D>(new D(3)));
    alloc_first::allocator_constructed() = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1)->id_ == 3);
  }
  {
    using T0 = cuda::std::tuple<int, int, cuda::std::unique_ptr<D>>;
    using T1 = cuda::std::tuple<alloc_last, alloc_first, cuda::std::unique_ptr<B>>;
    T0 t0(1, 2, cuda::std::unique_ptr<D>(new D(3)));
    alloc_first::allocator_constructed() = false;
    alloc_last::allocator_constructed()  = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(alloc_last::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 1);
    assert(cuda::std::get<1>(t1) == 2);
    assert(cuda::std::get<2>(t1)->id_ == 3);
  }
#endif // !_CCCL_TILE_COMPILATION()
  {
    cuda::std::tuple<int> t1(42);
    cuda::std::tuple<Explicit> t2{cuda::std::allocator_arg, cuda::std::allocator<void>{}, cuda::std::move(t1)};
    assert(cuda::std::get<0>(t2).value == 42);
  }
  {
    cuda::std::tuple<int> t1(42);
    cuda::std::tuple<Implicit> t2 = {cuda::std::allocator_arg, cuda::std::allocator<void>{}, cuda::std::move(t1)};
    assert(cuda::std::get<0>(t2).value == 42);
  }

  return 0;
}
