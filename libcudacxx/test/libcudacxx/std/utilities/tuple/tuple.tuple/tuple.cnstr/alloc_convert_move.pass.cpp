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
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../alloc_constexpr_types.h"
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
  TEST_FUNC constexpr explicit Explicit(int x)
      : value(x)
  {}
};

struct Implicit
{
  int value;
  TEST_FUNC constexpr Implicit(int x)
      : value(x)
  {}
};

TEST_FUNC void test_runtime()
{
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
}

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using FromComplex     = cuda::std::tuple<throw_on_alloc_arg, throw_on_alloc_arg>;
  using FromComplexLast = cuda::std::tuple<throw_on_alloc_last, throw_on_alloc_last>;
  using FromArray       = cuda::std::array<int, 2>;
  static_assert(
    cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromArray>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(!cuda::std::is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromArray>);
  static_assert(
    cuda::std::is_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(cuda::std::is_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, FromArray>);
  static_assert(
    !cuda::std::
      is_nothrow_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, FromArray>);

  try
  {
    [[maybe_unused]] FromComplex t(cuda::std::allocator_arg, A1<int>{}, cuda::std::complex<float>{1.f, 2.f});
    assert(false);
  }
  catch (int)
  {}

  try
  {
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, cuda::std::complex<float>{1.f, 2.f});
    assert(false);
  }
  catch (int)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    cuda::std::array<int, 2> rvalue_array_vals{12, 13};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_rvalue_array(
      cuda::std::allocator_arg, alloc, cuda::std::move(rvalue_array_vals));
    assert(cuda::std::get<0>(from_rvalue_array).value == 12);
    assert(cuda::std::get<1>(from_rvalue_array).value == 13);
  }
  {
    cuda::std::tuple<int> src(2);
    cuda::std::tuple<constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, cuda::std::move(src));
    assert(cuda::std::get<0>(t).value == 2);
  }
  {
    cuda::std::tuple<int> src(42);
    cuda::std::tuple<Explicit> t{cuda::std::allocator_arg, alloc, cuda::std::move(src)};
    assert(cuda::std::get<0>(t).value == 42);
  }
  {
    cuda::std::tuple<int> src(42);
    cuda::std::tuple<Implicit> t = {cuda::std::allocator_arg, alloc, cuda::std::move(src)};
    assert(cuda::std::get<0>(t).value == 42);
  }
  static_assert(
    cuda::std::is_constructible_v<cuda::std::tuple<>, cuda::std::allocator_arg_t, A1<int>, cuda::std::array<int, 0>>);
  static_assert(
    cuda::std::is_constructible_v<cuda::std::tuple<int>, cuda::std::allocator_arg_t, A1<int>, cuda::std::array<int, 1>>);
  static_assert(
    cuda::std::
      is_constructible_v<cuda::std::tuple<int, int>, cuda::std::allocator_arg_t, A1<int>, cuda::std::array<int, 2>>);
  static_assert(
    cuda::std::
      is_constructible_v<cuda::std::tuple<int, int, int>, cuda::std::allocator_arg_t, A1<int>, cuda::std::array<int, 3>>);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  test_runtime();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
