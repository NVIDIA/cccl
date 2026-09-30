//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template<class Tuple>
// tuple(Tuple&& u);

#include <cuda/std/__memory_>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <complex>
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "MoveOnly.h"
#include "test_macros.h"

// move_only type which triggers the empty base optimization
struct move_only_ebo
{
  move_only_ebo()                = default;
  move_only_ebo(move_only_ebo&&) = default;
};

// a move_only type which does not trigger the empty base optimization
struct move_only_large final
{
  TEST_FUNC move_only_large()
      : value(42)
  {}
  move_only_large(move_only_large&&) = default;
  int value;
};

template <class Elem>
TEST_FUNC void test_sfinae()
{
  using Tup   = cuda::std::tuple<Elem>;
  using Alloc = cuda::std::allocator<void>;
  using Tag   = cuda::std::allocator_arg_t;
  // special members
  {
    static_assert(cuda::std::is_default_constructible<Tup>::value);
    static_assert(cuda::std::is_move_constructible<Tup>::value);
    static_assert(!cuda::std::is_copy_constructible<Tup>::value);
    static_assert(!cuda::std::is_constructible<Tup, Tup&>::value);
  }
  // args constructors
  {
    static_assert(cuda::std::is_constructible<Tup, Elem&&>::value);
    static_assert(!cuda::std::is_constructible<Tup, Elem const&>::value);
    static_assert(!cuda::std::is_constructible<Tup, Elem&>::value);
  }
  // uses-allocator special member constructors
  {
    static_assert(cuda::std::is_constructible<Tup, Tag, Alloc>::value);
    static_assert(cuda::std::is_constructible<Tup, Tag, Alloc, Tup&&>::value);
    static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Tup const&>::value);
    static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Tup&>::value);
  }
  // uses-allocator args constructors
  {
    static_assert(cuda::std::is_constructible<Tup, Tag, Alloc, Elem&&>::value);
    static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Elem const&>::value);
    static_assert(!cuda::std::is_constructible<Tup, Tag, Alloc, Elem&>::value);
  }
}

int main(int, char**)
{
  {
    using T = cuda::std::tuple<MoveOnly>;
    cuda::std::array<MoveOnly, 1> t0{MoveOnly(0)};
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<MoveOnly>;
                 std::array<MoveOnly, 1> t0{MoveOnly(0)};
                 T t = cuda::std::move(t0);
                 assert(cuda::std::get<0>(t) == 0);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using T = cuda::std::tuple<MoveOnly, MoveOnly>;
    cuda::std::array<MoveOnly, 2> t0{MoveOnly(0), MoveOnly(1)};
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<MoveOnly, MoveOnly>;
                 std::array<MoveOnly, 2> t0{MoveOnly(0), MoveOnly(1)};
                 T t = cuda::std::move(t0);
                 assert(cuda::std::get<0>(t) == 0);
                 assert(cuda::std::get<1>(t) == 1);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using T = cuda::std::tuple<MoveOnly, MoveOnly>;
    cuda::std::pair<MoveOnly, MoveOnly> t0{MoveOnly(0), MoveOnly(1)};
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<MoveOnly, MoveOnly>;
                 std::pair<MoveOnly, MoveOnly> t0{MoveOnly(0), MoveOnly(1)};
                 T t = cuda::std::move(t0);
                 assert(cuda::std::get<0>(t) == 0);
                 assert(cuda::std::get<1>(t) == 1);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    using T = cuda::std::tuple<float, float>;
    cuda::std::complex<float> t0{0.0f, 1.0f};
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0.0f);
    assert(cuda::std::get<1>(t) == 1.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 using T = cuda::std::tuple<float, float>;
                 std::complex<float> t0{0.0f, 1.0f};
                 T t = cuda::std::move(t0);
                 assert(cuda::std::get<0>(t) == 0.0f);
                 assert(cuda::std::get<1>(t) == 1.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  {
    using T = cuda::std::tuple<MoveOnly, MoveOnly, MoveOnly>;
    cuda::std::array<MoveOnly, 3> t0{MoveOnly(0), MoveOnly(1), MoveOnly(2)};
    T t = cuda::std::move(t0);
    assert(cuda::std::get<0>(t) == 0);
    assert(cuda::std::get<1>(t) == 1);
    assert(cuda::std::get<2>(t) == 2);
  }
  {
    test_sfinae<move_only_ebo>();
    test_sfinae<move_only_large>();
  }

  return 0;
}
