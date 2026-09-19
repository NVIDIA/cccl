//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <utility>

// template <class T1, class T2> struct pair

// pair& operator=(pair const& p);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/utility>

#include "archetypes.h"
#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4244) // 'initializing': conversion from '_Tp' to '_T2', possible loss of data

struct CountAssign
{
  int copied              = 0;
  int moved               = 0;
  constexpr CountAssign() = default;
  TEST_FUNC constexpr CountAssign& operator=(CountAssign const&)
  {
    ++copied;
    return *this;
  }
  TEST_FUNC constexpr CountAssign& operator=(CountAssign&&)
  {
    ++moved;
    return *this;
  }
};

#if !TEST_COMPILER(MSVC) // an undefined class is not allowed as an argument to compiler intrinsic __is_assignable
struct Incomplete;
_CCCL_GLOBAL_VARIABLE extern Incomplete inc_obj;
#endif // !TEST_COMPILER(MSVC)

TEST_FUNC constexpr bool test()
{
  {
    typedef cuda::std::pair<ConstexprTestTypes::CopyOnly, int> P;
    const P p1(ConstexprTestTypes::CopyOnly(), short{4});
    P p2;
    p2 = p1;
    assert(p2.second == 4);
  }
  {
    using P = cuda::std::pair<int&, int&&>;
    int x   = 42;
    int y   = 101;
    int x2  = -1;
    int y2  = 300;
    P p1(x, cuda::std::move(y));
    P p2(x2, cuda::std::move(y2));
    p1 = p2;
    assert(p1.first == x2);
    assert(p1.second == y2);
  }
  {
    using P = cuda::std::pair<int, ConstexprTestTypes::NonCopyable>;
    static_assert(!cuda::std::is_copy_assignable<P>::value);
  }
  {
    using P = cuda::std::pair<CountAssign, ConstexprTestTypes::Copyable>;
    static_assert(cuda::std::is_copy_assignable<P>::value);
    P p;
    P p2;
    p = p2;
    assert(p.first.copied == 1);
    assert(p.first.moved == 0);
    assert(p2.first.copied == 0);
    assert(p2.first.moved == 0);
  }
  {
    using P = cuda::std::pair<int, ConstexprTestTypes::MoveAssignOnly>;
    static_assert(!cuda::std::is_copy_assignable<P>::value);
  }
  {
    using P = cuda::std::pair<int, cuda::std::unique_ptr<int>>;
    static_assert(!cuda::std::is_copy_assignable<P>::value);
  }
#if !TEST_COMPILER(MSVC) // an undefined class is not allowed as an argument to compiler intrinsic __is_assignable
#  if !_CCCL_TILE_COMPILATION() // error: a non-__tile__ variable cannot be used in tile code
  {
    using P = cuda::std::pair<int, Incomplete&>;
    static_assert(!cuda::std::is_copy_assignable<P>::value);
    [[maybe_unused]] P p(42, inc_obj);
    assert(&p.second == &inc_obj);
  }
#  endif // !_CCCL_TILE_COMPILATION()
#endif // !TEST_COMPILER(MSVC)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}

#if !TEST_COMPILER(MSVC) // an undefined class is not allowed as an argument to compiler intrinsic __is_assignable
struct Incomplete
{};
_CCCL_GLOBAL_VARIABLE Incomplete inc_obj;
#endif // !TEST_COMPILER(MSVC)
