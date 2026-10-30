//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types> class tuple;

// template <class... UTypes>
//   tuple& operator=(tuple<UTypes...>&& u);

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"

#if !_CCCL_TILE_COMPILATION() // error: global scope non-placement dynamic deallocation is unsupported in tile code
struct B
{
  int id_;
  TEST_FUNC explicit B(int i = 0)
      : id_(i)
  {}
  B(const B&)            = default;
  B& operator=(const B&) = default;
  TEST_FUNC virtual ~B() {}
};

struct D : B
{
  TEST_FUNC explicit D(int i)
      : B(i)
  {}
};
#endif // !_CCCL_TILE_COMPILATION()

struct E
{
  constexpr E() = default;
  TEST_FUNC constexpr E& operator=(int)
  {
    return *this;
  }
};

struct NothrowMoveAssignable
{
  TEST_FUNC constexpr NothrowMoveAssignable& operator=(NothrowMoveAssignable&&) noexcept
  {
    return *this;
  }
};

struct PotentiallyThrowingMoveAssignable
{
  TEST_FUNC constexpr PotentiallyThrowingMoveAssignable& operator=(PotentiallyThrowingMoveAssignable&&)
  {
    return *this;
  }
};

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&)      = delete;
};

struct MoveAssignable
{
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&)      = default;
};

struct CopyAssignable
{
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&)      = delete;
};

struct TrackMove
{
  TEST_FUNC TrackMove()
      : value(0)
      , moved_from(false)
  {}
  TEST_FUNC explicit TrackMove(int v)
      : value(v)
      , moved_from(false)
  {}
  TEST_FUNC TrackMove(TrackMove const& other)
      : value(other.value)
      , moved_from(false)
  {}
  TEST_FUNC TrackMove(TrackMove&& other)
      : value(other.value)
      , moved_from(false)
  {
    other.moved_from = true;
  }
  TEST_FUNC TrackMove& operator=(TrackMove const& other)
  {
    value      = other.value;
    moved_from = false;
    return *this;
  }
  TEST_FUNC TrackMove& operator=(TrackMove&& other)
  {
    value            = other.value;
    moved_from       = false;
    other.moved_from = true;
    return *this;
  }

  int value;
  bool moved_from;
};

TEST_FUNC constexpr bool test()
{
  {
    using T0 = cuda::std::tuple<long>;
    using T1 = cuda::std::tuple<long long>;
    T0 t0(2);
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
  }
  {
    using T0 = cuda::std::tuple<long, char>;
    using T1 = cuda::std::tuple<long long, int>;
    T0 t0(2, 'a');
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
  }
  {
    // Test that tuple evaluates correctly applies an lvalue reference
    // before evaluating is_assignable (i.e. 'is_assignable<int&, int&&>')
    // instead of evaluating 'is_assignable<int&&, int&&>' which is false.
    int x = 42;
    int y = 43;
    cuda::std::tuple<int&&, E> t(cuda::std::move(x), E{});
    cuda::std::tuple<int&&, int> t2(cuda::std::move(y), 44);
    t = cuda::std::move(t2);
    assert(cuda::std::get<0>(t) == 43);
    assert(&cuda::std::get<0>(t) == &x);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

#if !_CCCL_TILE_COMPILATION() // error: global scope non-placement dynamic deallocation is unsupported in tile code
  {
    using T0 = cuda::std::tuple<long, char, D>;
    using T1 = cuda::std::tuple<long long, int, B>;
    T0 t0(2, 'a', D(3));
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 3);
  }
  {
    D d(3);
    D d2(2);
    using T0 = cuda::std::tuple<long, char, D&>;
    using T1 = cuda::std::tuple<long long, int, B&>;
    T0 t0(2, 'a', d2);
    T1 t1(1, 'b', d);
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1).id_ == 2);
  }
#endif // !_CCCL_TILE_COMPILATION()
#if !_CCCL_TILE_COMPILATION() // error: global scope non-placement dynamic deallocation is unsupported in tile code
  {
    using T0 = cuda::std::tuple<long, char, cuda::std::unique_ptr<D>>;
    using T1 = cuda::std::tuple<long long, int, cuda::std::unique_ptr<B>>;
    T0 t0(2, 'a', cuda::std::unique_ptr<D>(new D(3)));
    T1 t1;
    t1 = cuda::std::move(t0);
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1) == int('a'));
    assert(cuda::std::get<2>(t1)->id_ == 3);
  }
#endif // !_CCCL_TILE_COMPILATION()

  {
    using T = cuda::std::tuple<int, NonAssignable>;
    using U = cuda::std::tuple<NonAssignable, int>;
    static_assert(!cuda::std::is_assignable<T&, U&&>::value);
    static_assert(!cuda::std::is_assignable<U&, T&&>::value);
  }
  {
    using T0 = cuda::std::tuple<NothrowMoveAssignable, long>;
    using T1 = cuda::std::tuple<NothrowMoveAssignable, int>;
    static_assert(cuda::std::is_nothrow_assignable<T0&, T1&&>::value);
  }
  {
    typedef cuda::std::tuple<PotentiallyThrowingMoveAssignable, long> T0;
    typedef cuda::std::tuple<PotentiallyThrowingMoveAssignable, int> T1;
    static_assert(cuda::std::is_assignable<T0&, T1&&>::value);
    static_assert(!cuda::std::is_nothrow_assignable<T0&, T1&&>::value);
  }
  {
    // We assign through the reference and don't move out of the incoming ref,
    // so this doesn't work (but would if the type were CopyAssignable).
    {
      using T1 = cuda::std::tuple<MoveAssignable&, long>;
      using T2 = cuda::std::tuple<MoveAssignable&, int>;
      static_assert(!cuda::std::is_assignable<T1&, T2&&>::value);
    }

    // ... works if it's CopyAssignable
    {
      using T1 = cuda::std::tuple<CopyAssignable&, long>;
      using T2 = cuda::std::tuple<CopyAssignable&, int>;
      static_assert(cuda::std::is_assignable<T1&, T2&&>::value);
    }

    // For rvalue-references, we can move-assign if the type is MoveAssignable
    // or CopyAssignable (since in the worst case the move will decay into a copy).
    {
      using T1 = cuda::std::tuple<MoveAssignable&&, long>;
      using T2 = cuda::std::tuple<MoveAssignable&&, int>;
      static_assert(cuda::std::is_assignable<T1&, T2&&>::value);

      using T3 = cuda::std::tuple<CopyAssignable&&, long>;
      using T4 = cuda::std::tuple<CopyAssignable&&, int>;
      static_assert(cuda::std::is_assignable<T3&, T4&&>::value);
    }

    // In all cases, we can't move-assign if the types are not assignable,
    // since we assign through the reference.
    {
      using T1 = cuda::std::tuple<NonAssignable&, long>;
      using T2 = cuda::std::tuple<NonAssignable&, int>;
      static_assert(!cuda::std::is_assignable<T1&, T2&&>::value);

      using T3 = cuda::std::tuple<NonAssignable&&, long>;
      using T4 = cuda::std::tuple<NonAssignable&&, int>;
      static_assert(!cuda::std::is_assignable<T3&, T4&&>::value);
    }
  }
  {
    // Make sure that we don't incorrectly move out of the source's reference.
    using Dest   = cuda::std::tuple<TrackMove, long>;
    using Source = cuda::std::tuple<TrackMove&, int>;
    TrackMove track{3};
    Source src(track, 4);
    assert(!track.moved_from);

    Dest dst;
    dst = cuda::std::move(src); // here we should make a copy
    assert(!track.moved_from);
    assert(cuda::std::get<0>(dst).value == 3);
  }
  {
    // But we do move out of the source's reference if it's a rvalue ref
    using Dest   = cuda::std::tuple<TrackMove, long>;
    using Source = cuda::std::tuple<TrackMove&&, int>;
    TrackMove track{3};
    Source src(cuda::std::move(track), 4);
    assert(!track.moved_from); // we just took a reference

    Dest dst;
    dst = cuda::std::move(src);
    assert(track.moved_from);
    assert(cuda::std::get<0>(dst).value == 3);
  }
  {
    // If the source holds a value, then we move out of it too
    using Dest   = cuda::std::tuple<TrackMove, long>;
    using Source = cuda::std::tuple<TrackMove, int>;
    Source src(TrackMove{3}, 4);
    Dest dst;
    dst = cuda::std::move(src);
    assert(cuda::std::get<0>(src).moved_from);
    assert(cuda::std::get<0>(dst).value == 3);
  }

  return 0;
}
