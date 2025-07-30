//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// constexpr variant(variant&&) noexcept(see below);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"
#include "test_workarounds.h"

struct ThrowsMove
{
  __host__ __device__ ThrowsMove(ThrowsMove&&) noexcept(false) {}
};

struct NoCopy
{
  NoCopy(const NoCopy&) = delete;
};

struct MoveOnly
{
  int value;
  __host__ __device__ MoveOnly(int v)
      : value(v)
  {}
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&)      = default;
};

struct MoveOnlyNT
{
  int value;
  __host__ __device__ MoveOnlyNT(int v)
      : value(v)
  {}
  MoveOnlyNT(const MoveOnlyNT&) = delete;
  __host__ __device__ MoveOnlyNT(MoveOnlyNT&& other)
      : value(other.value)
  {
    other.value = -1;
  }
};

struct NTMove
{
  __host__ __device__ constexpr NTMove(int v)
      : value(v)
  {}
  NTMove(const NTMove&) = delete;
  __host__ __device__ NTMove(NTMove&& that)
      : value(that.value)
  {
    that.value = -1;
  }
  int value;
};

static_assert(!cuda::std::is_trivially_move_constructible<NTMove>::value, "");
static_assert(cuda::std::is_move_constructible<NTMove>::value, "");

struct TMove
{
  __host__ __device__ constexpr TMove(int v)
      : value(v)
  {}
  TMove(const TMove&) = delete;
  TMove(TMove&&)      = default;
  int value;
};

static_assert(cuda::std::is_trivially_move_constructible<TMove>::value, "");

struct TMoveNTCopy
{
  __host__ __device__ constexpr TMoveNTCopy(int v)
      : value(v)
  {}
  __host__ __device__ TMoveNTCopy(const TMoveNTCopy& that)
      : value(that.value)
  {}
  TMoveNTCopy(TMoveNTCopy&&) = default;
  int value;
};

static_assert(cuda::std::is_trivially_move_constructible<TMoveNTCopy>::value, "");

#if TEST_HAS_EXCEPTIONS()
struct MakeEmptyT
{
  static int alive;
  MakeEmptyT()
  {
    ++alive;
  }
  MakeEmptyT(const MakeEmptyT&)
  {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  MakeEmptyT(MakeEmptyT&&)
  {
    throw 42;
  }
  MakeEmptyT& operator=(const MakeEmptyT&)
  {
    throw 42;
  }
  MakeEmptyT& operator=(MakeEmptyT&&)
  {
    throw 42;
  }
  ~MakeEmptyT()
  {
    --alive;
  }
};

int MakeEmptyT::alive = 0;

template <class Variant>
void makeEmpty(Variant& v)
{
  Variant v2(cuda::std::in_place_type<MakeEmptyT>);
  try
  {
    v = cuda::std::move(v2);
    assert(false);
  }
  catch (...)
  {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_EXCEPTIONS()

__host__ __device__ void test_move_noexcept()
{
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_nothrow_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnly>;
    static_assert(cuda::std::is_nothrow_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnlyNT>;
    static_assert(!cuda::std::is_nothrow_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, ThrowsMove>;
    static_assert(!cuda::std::is_nothrow_move_constructible<V>::value, "");
  }
}

__host__ __device__ void test_move_ctor_sfinae()
{
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnly>;
    static_assert(cuda::std::is_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnlyNT>;
    static_assert(cuda::std::is_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NoCopy>;
    static_assert(!cuda::std::is_move_constructible<V>::value, "");
  }

  // Make sure we properly propagate triviality (see P0602R4).
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_trivially_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NTMove>;
    static_assert(!cuda::std::is_trivially_move_constructible<V>::value, "");
    static_assert(cuda::std::is_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TMove>;
    static_assert(cuda::std::is_trivially_move_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TMoveNTCopy>;
    static_assert(cuda::std::is_trivially_move_constructible<V>::value, "");
  }
}

template <typename T>
struct Result
{
  size_t index;
  T value;
};

__host__ __device__ void test_move_ctor_basic()
{
  {
    cuda::std::variant<int> v(cuda::std::in_place_index<0>, 42);
    cuda::std::variant<int> v2 = cuda::std::move(v);
    assert(v2.index() == 0);
    assert(cuda::std::get<0>(v2) == 42);
  }
  {
    cuda::std::variant<int, long> v(cuda::std::in_place_index<1>, 42);
    cuda::std::variant<int, long> v2 = cuda::std::move(v);
    assert(v2.index() == 1);
    assert(cuda::std::get<1>(v2) == 42);
  }
  {
    cuda::std::variant<MoveOnly> v(cuda::std::in_place_index<0>, 42);
    assert(v.index() == 0);
    cuda::std::variant<MoveOnly> v2(cuda::std::move(v));
    assert(v2.index() == 0);
    assert(cuda::std::get<0>(v2).value == 42);
  }
  {
    cuda::std::variant<int, MoveOnly> v(cuda::std::in_place_index<1>, 42);
    assert(v.index() == 1);
    cuda::std::variant<int, MoveOnly> v2(cuda::std::move(v));
    assert(v2.index() == 1);
    assert(cuda::std::get<1>(v2).value == 42);
  }
  {
    cuda::std::variant<MoveOnlyNT> v(cuda::std::in_place_index<0>, 42);
    assert(v.index() == 0);
    cuda::std::variant<MoveOnlyNT> v2(cuda::std::move(v));
    assert(v2.index() == 0);
    assert(cuda::std::get<0>(v).value == -1);
    assert(cuda::std::get<0>(v2).value == 42);
  }
  {
    cuda::std::variant<int, MoveOnlyNT> v(cuda::std::in_place_index<1>, 42);
    assert(v.index() == 1);
    cuda::std::variant<int, MoveOnlyNT> v2(cuda::std::move(v));
    assert(v2.index() == 1);
    assert(cuda::std::get<1>(v).value == -1);
    assert(cuda::std::get<1>(v2).value == 42);
  }

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    struct
    {
      __host__ __device__ constexpr Result<int> operator()() const
      {
        cuda::std::variant<int> v(cuda::std::in_place_index<0>, 42);
        cuda::std::variant<int> v2 = cuda::std::move(v);
        return {v2.index(), cuda::std::get<0>(cuda::std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value == 42, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<long> operator()() const
      {
        cuda::std::variant<int, long> v(cuda::std::in_place_index<1>, 42);
        cuda::std::variant<int, long> v2 = cuda::std::move(v);
        return {v2.index(), cuda::std::get<1>(cuda::std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<TMove> operator()() const
      {
        cuda::std::variant<TMove> v(cuda::std::in_place_index<0>, 42);
        cuda::std::variant<TMove> v2(cuda::std::move(v));
        return {v2.index(), cuda::std::get<0>(cuda::std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<TMove> operator()() const
      {
        cuda::std::variant<int, TMove> v(cuda::std::in_place_index<1>, 42);
        cuda::std::variant<int, TMove> v2(cuda::std::move(v));
        return {v2.index(), cuda::std::get<1>(cuda::std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<TMoveNTCopy> operator()() const
      {
        cuda::std::variant<TMoveNTCopy> v(cuda::std::in_place_index<0>, 42);
        cuda::std::variant<TMoveNTCopy> v2(cuda::std::move(v));
        return {v2.index(), cuda::std::get<0>(cuda::std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 0, "");
    static_assert(result.value.value == 42, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<TMoveNTCopy> operator()() const
      {
        cuda::std::variant<int, TMoveNTCopy> v(cuda::std::in_place_index<1>, 42);
        cuda::std::variant<int, TMoveNTCopy> v2(cuda::std::move(v));
        return {v2.index(), cuda::std::get<1>(cuda::std::move(v2))};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value.value == 42, "");
  }
}

#if TEST_HAS_EXCEPTIONS()
void test_move_ctor_valueless_by_exception()
{
  using V = cuda::std::variant<int, MakeEmptyT>;
  V v1;
  makeEmpty(v1);
  V v(cuda::std::move(v1));
  assert(v.valueless_by_exception());
}
#endif // TEST_HAS_EXCEPTIONS()

template <size_t Idx>
__host__ __device__ constexpr bool test_constexpr_ctor_imp(cuda::std::variant<long, void*, const int> const& v)
{
  auto copy = v;
  auto v2   = cuda::std::move(copy);
  return v2.index() == v.index() && v2.index() == Idx && cuda::std::get<Idx>(v2) == cuda::std::get<Idx>(v);
}

__host__ __device__ void test_constexpr_move_ctor()
{
  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  using V = cuda::std::variant<long, void*, const int>;
#ifdef TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(cuda::std::is_trivially_destructible<V>::value, "");
  static_assert(cuda::std::is_trivially_copy_constructible<V>::value, "");
  static_assert(cuda::std::is_trivially_move_constructible<V>::value, "");
  static_assert(!cuda::std::is_copy_assignable<V>::value, "");
  static_assert(!cuda::std::is_move_assignable<V>::value, "");
#else // TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(cuda::std::is_trivially_copyable<V>::value, "");
#endif // TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(cuda::std::is_trivially_move_constructible<V>::value, "");
  static_assert(test_constexpr_ctor_imp<0>(V(42l)), "");
  static_assert(test_constexpr_ctor_imp<1>(V(nullptr)), "");
  static_assert(test_constexpr_ctor_imp<2>(V(101)), "");
}

int main(int, char**)
{
  test_move_ctor_basic();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_move_ctor_valueless_by_exception();))
#endif // TEST_HAS_EXCEPTIONS()
  test_move_noexcept();
  test_move_ctor_sfinae();
  test_constexpr_move_ctor();

  return 0;
}
