//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: clang-7, clang-8

// <cuda/std/variant>

// template <class ...Types> class variant;

// constexpr variant(variant const&);

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"
#include "test_workarounds.h"

struct NonT
{
  __host__ __device__ NonT(int v)
      : value(v)
  {}
  __host__ __device__ NonT(const NonT& o)
      : value(o.value)
  {}
  int value;
};
static_assert(!cuda::std::is_trivially_copy_constructible<NonT>::value, "");

struct NoCopy
{
  NoCopy(const NoCopy&) = delete;
};

struct MoveOnly
{
  MoveOnly(const MoveOnly&) = delete;
  MoveOnly(MoveOnly&&)      = default;
};

struct MoveOnlyNT
{
  MoveOnlyNT(const MoveOnlyNT&) = delete;
  __host__ __device__ MoveOnlyNT(MoveOnlyNT&&) {}
};

struct NTCopy
{
  __host__ __device__ constexpr NTCopy(int v)
      : value(v)
  {}
  __host__ __device__ NTCopy(const NTCopy& that)
      : value(that.value)
  {}
  NTCopy(NTCopy&&) = delete;
  int value;
};

static_assert(!cuda::std::is_trivially_copy_constructible<NTCopy>::value, "");
static_assert(cuda::std::is_copy_constructible<NTCopy>::value, "");

struct TCopy
{
  __host__ __device__ constexpr TCopy(int v)
      : value(v)
  {}
  TCopy(TCopy const&) = default;
  TCopy(TCopy&&)      = delete;
  int value;
};

static_assert(cuda::std::is_trivially_copy_constructible<TCopy>::value, "");

struct TCopyNTMove
{
  __host__ __device__ constexpr TCopyNTMove(int v)
      : value(v)
  {}
  TCopyNTMove(const TCopyNTMove&) = default;
  __host__ __device__ TCopyNTMove(TCopyNTMove&& that)
      : value(that.value)
  {
    that.value = -1;
  }
  int value;
};

static_assert(cuda::std::is_trivially_copy_constructible<TCopyNTMove>::value, "");

#ifndef TEST_HAS_NO_EXCEPTIONS
struct MakeEmptyT
{
  static int alive;
  __host__ __device__ MakeEmptyT()
  {
    ++alive;
  }
  __host__ __device__ MakeEmptyT(const MakeEmptyT&)
  {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  __host__ __device__ MakeEmptyT(MakeEmptyT&&)
  {
    throw 42;
  }
  __host__ __device__ MakeEmptyT& operator=(const MakeEmptyT&)
  {
    throw 42;
  }
  __host__ __device__ MakeEmptyT& operator=(MakeEmptyT&&)
  {
    throw 42;
  }
  __host__ __device__ ~MakeEmptyT()
  {
    --alive;
  }
};

int MakeEmptyT::alive = 0;
template <class Variant>
__host__ __device__ void makeEmpty(Variant& v)
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
#endif // !TEST_HAS_NO_EXCEPTIONS

__host__ __device__ void test_copy_ctor_sfinae()
{
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_copy_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NoCopy>;
    static_assert(!cuda::std::is_copy_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnly>;
    static_assert(!cuda::std::is_copy_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnlyNT>;
    static_assert(!cuda::std::is_copy_constructible<V>::value, "");
  }

  // Make sure we properly propagate triviality (see P0602R4).
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_trivially_copy_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NTCopy>;
    static_assert(!cuda::std::is_trivially_copy_constructible<V>::value, "");
    static_assert(cuda::std::is_copy_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TCopy>;
    static_assert(cuda::std::is_trivially_copy_constructible<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TCopyNTMove>;
    static_assert(cuda::std::is_trivially_copy_constructible<V>::value, "");
  }
}

__host__ __device__ void test_copy_ctor_basic()
{
  {
    cuda::std::variant<int> v(cuda::std::in_place_index<0>, 42);
    cuda::std::variant<int> v2 = v;
    assert(v2.index() == 0);
    assert(cuda::std::get<0>(v2) == 42);
  }
  {
    cuda::std::variant<int, long> v(cuda::std::in_place_index<1>, 42);
    cuda::std::variant<int, long> v2 = v;
    assert(v2.index() == 1);
    assert(cuda::std::get<1>(v2) == 42);
  }
  {
    cuda::std::variant<NonT> v(cuda::std::in_place_index<0>, 42);
    assert(v.index() == 0);
    cuda::std::variant<NonT> v2(v);
    printf("%d\n", (int) v2.index());
    assert(v2.index() == 0);
    // assert(cuda::std::get<0>(v2).value == 42);
  }
  {
    cuda::std::variant<int, NonT> v(cuda::std::in_place_index<1>, 42);
    assert(v.index() == 1);
    cuda::std::variant<int, NonT> v2(v);
    // assert(v2.index() == 1);
    // assert(cuda::std::get<1>(v2).value == 42);
  }

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    constexpr cuda::std::variant<int> v(cuda::std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    constexpr cuda::std::variant<int> v2 = v;
    static_assert(v2.index() == 0, "");
    static_assert(cuda::std::get<0>(v2) == 42, "");
  }
  {
    constexpr cuda::std::variant<int, long> v(cuda::std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    constexpr cuda::std::variant<int, long> v2 = v;
    static_assert(v2.index() == 1, "");
    static_assert(cuda::std::get<1>(v2) == 42, "");
  }
  {
    constexpr cuda::std::variant<TCopy> v(cuda::std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    constexpr cuda::std::variant<TCopy> v2(v);
    static_assert(v2.index() == 0, "");
    static_assert(cuda::std::get<0>(v2).value == 42, "");
  }
  {
    constexpr cuda::std::variant<int, TCopy> v(cuda::std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    constexpr cuda::std::variant<int, TCopy> v2(v);
    static_assert(v2.index() == 1, "");
    static_assert(cuda::std::get<1>(v2).value == 42, "");
  }
  {
    constexpr cuda::std::variant<TCopyNTMove> v(cuda::std::in_place_index<0>, 42);
    static_assert(v.index() == 0, "");
    constexpr cuda::std::variant<TCopyNTMove> v2(v);
    static_assert(v2.index() == 0, "");
    static_assert(cuda::std::get<0>(v2).value == 42, "");
  }
  {
    constexpr cuda::std::variant<int, TCopyNTMove> v(cuda::std::in_place_index<1>, 42);
    static_assert(v.index() == 1, "");
    constexpr cuda::std::variant<int, TCopyNTMove> v2(v);
    static_assert(v2.index() == 1, "");
    static_assert(cuda::std::get<1>(v2).value == 42, "");
  }
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_copy_ctor_valueless_by_exception()
{
  using V = cuda::std::variant<int, MakeEmptyT>;
  V v1;
  makeEmpty(v1);
  const V& cv1 = v1;
  V v(cv1);
  assert(v.valueless_by_exception());
}
#endif // !TEST_HAS_NO_EXCEPTIONS

template <size_t Idx>
__host__ __device__ constexpr bool test_constexpr_copy_ctor_imp(cuda::std::variant<long, void*, const int> const& v)
{
  auto v2 = v;
  return v2.index() == v.index() && v2.index() == Idx && cuda::std::get<Idx>(v2) == cuda::std::get<Idx>(v);
}

__host__ __device__ void test_constexpr_copy_ctor()
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
  static_assert(test_constexpr_copy_ctor_imp<0>(V(42l)), "");
  static_assert(test_constexpr_copy_ctor_imp<1>(V(nullptr)), "");
  static_assert(test_constexpr_copy_ctor_imp<2>(V(101)), "");
}

int main(int, char**)
{
  test_copy_ctor_basic();
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_copy_ctor_valueless_by_exception();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  test_copy_ctor_sfinae();
  test_constexpr_copy_ctor();
  return 0;
}
