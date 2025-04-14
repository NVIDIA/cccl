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

// constexpr variant& operator=(variant const&);

#include <cuda/std/cassert>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#endif // _LIBCUDACXX_HAS_STRING
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"

struct NoCopy
{
  NoCopy(const NoCopy&)            = delete;
  NoCopy& operator=(const NoCopy&) = default;
};

struct CopyOnly
{
  CopyOnly(const CopyOnly&)            = default;
  CopyOnly(CopyOnly&&)                 = delete;
  CopyOnly& operator=(const CopyOnly&) = default;
  CopyOnly& operator=(CopyOnly&&)      = delete;
};

struct MoveOnly
{
  MoveOnly(const MoveOnly&)            = delete;
  MoveOnly(MoveOnly&&)                 = default;
  MoveOnly& operator=(const MoveOnly&) = default;
};

struct MoveOnlyNT
{
  MoveOnlyNT(const MoveOnlyNT&) = delete;
  __host__ __device__ MoveOnlyNT(MoveOnlyNT&&) {}
  MoveOnlyNT& operator=(const MoveOnlyNT&) = default;
};

struct CopyAssign
{
  STATIC_MEMBER_VAR(alive, int)
  STATIC_MEMBER_VAR(copy_construct, int)
  STATIC_MEMBER_VAR(copy_assign, int)
  STATIC_MEMBER_VAR(move_construct, int)
  STATIC_MEMBER_VAR(move_assign, int)
  __host__ __device__ static void reset()
  {
    copy_construct() = copy_assign() = move_construct() = move_assign() = alive() = 0;
  }
  __host__ __device__ CopyAssign(int v)
      : value(v)
  {
    ++alive();
  }
  __host__ __device__ CopyAssign(const CopyAssign& o)
      : value(o.value)
  {
    ++alive();
    ++copy_construct();
  }
  __host__ __device__ CopyAssign(CopyAssign&& o) noexcept
      : value(o.value)
  {
    o.value = -1;
    ++alive();
    ++move_construct();
  }
  __host__ __device__ CopyAssign& operator=(const CopyAssign& o)
  {
    value = o.value;
    ++copy_assign();
    return *this;
  }
  __host__ __device__ CopyAssign& operator=(CopyAssign&& o) noexcept
  {
    value   = o.value;
    o.value = -1;
    ++move_assign();
    return *this;
  }
  __host__ __device__ ~CopyAssign()
  {
    --alive();
  }
  int value;
};

struct CopyMaybeThrows
{
  __host__ __device__ CopyMaybeThrows(const CopyMaybeThrows&);
  __host__ __device__ CopyMaybeThrows& operator=(const CopyMaybeThrows&);
};
struct CopyDoesThrow
{
  __host__ __device__ CopyDoesThrow(const CopyDoesThrow&) noexcept(false);
  __host__ __device__ CopyDoesThrow& operator=(const CopyDoesThrow&) noexcept(false);
};

struct NTCopyAssign
{
  __host__ __device__ constexpr NTCopyAssign(int v)
      : value(v)
  {}
  NTCopyAssign(const NTCopyAssign&) = default;
  NTCopyAssign(NTCopyAssign&&)      = default;
  __host__ __device__ NTCopyAssign& operator=(const NTCopyAssign& that)
  {
    value = that.value;
    return *this;
  };
  NTCopyAssign& operator=(NTCopyAssign&&) = delete;
  int value;
};

static_assert(!cuda::std::is_trivially_copy_assignable<NTCopyAssign>::value, "");
static_assert(cuda::std::is_copy_assignable<NTCopyAssign>::value, "");

struct TCopyAssign
{
  __host__ __device__ constexpr TCopyAssign(int v)
      : value(v)
  {}
  TCopyAssign(const TCopyAssign&)            = default;
  TCopyAssign(TCopyAssign&&)                 = default;
  TCopyAssign& operator=(const TCopyAssign&) = default;
  TCopyAssign& operator=(TCopyAssign&&)      = delete;
  int value;
};

static_assert(cuda::std::is_trivially_copy_assignable<TCopyAssign>::value, "");

struct TCopyAssignNTMoveAssign
{
  __host__ __device__ constexpr TCopyAssignNTMoveAssign(int v)
      : value(v)
  {}
  TCopyAssignNTMoveAssign(const TCopyAssignNTMoveAssign&)            = default;
  TCopyAssignNTMoveAssign(TCopyAssignNTMoveAssign&&)                 = default;
  TCopyAssignNTMoveAssign& operator=(const TCopyAssignNTMoveAssign&) = default;
  __host__ __device__ TCopyAssignNTMoveAssign& operator=(TCopyAssignNTMoveAssign&& that)
  {
    value      = that.value;
    that.value = -1;
    return *this;
  }
  int value;
};

static_assert(cuda::std::is_trivially_copy_assignable_v<TCopyAssignNTMoveAssign>, "");

#if TEST_HAS_EXCEPTIONS()
struct CopyThrows
{
  CopyThrows() = default;
  CopyThrows(const CopyThrows&)
  {
    throw 42;
  }
  CopyThrows& operator=(const CopyThrows&)
  {
    throw 42;
  }
};

struct CopyCannotThrow
{
  static int alive;
  CopyCannotThrow()
  {
    ++alive;
  }
  CopyCannotThrow(const CopyCannotThrow&) noexcept
  {
    ++alive;
  }
  CopyCannotThrow(CopyCannotThrow&&) noexcept
  {
    assert(false);
  }
  CopyCannotThrow& operator=(const CopyCannotThrow&) noexcept = default;
  CopyCannotThrow& operator=(CopyCannotThrow&&) noexcept
  {
    assert(false);
    return *this;
  }
};

int CopyCannotThrow::alive = 0;

struct MoveThrows
{
  static int alive;
  MoveThrows()
  {
    ++alive;
  }
  MoveThrows(const MoveThrows&)
  {
    ++alive;
  }
  MoveThrows(MoveThrows&&)
  {
    throw 42;
  }
  MoveThrows& operator=(const MoveThrows&)
  {
    return *this;
  }
  MoveThrows& operator=(MoveThrows&&)
  {
    throw 42;
  }
  ~MoveThrows()
  {
    --alive;
  }
};

int MoveThrows::alive = 0;

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

__host__ __device__ void test_copy_assignment_not_noexcept()
{
  {
    using V = cuda::std::variant<CopyMaybeThrows>;
    static_assert(!cuda::std::is_nothrow_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, CopyDoesThrow>;
    static_assert(!cuda::std::is_nothrow_copy_assignable<V>::value, "");
  }
}

__host__ __device__ void test_copy_assignment_sfinae()
{
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, CopyOnly>;
    static_assert(cuda::std::is_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NoCopy>;
    static_assert(!cuda::std::is_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnly>;
    static_assert(!cuda::std::is_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnlyNT>;
    static_assert(!cuda::std::is_copy_assignable<V>::value, "");
  }

  // Make sure we properly propagate triviality (see P0602R4).
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_trivially_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NTCopyAssign>;
    static_assert(!cuda::std::is_trivially_copy_assignable<V>::value, "");
    static_assert(cuda::std::is_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TCopyAssign>;
    static_assert(cuda::std::is_trivially_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TCopyAssignNTMoveAssign>;
    static_assert(cuda::std::is_trivially_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, CopyOnly>;
    static_assert(cuda::std::is_trivially_copy_assignable<V>::value, "");
  }
}

#if TEST_HAS_EXCEPTIONS()
void test_copy_assignment_empty_empty()
{
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, long, MET>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
}

void test_copy_assignment_non_empty_empty()
{
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET>;
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
#  if defined(_LIBCUDACXX_HAS_STRING)
  {
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_index<2>, "hello");
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
#  endif // _LIBCUDACXX_HAS_STRING
}

void test_copy_assignment_empty_non_empty()
{
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_index<0>, 42);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 0);
    assert(cuda::std::get<0>(v1) == 42);
  }
#  if defined(_LIBCUDACXX_HAS_STRING)
  {
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_type<cuda::std::string>, "hello");
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
  }
#  endif // _LIBCUDACXX_HAS_STRING
}
#endif // TEST_HAS_EXCEPTIONS()

template <typename T>
struct Result
{
  size_t index;
  T value;
};

__host__ __device__ void test_copy_assignment_same_index()
{
  {
    using V = cuda::std::variant<int>;
    V v1(43);
    V v2(42);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 0);
    assert(cuda::std::get<0>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, long, unsigned>;
    V v1(43l);
    V v2(42l);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, CopyAssign, unsigned>;
    V v1(cuda::std::in_place_type<CopyAssign>, 43);
    V v2(cuda::std::in_place_type<CopyAssign>, 42);
    CopyAssign::reset();
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1).value == 42);
#if !TEST_COMPILER(MSVC)
    assert(CopyAssign::copy_construct() == 0);
    assert(CopyAssign::move_construct() == 0);
    // FIXME(mdominiak): try to narrow down what in the compiler makes it emit an invalid PTX call instruction without
    // this barrier this seems like it is not going to be a fun exercise trying to reproduce this in a minimal enough
    // case that the compiler can fix it so I am leaving it with this workaround for now, as it seems to be a strange
    // interactions of many weird things these tests are doing.
    asm volatile("" ::: "memory");
    assert(CopyAssign::copy_assign() == 1);
#endif // !TEST_COMPILER(MSVC)
  }
#if TEST_HAS_EXCEPTIONS()
#  if defined(_LIBCUDACXX_HAS_STRING)
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_type<MET>);
    MET& mref = cuda::std::get<1>(v1);
    V v2(cuda::std::in_place_type<MET>);
    try
    {
      v1 = v2;
      assert(false);
    }
    catch (...)
    {}
    assert(v1.index() == 1);
    assert(&cuda::std::get<1>(v1) == &mref);
  }
#  endif // _LIBCUDACXX_HAS_STRING
#endif // TEST_HAS_EXCEPTIONS()

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    struct
    {
      __host__ __device__ constexpr Result<int> operator()() const
      {
        using V = cuda::std::variant<int>;
        V v(43);
        V v2(42);
        v = v2;
        return {v.index(), cuda::std::get<0>(v)};
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
        using V = cuda::std::variant<int, long, unsigned>;
        V v(43l);
        V v2(42l);
        v = v2;
        return {v.index(), cuda::std::get<1>(v)};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42l, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<int> operator()() const
      {
        using V = cuda::std::variant<int, TCopyAssign, unsigned>;
        V v(cuda::std::in_place_type<TCopyAssign>, 43);
        V v2(cuda::std::in_place_type<TCopyAssign>, 42);
        v = v2;
        return {v.index(), cuda::std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<int> operator()() const
      {
        using V = cuda::std::variant<int, TCopyAssignNTMoveAssign, unsigned>;
        V v(cuda::std::in_place_type<TCopyAssignNTMoveAssign>, 43);
        V v2(cuda::std::in_place_type<TCopyAssignNTMoveAssign>, 42);
        v = v2;
        return {v.index(), cuda::std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
}

__host__ __device__ void test_copy_assignment_different_index()
{
  {
    using V = cuda::std::variant<int, long, unsigned>;
    V v1(43);
    V v2(42l);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, CopyAssign, unsigned>;
    CopyAssign::reset();
    V v1(cuda::std::in_place_type<unsigned>, 43u);
    V v2(cuda::std::in_place_type<CopyAssign>, 42);
    assert(CopyAssign::copy_construct() == 0);
    assert(CopyAssign::move_construct() == 0);
    assert(CopyAssign::alive() == 1);
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1).value == 42);
#if !TEST_COMPILER(MSVC)
    assert(CopyAssign::alive() == 2);
    assert(CopyAssign::copy_construct() == 1);
    assert(CopyAssign::move_construct() == 1);
    // FIXME(mdominiak): try to narrow down what in the compiler makes it emit an invalid PTX call instruction without
    // this barrier this seems like it is not going to be a fun exercise trying to reproduce this in a minimal enough
    // case that the compiler can fix it so I am leaving it with this workaround for now, as it seems to be a strange
    // interactions of many weird things these tests are doing.
    asm volatile("" ::: "memory");
    assert(CopyAssign::copy_assign() == 0);
#endif // !TEST_COMPILER(MSVC)
  }
#if TEST_HAS_EXCEPTIONS()
#  if defined(_LIBCUDACXX_HAS_STRING)
  {
    using V = cuda::std::variant<int, CopyThrows, cuda::std::string>;
    V v1(cuda::std::in_place_type<cuda::std::string>, "hello");
    V v2(cuda::std::in_place_type<CopyThrows>);
    try
    {
      v1 = v2;
      assert(false);
    }
    catch (...)
    {
      / *...* /
    }
    // Test that copy construction is used directly if move construction may throw,
    // resulting in a valueless variant if copy throws.
    assert(v1.valueless_by_exception());
  }
  {
    using V = cuda::std::variant<int, MoveThrows, cuda::std::string>;
    V v1(cuda::std::in_place_type<cuda::std::string>, "hello");
    V v2(cuda::std::in_place_type<MoveThrows>);
    assert(MoveThrows::alive == 1);
    // Test that copy construction is used directly if move construction may throw.
    v1 = v2;
    assert(v1.index() == 1);
    assert(v2.index() == 1);
    assert(MoveThrows::alive == 2);
  }
  {
    // Test that direct copy construction is preferred when it cannot throw.
    using V = cuda::std::variant<int, CopyCannotThrow, cuda::std::string>;
    V v1(cuda::std::in_place_type<cuda::std::string>, "hello");
    V v2(cuda::std::in_place_type<CopyCannotThrow>);
    assert(CopyCannotThrow::alive == 1);
    v1 = v2;
    assert(v1.index() == 1);
    assert(v2.index() == 1);
    assert(CopyCannotThrow::alive == 2);
  }
  {
    using V = cuda::std::variant<int, CopyThrows, cuda::std::string>;
    V v1(cuda::std::in_place_type<CopyThrows>);
    V v2(cuda::std::in_place_type<cuda::std::string>, "hello");
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
    assert(v2.index() == 2);
    assert(cuda::std::get<2>(v2) == "hello");
  }
  {
    using V = cuda::std::variant<int, MoveThrows, cuda::std::string>;
    V v1(cuda::std::in_place_type<MoveThrows>);
    V v2(cuda::std::in_place_type<cuda::std::string>, "hello");
    V& vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
    assert(v2.index() == 2);
    assert(cuda::std::get<2>(v2) == "hello");
  }
#  endif // _LIBCUDACXX_HAS_STRING
#endif // TEST_HAS_EXCEPTIONS()

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    struct
    {
      __host__ __device__ constexpr Result<long> operator()() const
      {
        using V = cuda::std::variant<int, long, unsigned>;
        V v(43);
        V v2(42l);
        v = v2;
        return {v.index(), cuda::std::get<1>(v)};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42l, "");
  }
  {
    struct
    {
      __host__ __device__ constexpr Result<int> operator()() const
      {
        using V = cuda::std::variant<int, TCopyAssign, unsigned>;
        V v(cuda::std::in_place_type<unsigned>, 43u);
        V v2(cuda::std::in_place_type<TCopyAssign>, 42);
        v = v2;
        return {v.index(), cuda::std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
}

template <size_t NewIdx, class ValueType>
__host__ __device__ constexpr bool
test_constexpr_assign_imp(cuda::std::variant<long, void*, int>&& v, ValueType&& new_value)
{
  const cuda::std::variant<long, void*, int> cp(cuda::std::forward<ValueType>(new_value));
  v = cp;
  return v.index() == NewIdx && cuda::std::get<NewIdx>(v) == cuda::std::get<NewIdx>(cp);
}

__host__ __device__ void test_constexpr_copy_assignment()
{
  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  using V = cuda::std::variant<long, void*, int>;
  static_assert(cuda::std::is_trivially_copyable<V>::value, "");
  static_assert(cuda::std::is_trivially_copy_assignable<V>::value, "");
  static_assert(test_constexpr_assign_imp<0>(V(42l), 101l), "");
  static_assert(test_constexpr_assign_imp<0>(V(nullptr), 101l), "");
  static_assert(test_constexpr_assign_imp<1>(V(42l), nullptr), "");
  static_assert(test_constexpr_assign_imp<2>(V(42l), 101), "");
}

int main(int, char**)
{
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_copy_assignment_empty_empty();))
  NV_IF_TARGET(NV_IS_HOST, (test_copy_assignment_non_empty_empty();))
  NV_IF_TARGET(NV_IS_HOST, (test_copy_assignment_empty_non_empty();))
#endif // TEST_HAS_EXCEPTIONS()
  test_copy_assignment_same_index();
  test_copy_assignment_different_index();
  test_copy_assignment_sfinae();
  test_copy_assignment_not_noexcept();
  test_constexpr_copy_assignment();

  return 0;
}
