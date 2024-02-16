//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16

// <cuda/std/variant>

// template <class ...Types> class variant;

// constexpr variant& operator=(variant const&);

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/variant>

#include "test_macros.h"

struct NoCopy {
  NoCopy(const NoCopy &) = delete;
  NoCopy &operator=(const NoCopy &) = default;
};

struct CopyOnly {
  CopyOnly(const CopyOnly &) = default;
  CopyOnly(CopyOnly &&) = delete;
  CopyOnly &operator=(const CopyOnly &) = default;
  CopyOnly &operator=(CopyOnly &&) = delete;
};

struct MoveOnly {
  MoveOnly(const MoveOnly &) = delete;
  MoveOnly(MoveOnly &&) = default;
  MoveOnly &operator=(const MoveOnly &) = default;
};

struct MoveOnlyNT {
  MoveOnlyNT(const MoveOnlyNT &) = delete;
  TEST_HOST_DEVICE
  MoveOnlyNT(MoveOnlyNT &&) {}
  MoveOnlyNT &operator=(const MoveOnlyNT &) = default;
};

struct CopyAssign {
  STATIC_MEMBER_VAR(alive, int);
  STATIC_MEMBER_VAR(copy_construct, int);
  STATIC_MEMBER_VAR(copy_assign, int);
  STATIC_MEMBER_VAR(move_construct, int);
  STATIC_MEMBER_VAR(move_assign, int);
  TEST_HOST_DEVICE
  static void reset() {
    copy_construct() = copy_assign() = move_construct() = move_assign() = alive() = 0;
  }
  TEST_HOST_DEVICE
  CopyAssign(int v) : value(v) { ++alive(); }
  TEST_HOST_DEVICE
  CopyAssign(const CopyAssign &o) : value(o.value) {
    ++alive();
    ++copy_construct();
  }
  TEST_HOST_DEVICE
  CopyAssign(CopyAssign &&o) noexcept : value(o.value) {
    o.value = -1;
    ++alive();
    ++move_construct();
  }
  TEST_HOST_DEVICE
  CopyAssign &operator=(const CopyAssign &o) {
    value = o.value;
    ++copy_assign();
    return *this;
  }
  TEST_HOST_DEVICE
  CopyAssign &operator=(CopyAssign &&o) noexcept {
    value = o.value;
    o.value = -1;
    ++move_assign();
    return *this;
  }
  TEST_HOST_DEVICE
  ~CopyAssign() { --alive(); }
  int value;
};

struct CopyMaybeThrows {
  TEST_HOST_DEVICE CopyMaybeThrows(const CopyMaybeThrows &);
  TEST_HOST_DEVICE CopyMaybeThrows &operator=(const CopyMaybeThrows &);
};
struct CopyDoesThrow {
  TEST_HOST_DEVICE CopyDoesThrow(const CopyDoesThrow &) noexcept(false);
  TEST_HOST_DEVICE CopyDoesThrow &operator=(const CopyDoesThrow &) noexcept(false);
};


struct NTCopyAssign {
  TEST_HOST_DEVICE
  constexpr NTCopyAssign(int v) : value(v) {}
  NTCopyAssign(const NTCopyAssign &) = default;
  NTCopyAssign(NTCopyAssign &&) = default;
  TEST_HOST_DEVICE
  NTCopyAssign &operator=(const NTCopyAssign &that) {
    value = that.value;
    return *this;
  };
  NTCopyAssign &operator=(NTCopyAssign &&) = delete;
  int value;
};

static_assert(!cuda::std::is_trivially_copy_assignable<NTCopyAssign>::value, "");
static_assert(cuda::std::is_copy_assignable<NTCopyAssign>::value, "");

struct TCopyAssign {
  TEST_HOST_DEVICE
  constexpr TCopyAssign(int v) : value(v) {}
  TCopyAssign(const TCopyAssign &) = default;
  TCopyAssign(TCopyAssign &&) = default;
  TCopyAssign &operator=(const TCopyAssign &) = default;
  TCopyAssign &operator=(TCopyAssign &&) = delete;
  int value;
};

static_assert(cuda::std::is_trivially_copy_assignable<TCopyAssign>::value, "");

struct TCopyAssignNTMoveAssign {
  TEST_HOST_DEVICE
  constexpr TCopyAssignNTMoveAssign(int v) : value(v) {}
  TCopyAssignNTMoveAssign(const TCopyAssignNTMoveAssign &) = default;
  TCopyAssignNTMoveAssign(TCopyAssignNTMoveAssign &&) = default;
  TCopyAssignNTMoveAssign &operator=(const TCopyAssignNTMoveAssign &) = default;
  TEST_HOST_DEVICE
  TCopyAssignNTMoveAssign &operator=(TCopyAssignNTMoveAssign &&that) {
    value = that.value;
    that.value = -1;
    return *this;
  }
  int value;
};

static_assert(cuda::std::is_trivially_copy_assignable_v<TCopyAssignNTMoveAssign>, "");

#ifndef TEST_HAS_NO_EXCEPTIONS
struct CopyThrows {
  CopyThrows() = default;
  TEST_HOST_DEVICE
  CopyThrows(const CopyThrows &) { throw 42; }
  TEST_HOST_DEVICE
  CopyThrows &operator=(const CopyThrows &) { throw 42; }
};

struct CopyCannotThrow {
  static int alive;
  TEST_HOST_DEVICE
  CopyCannotThrow() { ++alive; }
  TEST_HOST_DEVICE
  CopyCannotThrow(const CopyCannotThrow &) noexcept { ++alive; }
  TEST_HOST_DEVICE
  CopyCannotThrow(CopyCannotThrow &&) noexcept { assert(false); }
  TEST_HOST_DEVICE
  CopyCannotThrow &operator=(const CopyCannotThrow &) noexcept = default;
  TEST_HOST_DEVICE
  CopyCannotThrow &operator=(CopyCannotThrow &&) noexcept { assert(false); return *this; }
};

int CopyCannotThrow::alive = 0;

struct MoveThrows {
  static int alive;
  TEST_HOST_DEVICE
  MoveThrows() { ++alive; }
  TEST_HOST_DEVICE
  MoveThrows(const MoveThrows &) { ++alive; }
  TEST_HOST_DEVICE
  MoveThrows(MoveThrows &&) { throw 42; }
  TEST_HOST_DEVICE
  MoveThrows &operator=(const MoveThrows &) { return *this; }
  TEST_HOST_DEVICE
  MoveThrows &operator=(MoveThrows &&) { throw 42; }
  TEST_HOST_DEVICE
  ~MoveThrows() { --alive; }
};

int MoveThrows::alive = 0;

struct MakeEmptyT {
  static int alive;
  TEST_HOST_DEVICE
  MakeEmptyT() { ++alive; }
  TEST_HOST_DEVICE
  MakeEmptyT(const MakeEmptyT &) {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  TEST_HOST_DEVICE
  MakeEmptyT(MakeEmptyT &&) { throw 42; }
  TEST_HOST_DEVICE
  MakeEmptyT &operator=(const MakeEmptyT &) { throw 42; }
  TEST_HOST_DEVICE
  MakeEmptyT &operator=(MakeEmptyT &&) { throw 42; }
  TEST_HOST_DEVICE
  ~MakeEmptyT() { --alive; }
};

int MakeEmptyT::alive = 0;

TEST_HOST_DEVICE
template <class Variant> void makeEmpty(Variant &v) {
  Variant v2(cuda::std::in_place_type<MakeEmptyT>);
  try {
    v = cuda::std::move(v2);
    assert(false);
  } catch (...) {
    assert(v.valueless_by_exception());
  }
}
#endif // TEST_HAS_NO_EXCEPTIONS

TEST_HOST_DEVICE
void test_copy_assignment_not_noexcept() {
#if !defined(TEST_COMPILER_ICC)
  {
    using V = cuda::std::variant<CopyMaybeThrows>;
    static_assert(!cuda::std::is_nothrow_copy_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, CopyDoesThrow>;
    static_assert(!cuda::std::is_nothrow_copy_assignable<V>::value, "");
  }
#endif // !TEST_COMPILER_ICC
}

TEST_HOST_DEVICE
void test_copy_assignment_sfinae() {
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

TEST_HOST_DEVICE
void test_copy_assignment_empty_empty() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, long, MET>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

TEST_HOST_DEVICE
void test_copy_assignment_non_empty_empty() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET>;
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_index<2>, "hello");
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS
}

TEST_HOST_DEVICE
void test_copy_assignment_empty_non_empty() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_index<0>, 42);
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 0);
    assert(cuda::std::get<0>(v1) == 42);
  }
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_type<cuda::std::string>, "hello");
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS
}

template <typename T> struct Result { size_t index; T value; };

TEST_HOST_DEVICE
void test_copy_assignment_same_index() {
  {
    using V = cuda::std::variant<int>;
    V v1(43);
    V v2(42);
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 0);
    assert(cuda::std::get<0>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, long, unsigned>;
    V v1(43l);
    V v2(42l);
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, CopyAssign, unsigned>;
    V v1(cuda::std::in_place_type<CopyAssign>, 43);
    V v2(cuda::std::in_place_type<CopyAssign>, 42);
    CopyAssign::reset();
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1).value == 42);
#if !defined(TEST_COMPILER_MSVC)
    assert(CopyAssign::copy_construct() == 0);
    assert(CopyAssign::move_construct() == 0);
    // FIXME(mdominiak): try to narrow down what in the compiler makes it emit an invalid PTX call instruction without this barrier
    // this seems like it is not going to be a fun exercise trying to reproduce this in a minimal enough case that the compiler can fix it
    // so I am leaving it with this workaround for now, as it seems to be a strange interactions of many weird things these tests are doing.
    asm volatile ("" ::: "memory");
    assert(CopyAssign::copy_assign() == 1);
#endif // !TEST_COMPILER_MSVC
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_type<MET>);
    MET &mref = cuda::std::get<1>(v1);
    V v2(cuda::std::in_place_type<MET>);
    try {
      v1 = v2;
      assert(false);
    } catch (...) {
    }
    assert(v1.index() == 1);
    assert(&cuda::std::get<1>(v1) == &mref);
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    struct {
      TEST_HOST_DEVICE
      constexpr Result<int> operator()() const {
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
    struct {
      TEST_HOST_DEVICE
      constexpr Result<long> operator()() const {
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
    struct {
      TEST_HOST_DEVICE
      constexpr Result<int> operator()() const {
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
    struct {
      TEST_HOST_DEVICE
      constexpr Result<int> operator()() const {
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

TEST_HOST_DEVICE
void test_copy_assignment_different_index() {
  {
    using V = cuda::std::variant<int, long, unsigned>;
    V v1(43);
    V v2(42l);
    V &vref = (v1 = v2);
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
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1).value == 42);
#if !defined(TEST_COMPILER_MSVC) && !defined(TEST_COMPILER_ICC)
    assert(CopyAssign::alive() == 2);
    assert(CopyAssign::copy_construct() == 1);
    assert(CopyAssign::move_construct() == 1);
    // FIXME(mdominiak): try to narrow down what in the compiler makes it emit an invalid PTX call instruction without this barrier
    // this seems like it is not going to be a fun exercise trying to reproduce this in a minimal enough case that the compiler can fix it
    // so I am leaving it with this workaround for now, as it seems to be a strange interactions of many weird things these tests are doing.
    asm volatile ("" ::: "memory");
    assert(CopyAssign::copy_assign() == 0);
#endif // !TEST_COMPILER_MSVC && !TEST_COMPILER_ICC
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  /*{
    using V = cuda::std::variant<int, CopyThrows, cuda::std::string>;
    V v1(cuda::std::in_place_type<cuda::std::string>, "hello");
    V v2(cuda::std::in_place_type<CopyThrows>);
    try {
      v1 = v2;
      assert(false);
    } catch (...) { / * ... * /
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
    V &vref = (v1 = v2);
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
    V &vref = (v1 = v2);
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
    assert(v2.index() == 2);
    assert(cuda::std::get<2>(v2) == "hello");
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS

  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  {
    struct {
      TEST_HOST_DEVICE
      constexpr Result<long> operator()() const {
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
    struct {
      TEST_HOST_DEVICE
      constexpr Result<int> operator()() const {
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
TEST_HOST_DEVICE
constexpr bool test_constexpr_assign_imp(
    cuda::std::variant<long, void*, int>&& v, ValueType&& new_value)
{
  const cuda::std::variant<long, void*, int> cp(
      cuda::std::forward<ValueType>(new_value));
  v = cp;
  return v.index() == NewIdx &&
        cuda::std::get<NewIdx>(v) == cuda::std::get<NewIdx>(cp);
}

TEST_HOST_DEVICE
void test_constexpr_copy_assignment() {
  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  using V = cuda::std::variant<long, void*, int>;
  static_assert(cuda::std::is_trivially_copyable<V>::value, "");
  static_assert(cuda::std::is_trivially_copy_assignable<V>::value, "");
  static_assert(test_constexpr_assign_imp<0>(V(42l), 101l), "");
  static_assert(test_constexpr_assign_imp<0>(V(nullptr), 101l), "");
  static_assert(test_constexpr_assign_imp<1>(V(42l), nullptr), "");
  static_assert(test_constexpr_assign_imp<2>(V(42l), 101), "");
}

int main(int, char**) {
  test_copy_assignment_empty_empty();
  test_copy_assignment_non_empty_empty();
  test_copy_assignment_empty_non_empty();
  test_copy_assignment_same_index();
  test_copy_assignment_different_index();
  test_copy_assignment_sfinae();
  test_copy_assignment_not_noexcept();
  test_constexpr_copy_assignment();

  return 0;
}
