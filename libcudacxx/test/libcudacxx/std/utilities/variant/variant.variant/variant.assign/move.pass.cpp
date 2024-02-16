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

// constexpr variant& operator=(variant&&) noexcept(see below);

#include <cuda/std/cassert>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

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
  MoveOnly &operator=(const MoveOnly &) = delete;
  MoveOnly &operator=(MoveOnly &&) = default;
};

struct MoveOnlyNT {
  MoveOnlyNT(const MoveOnlyNT &) = delete;
  TEST_HOST_DEVICE
  MoveOnlyNT(MoveOnlyNT &&) {}
  MoveOnlyNT &operator=(const MoveOnlyNT &) = delete;
  MoveOnlyNT &operator=(MoveOnlyNT &&) = default;
};

struct MoveOnlyOddNothrow {
  TEST_HOST_DEVICE
  MoveOnlyOddNothrow(MoveOnlyOddNothrow &&) noexcept(false) {}
  MoveOnlyOddNothrow(const MoveOnlyOddNothrow &) = delete;
  MoveOnlyOddNothrow &operator=(MoveOnlyOddNothrow &&) noexcept = default;
  MoveOnlyOddNothrow &operator=(const MoveOnlyOddNothrow &) = delete;
};

struct MoveAssignOnly {
  MoveAssignOnly(MoveAssignOnly &&) = delete;
  MoveAssignOnly &operator=(MoveAssignOnly &&) = default;
};

struct MoveAssign {
  STATIC_MEMBER_VAR(move_construct, int);
  STATIC_MEMBER_VAR(move_assign, int);
  TEST_HOST_DEVICE
  static void reset() { move_construct() = move_assign() = 0; }
  TEST_HOST_DEVICE
  MoveAssign(int v) : value(v) {}
  TEST_HOST_DEVICE
  MoveAssign(MoveAssign &&o) : value(o.value) {
    ++move_construct();
    o.value = -1;
  }
  TEST_HOST_DEVICE
  MoveAssign &operator=(MoveAssign &&o) {
    value = o.value;
    ++move_assign();
    o.value = -1;
    return *this;
  }
  int value;
};

struct NTMoveAssign {
  TEST_HOST_DEVICE
  constexpr NTMoveAssign(int v) : value(v) {}
  NTMoveAssign(const NTMoveAssign &) = default;
  NTMoveAssign(NTMoveAssign &&) = default;
  NTMoveAssign &operator=(const NTMoveAssign &that) = default;
  TEST_HOST_DEVICE
  NTMoveAssign &operator=(NTMoveAssign &&that) {
    value = that.value;
    that.value = -1;
    return *this;
  };
  int value;
};

static_assert(!cuda::std::is_trivially_move_assignable<NTMoveAssign>::value, "");
static_assert(cuda::std::is_move_assignable<NTMoveAssign>::value, "");

struct TMoveAssign {
  TEST_HOST_DEVICE
  constexpr TMoveAssign(int v) : value(v) {}
  TMoveAssign(const TMoveAssign &) = delete;
  TMoveAssign(TMoveAssign &&) = default;
  TMoveAssign &operator=(const TMoveAssign &) = delete;
  TMoveAssign &operator=(TMoveAssign &&) = default;
  int value;
};

static_assert(cuda::std::is_trivially_move_assignable<TMoveAssign>::value, "");

struct TMoveAssignNTCopyAssign {
  TEST_HOST_DEVICE
  constexpr TMoveAssignNTCopyAssign(int v) : value(v) {}
  TMoveAssignNTCopyAssign(const TMoveAssignNTCopyAssign &) = default;
  TMoveAssignNTCopyAssign(TMoveAssignNTCopyAssign &&) = default;
  TEST_HOST_DEVICE
  TMoveAssignNTCopyAssign &operator=(const TMoveAssignNTCopyAssign &that) {
    value = that.value;
    return *this;
  }
  TMoveAssignNTCopyAssign &operator=(TMoveAssignNTCopyAssign &&) = default;
  int value;
};

static_assert(cuda::std::is_trivially_move_assignable_v<TMoveAssignNTCopyAssign>, "");

struct TrivialCopyNontrivialMove {
  TrivialCopyNontrivialMove(TrivialCopyNontrivialMove const&) = default;
  TEST_HOST_DEVICE
  TrivialCopyNontrivialMove(TrivialCopyNontrivialMove&&) noexcept {}
  TrivialCopyNontrivialMove& operator=(TrivialCopyNontrivialMove const&) = default;
  TEST_HOST_DEVICE
  TrivialCopyNontrivialMove& operator=(TrivialCopyNontrivialMove&&) noexcept {
    return *this;
  }
};

static_assert(cuda::std::is_trivially_copy_assignable_v<TrivialCopyNontrivialMove>, "");
static_assert(!cuda::std::is_trivially_move_assignable_v<TrivialCopyNontrivialMove>, "");

TEST_HOST_DEVICE
void test_move_assignment_noexcept() {
  {
    using V = cuda::std::variant<int>;
    static_assert(cuda::std::is_nothrow_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<MoveOnly>;
    static_assert(cuda::std::is_nothrow_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_nothrow_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnly>;
    static_assert(cuda::std::is_nothrow_move_assignable<V>::value, "");
  }
#if !defined(TEST_COMPILER_ICC)
  {
    using V = cuda::std::variant<MoveOnlyNT>;
    static_assert(!cuda::std::is_nothrow_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<MoveOnlyOddNothrow>;
    static_assert(!cuda::std::is_nothrow_move_assignable<V>::value, "");
  }
#endif // !TEST_COMPILER_ICC
}

TEST_HOST_DEVICE
void test_move_assignment_sfinae() {
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, CopyOnly>;
    static_assert(cuda::std::is_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NoCopy>;
    static_assert(!cuda::std::is_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnly>;
    static_assert(cuda::std::is_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, MoveOnlyNT>;
    static_assert(cuda::std::is_move_assignable<V>::value, "");
  }
  {
    // variant only provides move assignment when the types also provide
    // a move constructor.
    using V = cuda::std::variant<int, MoveAssignOnly>;
    static_assert(!cuda::std::is_move_assignable<V>::value, "");
  }

  // Make sure we properly propagate triviality (see P0602R4).
  {
    using V = cuda::std::variant<int, long>;
    static_assert(cuda::std::is_trivially_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, NTMoveAssign>;
    static_assert(!cuda::std::is_trivially_move_assignable<V>::value, "");
    static_assert(cuda::std::is_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TMoveAssign>;
    static_assert(cuda::std::is_trivially_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TMoveAssignNTCopyAssign>;
    static_assert(cuda::std::is_trivially_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, TrivialCopyNontrivialMove>;
    static_assert(!cuda::std::is_trivially_move_assignable<V>::value, "");
  }
  {
    using V = cuda::std::variant<int, CopyOnly>;
    static_assert(cuda::std::is_trivially_move_assignable<V>::value, "");
  }
}

TEST_HOST_DEVICE
void test_move_assignment_empty_empty() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, long, MET>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
#endif // TEST_HAS_NO_EXCEPTIONS
}

TEST_HOST_DEVICE
void test_move_assignment_non_empty_empty() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET>;
    V v1(cuda::std::in_place_index<0>, 42);
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_index<2>, "hello");
    V v2(cuda::std::in_place_index<0>);
    makeEmpty(v2);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS
}

TEST_HOST_DEVICE
void test_move_assignment_empty_non_empty() {
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  {
    using V = cuda::std::variant<int, MET>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_index<0>, 42);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 0);
    assert(cuda::std::get<0>(v1) == 42);
  }
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_index<0>);
    makeEmpty(v1);
    V v2(cuda::std::in_place_type<cuda::std::string>, "hello");
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
  }*/
#endif // TEST_HAS_NO_EXCEPTIONS
}

template <typename T> struct Result { size_t index; T value; };

TEST_HOST_DEVICE
void test_move_assignment_same_index() {
  {
    using V = cuda::std::variant<int>;
    V v1(43);
    V v2(42);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 0);
    assert(cuda::std::get<0>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, long, unsigned>;
    V v1(43l);
    V v2(42l);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, MoveAssign, unsigned>;
    V v1(cuda::std::in_place_type<MoveAssign>, 43);
    V v2(cuda::std::in_place_type<MoveAssign>, 42);
    MoveAssign::reset();
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1).value == 42);
    assert(MoveAssign::move_construct() == 0);
    assert(MoveAssign::move_assign() == 1);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_type<MET>);
    MET &mref = cuda::std::get<1>(v1);
    V v2(cuda::std::in_place_type<MET>);
    try {
      v1 = cuda::std::move(v2);
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
        v = cuda::std::move(v2);
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
        v = cuda::std::move(v2);
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
        using V = cuda::std::variant<int, TMoveAssign, unsigned>;
        V v(cuda::std::in_place_type<TMoveAssign>, 43);
        V v2(cuda::std::in_place_type<TMoveAssign>, 42);
        v = cuda::std::move(v2);
        return {v.index(), cuda::std::get<1>(v).value};
      }
    } test;
    constexpr auto result = test();
    static_assert(result.index == 1, "");
    static_assert(result.value == 42, "");
  }
}

TEST_HOST_DEVICE
void test_move_assignment_different_index() {
  {
    using V = cuda::std::variant<int, long, unsigned>;
    V v1(43);
    V v2(42l);
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1) == 42);
  }
  {
    using V = cuda::std::variant<int, MoveAssign, unsigned>;
    V v1(cuda::std::in_place_type<unsigned>, 43u);
    V v2(cuda::std::in_place_type<MoveAssign>, 42);
    MoveAssign::reset();
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 1);
    assert(cuda::std::get<1>(v1).value == 42);
    assert(MoveAssign::move_construct() == 1);
    assert(MoveAssign::move_assign() == 0);
  }
#ifndef TEST_HAS_NO_EXCEPTIONS
  using MET = MakeEmptyT;
  /*{
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_type<int>);
    V v2(cuda::std::in_place_type<MET>);
    try {
      v1 = cuda::std::move(v2);
      assert(false);
    } catch (...) {
    }
    assert(v1.valueless_by_exception());
    assert(v1.index() == cuda::std::variant_npos);
  }
  {
    using V = cuda::std::variant<int, MET, cuda::std::string>;
    V v1(cuda::std::in_place_type<MET>);
    V v2(cuda::std::in_place_type<cuda::std::string>, "hello");
    V &vref = (v1 = cuda::std::move(v2));
    assert(&vref == &v1);
    assert(v1.index() == 2);
    assert(cuda::std::get<2>(v1) == "hello");
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
        v = cuda::std::move(v2);
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
      constexpr Result<long> operator()() const {
        using V = cuda::std::variant<int, TMoveAssign, unsigned>;
        V v(cuda::std::in_place_type<unsigned>, 43u);
        V v2(cuda::std::in_place_type<TMoveAssign>, 42);
        v = cuda::std::move(v2);
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
  cuda::std::variant<long, void*, int> v2(
      cuda::std::forward<ValueType>(new_value));
  const auto cp = v2;
  v = cuda::std::move(v2);
  return v.index() == NewIdx &&
        cuda::std::get<NewIdx>(v) == cuda::std::get<NewIdx>(cp);
}

TEST_HOST_DEVICE
void test_constexpr_move_assignment() {
  // Make sure we properly propagate triviality, which implies constexpr-ness (see P0602R4).
  using V = cuda::std::variant<long, void*, int>;
  static_assert(cuda::std::is_trivially_copyable<V>::value, "");
  static_assert(cuda::std::is_trivially_move_assignable<V>::value, "");
  static_assert(test_constexpr_assign_imp<0>(V(42l), 101l), "");
  static_assert(test_constexpr_assign_imp<0>(V(nullptr), 101l), "");
  static_assert(test_constexpr_assign_imp<1>(V(42l), nullptr), "");
  static_assert(test_constexpr_assign_imp<2>(V(42l), 101), "");
}

int main(int, char**) {
  test_move_assignment_empty_empty();
  test_move_assignment_non_empty_empty();
  test_move_assignment_empty_non_empty();
  test_move_assignment_same_index();
  test_move_assignment_different_index();
  test_move_assignment_sfinae();
  test_move_assignment_noexcept();
  test_constexpr_move_assignment();

  return 0;
}
