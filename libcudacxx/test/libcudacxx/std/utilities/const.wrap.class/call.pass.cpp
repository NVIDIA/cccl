//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// nvcc 12.0 segfaults.
// UNSUPPORTED: nvcc-12.0

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// todo(dabayer): It seems that msvc has problems picking up the consteval invoke path. Investigate.

// REQUIRES: !c++17

// constant_wrapper

// template<class... Args>
// static constexpr decltype(auto) operator()(Args&&... args) noexcept(see below);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "helpers.h"
#include "MoveOnly.h"
#include "test_macros.h"

struct MoveOnlyFn
{
  TEST_FUNC constexpr MoveOnly operator()(const MoveOnly& m1, MoveOnly m2, MoveOnly&& m3) const
  {
    return MoveOnly(m1.get() + m2.get() + m3.get());
  }
};

TEST_FUNC constexpr bool fun_ptr(int i)
{
  return i > 0;
}

struct OverloadSet
{
  TEST_FUNC constexpr int operator()(int) const
  {
    return 1;
  }

  TEST_FUNC constexpr int operator()(cuda::std::__constant_wrapper<42>) const
  {
    return 2;
  }
};

struct ReturnNonStructural
{
  TEST_FUNC constexpr NonStructural operator()(int i) const
  {
    return NonStructural{i};
  }
};

struct CWOnly
{
  TEST_FUNC constexpr int operator()(cuda::std::__constant_wrapper<42>) const
  {
    return 42;
  }
};

TEST_FUNC constexpr int nothrow_call(int) noexcept
{
  return 42;
}

TEST_FUNC constexpr int throwing_call(int)
{
  return 42;
}

struct S
{
  int member = 42;

  TEST_FUNC constexpr int mem_fun(int i) const
  {
    return member + i;
  }
};

constexpr S s_value{};

// Let call-expr be constant_wrapper<INVOKE (value, remove_cvref_t<Args>::value...)>{} if all types
// in remove_cvref_t<Args>... satisfy constexpr-param and constant_wrapper<INVOKE (value, remove_-
// cvref_t<Args>::value...)> is a valid type, otherwise let call-expr be INVOKE (value,
// cuda::std::forward<Args>(args)...).
//
// Constraints: call-expr is a valid expression.
// Remarks: The exception specification is equivalent to noexcept(call-expr).

// clang-format off
static_assert(cuda::std::is_invocable_v<cuda::std::__constant_wrapper<[] { return 42; }>>);
static_assert(!cuda::std::is_invocable_v<cuda::std::__constant_wrapper<[] { return 42; }>, int>);
static_assert(!cuda::std::is_invocable_v<cuda::std::__constant_wrapper<5>>);

static_assert(!cuda::std::is_invocable_v<cuda::std::__constant_wrapper<cuda::std::plus<>{}>, int>);
static_assert(cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<cuda::std::plus<>{}>, int, int>);
static_assert(cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<cuda::std::plus<>{}>, cuda::std::__constant_wrapper<42>, int>);
// todo(dabayer): This is failing when compiling with msvc with:
//   'cuda::std::__4::operator +': call to immediate function is not a constant expression
#if !_CCCL_COMPILER(MSVC)
static_assert(cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<cuda::std::plus<>{}>, cuda::std::__constant_wrapper<42>, cuda::std::__constant_wrapper<42>>);
#endif // !_CCCL_COMPILER(MSVC)

// gcc < 13 fails this test with error:
//   'nothrow_call'/'throwing_call' is not a valid template argument of type 'int (*)(int) noexcept' because it is not
//   a variable
#if !_CCCL_COMPILER(GCC, <, 13)
static_assert(cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<nothrow_call>, int>);
static_assert(cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<nothrow_call>, cuda::std::__constant_wrapper<42>>);

static_assert(cuda::std::is_invocable_v<cuda::std::__constant_wrapper<throwing_call>, int>);
static_assert(!cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<throwing_call>, int>);
static_assert(cuda::std::is_nothrow_invocable_v<cuda::std::__constant_wrapper<throwing_call>, cuda::std::__constant_wrapper<42>>,
              "the call expression is still nothrow because the constexpr path is taken");
#endif // !_CCCL_COMPILER(GCC, <, 13)
// clang-format on

template <class T>
struct MustBeInt
{
  static_assert(cuda::std::same_as<T, int>);
};

struct Poison
{
  template <class T>
  constexpr auto operator()(T) const noexcept -> MustBeInt<T>
  {
    return {};
  }
};

#if _CCCL_HAS_STATIC_CALL_OPERATOR()
#  define TEST_CALL(T, ...) T::operator()(__VA_ARGS__)
#else // ^^^ _CCCL_HAS_STATIC_CALL_OPERATOR() ^^^ / vvv !_CCCL_HAS_STATIC_CALL_OPERATOR() vvv
#  define TEST_CALL(T, ...) T{}(__VA_ARGS__)
#endif // ^^^ !_CCCL_HAS_STATIC_CALL_OPERATOR() ^^^

TEST_FUNC constexpr bool test()
{
  {
    // with runtime param
    using T                                       = cuda::std::__constant_wrapper<cuda::std::plus<>{}>;
    cuda::std::same_as<int> decltype(auto) result = TEST_CALL(T, 1, 2);
    assert(result == 3);
  }

  {
    // with runtime param and constexpr param
    using T                                       = cuda::std::__constant_wrapper<cuda::std::plus<>{}>;
    cuda::std::same_as<int> decltype(auto) result = TEST_CALL(T, cuda::std::__cw<1>, 2);
    assert(result == 3);
  }

  {
    // msvc believes this is not a constant expression.
#if !_CCCL_COMPILER(MSVC)
    // with only constexpr param
    using T = cuda::std::__constant_wrapper<cuda::std::plus<>{}>;
    cuda::std::same_as<cuda::std::__constant_wrapper<3>> decltype(auto) result =
      TEST_CALL(T, cuda::std::__cw<1>, cuda::std::__cw<2>);
    static_assert(result == 3);
#endif // !_CCCL_COMPILER(MSVC)
  }

  {
    // todo(dabayer): This is failing with msvc.
#if !_CCCL_COMPILER(MSVC)
    // nullary
    using T                                                                     = cuda::std::__constant_wrapper<[] {
      return 42;
                                                                        }>;
    cuda::std::same_as<cuda::std::__constant_wrapper<42>> decltype(auto) result = TEST_CALL(T, );
    static_assert(result == 42);
#endif // !_CCCL_COMPILER(MSVC)
  }

  {
    // return void with runtime param
    using T = cuda::std::__constant_wrapper<[](int) {}>;
    TEST_CALL(T, 5);
    static_assert(cuda::std::same_as<void, decltype(TEST_CALL(T, 5))>);
  }

  {
    // return void with constexpr param
    using T = cuda::std::__constant_wrapper<[](int) {}>;
    TEST_CALL(T, cuda::std::__cw<5>);
    static_assert(cuda::std::same_as<void, decltype(TEST_CALL(T, cuda::std::__cw<5>))>);
  }

  {
    // nullary return void
    using T = cuda::std::__constant_wrapper<[] {}>;
    TEST_CALL(T, );
    static_assert(cuda::std::same_as<void, decltype(TEST_CALL(T, ))>);
  }

  {
    // move only
    using T = cuda::std::__constant_wrapper<MoveOnlyFn{}>;
    MoveOnly m1(1), m2(2), m3(3);
    cuda::std::same_as<MoveOnly> decltype(auto) result = TEST_CALL(T, m1, cuda::std::move(m2), cuda::std::move(m3));
    assert(result.get() == 6);
  }

  {
    // gcc < 13 fails this test with error:
    //   'fun_ptr' is not a valid template argument of type 'bool (*)(int)' because 'fun_ptr' is not a variable
#if !_CCCL_COMPILER(GCC, <, 13)
    // function pointer
    using T                                        = cuda::std::__constant_wrapper<fun_ptr>;
    cuda::std::same_as<bool> decltype(auto) result = TEST_CALL(T, 5);
    assert(result);
#endif // !_CCCL_COMPILER(GCC, <, 13)
  }

  {
    // gcc < 13 fails this test with error:
    //   'fun_ptr' is not a valid template argument of type 'bool (*)(int)' because 'fun_ptr' is not a variable
#if !_CCCL_COMPILER(GCC, <, 13)
    // function pointer with constexpr param
    using T = cuda::std::__constant_wrapper<fun_ptr>;
    cuda::std::same_as<cuda::std::__constant_wrapper<true>> decltype(auto) result = TEST_CALL(T, cuda::std::__cw<5>);
    static_assert(result);
#endif // !_CCCL_COMPILER(GCC, <, 13)
  }
  {
    // member ptr with runtime param
    using T = cuda::std::__constant_wrapper<&S::member>;
    S s1;
    cuda::std::same_as<int&> decltype(auto) result = TEST_CALL(T, s1);
    assert(result == 42);
    assert(&result == &s1.member);
  }
  {
    // todo: Try to make this work with nvcc
    // member ptr with constexpr param
    using T = cuda::std::__constant_wrapper<&S::member>;
    cuda::std::same_as<cuda::std::__constant_wrapper<42>> decltype(auto) result =
      TEST_CALL(T, cuda::std::__cw<&s_value>);
    static_assert(result == 42);
  }
  {
    // member function ptr with runtime param
    using T = cuda::std::__constant_wrapper<&S::mem_fun>;
    S s1;
    cuda::std::same_as<int> decltype(auto) result = TEST_CALL(T, s1, 8);
    assert(result == 50);
  }
  {
    // todo(dabayer): This is failing with msvc.
#if !_CCCL_COMPILER(MSVC)
    // member function ptr with constexpr param
    using T = cuda::std::__constant_wrapper<&S::mem_fun>;
    cuda::std::same_as<cuda::std::__constant_wrapper<50>> decltype(auto) result =
      TEST_CALL(T, cuda::std::__cw<&s_value>, cuda::std::__cw<8>);
    static_assert(result == 50);
#endif // !_CCCL_COMPILER(MSVC)
  }
  {
    // nvcc < 13.2 fails to compile this test
#if !_CCCL_CUDA_COMPILER(NVCC, <, 13, 2)
    // overload set
    // will always unwrap the constexpr params and call the non-constexpr overload
    using T                                        = cuda::std::__constant_wrapper<OverloadSet{}>;
    cuda::std::same_as<int> decltype(auto) result1 = TEST_CALL(T, 42);
    assert(result1 == 1);
    cuda::std::same_as<cuda::std::__constant_wrapper<1>> decltype(auto) result2 = TEST_CALL(T, cuda::std::__cw<42>);
    static_assert(result2 == 1);
#endif // !_CCCL_CUDA_COMPILER(NVCC, <, 13, 2)
  }

  {
    // return non-structural type
    using T                                                 = cuda::std::__constant_wrapper<ReturnNonStructural{}>;
    cuda::std::same_as<NonStructural> decltype(auto) result = TEST_CALL(T, 5);
    assert(result.get() == 5);
  }

  {
    // return non-structural type with constexpr param
    using T                                                 = cuda::std::__constant_wrapper<ReturnNonStructural{}>;
    cuda::std::same_as<NonStructural> decltype(auto) result = TEST_CALL(T, cuda::std::__cw<5>);
    assert(result.get() == 5);
  }

  {
    // cw only
    // the upwrapping case doesn't work so it falls back to the normal invoke path
    using T                                       = cuda::std::__constant_wrapper<CWOnly{}>;
    cuda::std::same_as<int> decltype(auto) result = TEST_CALL(T, cuda::std::__cw<42>);
    assert(result == 42);
  }

  {
    // just use the call operator
    assert(cuda::std::__cw<[](int i) {
             return i + 1;
           }>(42)
           == 43);
    assert(cuda::std::__cw<[](int i) {
             return i + 1;
           }>(cuda::std::__cw<42>)
           == 43);
  }

  {
// todo(dabayer): This is failing with msvc.
#if !_CCCL_COMPILER(MSVC)
    // with integral_constant, will still call the constexpr path
    using T = cuda::std::__constant_wrapper<cuda::std::plus<>{}>;
    cuda::std::integral_constant<int, 1> ic1;
    cuda::std::integral_constant<int, 2> ic2;
    cuda::std::same_as<cuda::std::__constant_wrapper<3>> decltype(auto) result = TEST_CALL(T, ic1, ic2);
    static_assert(result == 3);
#endif // !_CCCL_COMPILER(MSVC)
  }

  {
// todo(dabayer): This is failing with msvc.
#if !_CCCL_COMPILER(MSVC)
    using T = cuda::std::__constant_wrapper<Poison{}>;
    [[maybe_unused]] cuda::std::same_as<cuda::std::__constant_wrapper<MustBeInt<int>{}>> decltype(auto) result =
      TEST_CALL(T, cuda::std::__cw<5>);
#endif // !_CCCL_COMPILER(MSVC)
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
