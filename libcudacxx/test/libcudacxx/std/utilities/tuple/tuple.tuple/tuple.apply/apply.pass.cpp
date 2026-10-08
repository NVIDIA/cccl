//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// UNSUPPORTED: force-tile
// error: function-to-pointer decay is unsupported in tile code
// error: taking address of a function is unsupported in tile code

// <cuda/std/tuple>

// template <class F, class T> constexpr decltype(auto) apply(F &&, T &&)

// Test with different ref/ptr/cv qualified argument types.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "test_macros.h"
#include "type_id.h"

// cuda::std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
TEST_DIAG_SUPPRESS_GCC("-Wmissing-braces")
TEST_DIAG_SUPPRESS_CLANG("-Wmissing-braces")

TEST_HOST_DEVICE_FUNC constexpr int constexpr_sum_fn()
{
  return 0;
}

template <class... Ints>
TEST_HOST_DEVICE_FUNC constexpr int constexpr_sum_fn(int x1, Ints... rest)
{
  return x1 + constexpr_sum_fn(rest...);
}

struct ConstexprSumT
{
  constexpr ConstexprSumT() = default;

  template <class... Ints>
  TEST_HOST_DEVICE_FUNC constexpr int operator()(Ints... values) const
  {
    return constexpr_sum_fn(values...);
  }
};

TEST_HOST_DEVICE_FUNC void test_constexpr_evaluation()
{
  constexpr ConstexprSumT sum_obj{};
  {
    using Tup = cuda::std::tuple<>;
    using Fn  = int (&)();
    constexpr Tup t;
    static_assert(cuda::std::apply(static_cast<Fn>(constexpr_sum_fn), t) == 0);
    static_assert(cuda::std::apply(sum_obj, t) == 0);
  }
  {
    using Tup = cuda::std::tuple<int>;
    using Fn  = int (&)(int);
    constexpr Tup t(42);
    static_assert(cuda::std::apply(static_cast<Fn>(constexpr_sum_fn), t) == 42);
    static_assert(cuda::std::apply(sum_obj, t) == 42);
  }
  {
    using Tup = cuda::std::tuple<int, long>;
    using Fn  = int (&)(int, int);
    constexpr Tup t(42, 101);
    static_assert(cuda::std::apply(static_cast<Fn>(constexpr_sum_fn), t) == 143);
    static_assert(cuda::std::apply(sum_obj, t) == 143);
  }
  {
    using Tup = cuda::std::pair<int, long>;
    using Fn  = int (&)(int, int);
    constexpr Tup t(42, 101);
    static_assert(cuda::std::apply(static_cast<Fn>(constexpr_sum_fn), t) == 143);
    static_assert(cuda::std::apply(sum_obj, t) == 143);
  }
  {
    using Tup = cuda::std::tuple<int, long, int>;
    using Fn  = int (&)(int, int, int);
    constexpr Tup t(42, 101, -1);
    static_assert(cuda::std::apply(static_cast<Fn>(constexpr_sum_fn), t) == 142);
    static_assert(cuda::std::apply(sum_obj, t) == 142);
  }
  {
    using Tup       = cuda::std::array<int, 3>;
    using Fn        = int (&)(int, int, int);
    constexpr Tup t = {42, 101, -1};
    static_assert(cuda::std::apply(static_cast<Fn>(constexpr_sum_fn), t) == 142);
    static_assert(cuda::std::apply(sum_obj, t) == 142);
  }
}

enum CallQuals
{
  CQ_None,
  CQ_LValue,
  CQ_ConstLValue,
  CQ_RValue,
  CQ_ConstRValue
};

template <class Tuple>
struct CallInfo
{
  CallQuals quals;
  TypeID const* arg_types;
  Tuple args;

  template <class... Args>
  TEST_HOST_DEVICE_FUNC CallInfo(CallQuals q, Args&&... xargs)
      : quals(q)
      , arg_types(&makeArgumentID<Args&&...>())
      , args(cuda::std::forward<Args>(xargs)...)
  {}
};

template <class... Args>
TEST_HOST_DEVICE_FUNC inline CallInfo<decltype(cuda::std::forward_as_tuple(cuda::std::declval<Args>()...))>
makeCallInfo(CallQuals quals, Args&&... args)
{
  return {quals, cuda::std::forward<Args>(args)...};
}

struct TrackedCallable
{
  TrackedCallable() = default;

  template <class... Args>
  TEST_HOST_DEVICE_FUNC auto operator()(Args&&... xargs) &
  {
    return makeCallInfo(CQ_LValue, cuda::std::forward<Args>(xargs)...);
  }

  template <class... Args>
  TEST_HOST_DEVICE_FUNC auto operator()(Args&&... xargs) const&
  {
    return makeCallInfo(CQ_ConstLValue, cuda::std::forward<Args>(xargs)...);
  }

  template <class... Args>
  TEST_HOST_DEVICE_FUNC auto operator()(Args&&... xargs) &&
  {
    return makeCallInfo(CQ_RValue, cuda::std::forward<Args>(xargs)...);
  }

  template <class... Args>
  TEST_HOST_DEVICE_FUNC auto operator()(Args&&... xargs) const&&
  {
    return makeCallInfo(CQ_ConstRValue, cuda::std::forward<Args>(xargs)...);
  }
};

template <class... ExpectArgs, class Tuple>
TEST_HOST_DEVICE_FUNC void check_apply_quals_and_types(Tuple&& t)
{
  TypeID const* const expect_args = &makeArgumentID<ExpectArgs...>();
  TrackedCallable obj;
  TrackedCallable const& cobj = obj;
  {
    auto ret = cuda::std::apply(obj, cuda::std::forward<Tuple>(t));
    assert(ret.quals == CQ_LValue);
    assert(ret.arg_types == expect_args);
    assert(ret.args == t);
  }
  {
    auto ret = cuda::std::apply(cobj, cuda::std::forward<Tuple>(t));
    assert(ret.quals == CQ_ConstLValue);
    assert(ret.arg_types == expect_args);
    assert(ret.args == t);
  }
  {
    auto ret = cuda::std::apply(cuda::std::move(obj), cuda::std::forward<Tuple>(t));
    assert(ret.quals == CQ_RValue);
    assert(ret.arg_types == expect_args);
    assert(ret.args == t);
  }
  {
    auto ret = cuda::std::apply(cuda::std::move(cobj), cuda::std::forward<Tuple>(t));
    assert(ret.quals == CQ_ConstRValue);
    assert(ret.arg_types == expect_args);
    assert(ret.args == t);
  }
}

TEST_HOST_DEVICE_FUNC void test_call_quals_and_arg_types()
{
  using Tup   = cuda::std::tuple<int, int const&, unsigned&&>;
  const int x = 42;
  unsigned y  = 101;
  Tup t(-1, x, cuda::std::move(y));
  Tup const& ct = t;
  check_apply_quals_and_types<int&, int const&, unsigned&>(t);
  check_apply_quals_and_types<int const&, int const&, unsigned&>(ct);
  check_apply_quals_and_types<int&&, int const&, unsigned&&>(cuda::std::move(t));
  check_apply_quals_and_types<int const&&, int const&, unsigned&&>(cuda::std::move(ct));
}

struct NothrowMoveable
{
  NothrowMoveable() noexcept = default;
  TEST_HOST_DEVICE_FUNC NothrowMoveable(NothrowMoveable const&) noexcept(false) {}
  TEST_HOST_DEVICE_FUNC NothrowMoveable(NothrowMoveable&&) noexcept {}
};

template <bool IsNoexcept>
struct TestNoexceptCallable
{
  template <class... Args>
  TEST_HOST_DEVICE_FUNC NothrowMoveable operator()(Args...) const noexcept(IsNoexcept)
  {
    return {};
  }
};

TEST_HOST_DEVICE_FUNC void test_noexcept()
{
  TestNoexceptCallable<true> nec;
  [[maybe_unused]] TestNoexceptCallable<false> tc;
  {
    // test that the functions noexcept-ness is propagated
    using Tup = cuda::std::tuple<int, const char*, long>;
    [[maybe_unused]] Tup t;
    static_assert(noexcept(cuda::std::apply(nec, t)));
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(cuda::std::apply(tc, t)));
#endif // TEST_COMPILER(NVHPC)
  }
  {
    // test that the noexcept-ness of the argument conversions is checked.
    using Tup = cuda::std::tuple<NothrowMoveable, int>;
    [[maybe_unused]] Tup t;
#if !TEST_COMPILER(NVHPC)
    static_assert(!noexcept(cuda::std::apply(nec, t)));
#endif // TEST_COMPILER(NVHPC)
    static_assert(noexcept(cuda::std::apply(nec, cuda::std::move(t))));
  }
}

namespace ReturnTypeTest
{
#ifdef __CUDA_ARCH__
__constant__ int my_int = 42;
#else
static int my_int = 42;
#endif

template <int N>
struct index
{};

TEST_HOST_DEVICE_FUNC void f(index<0>) {}

TEST_HOST_DEVICE_FUNC int f(index<1>)
{
  return 0;
}

TEST_HOST_DEVICE_FUNC int& f(index<2>)
{
  return static_cast<int&>(my_int);
}
TEST_HOST_DEVICE_FUNC int const& f(index<3>)
{
  return static_cast<int const&>(my_int);
}
TEST_HOST_DEVICE_FUNC int volatile& f(index<4>)
{
  return static_cast<int volatile&>(my_int);
}
TEST_HOST_DEVICE_FUNC int const volatile& f(index<5>)
{
  return static_cast<int const volatile&>(my_int);
}

TEST_HOST_DEVICE_FUNC int&& f(index<6>)
{
  return static_cast<int&&>(my_int);
}
TEST_HOST_DEVICE_FUNC int const&& f(index<7>)
{
  return static_cast<int const&&>(my_int);
}
TEST_HOST_DEVICE_FUNC int volatile&& f(index<8>)
{
  return static_cast<int volatile&&>(my_int);
}
TEST_HOST_DEVICE_FUNC int const volatile&& f(index<9>)
{
  return static_cast<int const volatile&&>(my_int);
}

TEST_HOST_DEVICE_FUNC int* f(index<10>)
{
  return static_cast<int*>(&my_int);
}
TEST_HOST_DEVICE_FUNC int const* f(index<11>)
{
  return static_cast<int const*>(&my_int);
}
TEST_HOST_DEVICE_FUNC int volatile* f(index<12>)
{
  return static_cast<int volatile*>(&my_int);
}
TEST_HOST_DEVICE_FUNC int const volatile* f(index<13>)
{
  return static_cast<int const volatile*>(&my_int);
}

template <int Func, class Expect>
TEST_HOST_DEVICE_FUNC void test()
{
  using RawInvokeResult = decltype(f(index<Func>{}));
  static_assert(cuda::std::is_same<RawInvokeResult, Expect>::value);
  using FnType               = RawInvokeResult (*)(index<Func>);
  [[maybe_unused]] FnType fn = f;
  [[maybe_unused]] cuda::std::tuple<index<Func>> t;
  using InvokeResult = decltype(cuda::std::apply(fn, t));
  static_assert(cuda::std::is_same<InvokeResult, Expect>::value);
}
} // end namespace ReturnTypeTest

// Callables that accept only one specific value category. A forwarding reference would accept every category and
// hide a missing cv-ref on the tuple element.
struct ApplyLvalueInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int&) const {}
};

struct ApplyConstLvalueInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int const&) const {}
};

struct ApplyRvalueInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int&&) const {}
};

struct ApplyConstRvalueInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int const&&) const {}
};

struct ApplyLvalueIntInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int&, int&) const {}
};

struct ApplyConstLvalueIntInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int const&, int const&) const {}
};

struct ApplyRvalueIntInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int&&, int&&) const {}
};

struct ApplyConstRvalueIntInt
{
  TEST_HOST_DEVICE_FUNC void operator()(int const&&, int const&&) const {}
};

struct ApplyNullary
{
  TEST_HOST_DEVICE_FUNC void operator()() const {}
};

struct ApplyLvalueConstLvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int&, int const&) const {}
};

struct ApplyRvalueConstRvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int&&, int const&&) const {}
};

struct ApplyMixedLvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int&, int const&, unsigned&) const {}
};

struct ApplyMixedConstLvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int const&, int const&, unsigned&) const {}
};

struct ApplyMixedRvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int&&, int const&, unsigned&&) const {}
};

struct ApplyMixedConstRvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int const&&, int const&, unsigned&&) const {}
};

struct ApplyVolatileLvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int volatile&) const {}
};

struct ApplyVolatileRvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int volatile&&) const {}
};

struct ApplyConstVolatileLvalue
{
  TEST_HOST_DEVICE_FUNC void operator()(int const volatile&) const {}
};

// Tuple has a single value element. Each cv-ref qualification of the tuple must surface as the matching qualification
// of that element.
template <class Tuple>
TEST_HOST_DEVICE_FUNC constexpr void test_can_apply_value_element()
{
  static_assert(cuda::std::__can_apply<ApplyLvalueInt, Tuple&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, Tuple&>);
  static_assert(!cuda::std::__can_apply<ApplyRvalueInt, Tuple&>);
  static_assert(!cuda::std::__can_apply<ApplyConstRvalueInt, Tuple&>);

  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, Tuple const&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, Tuple const&>);
  static_assert(!cuda::std::__can_apply<ApplyRvalueInt, Tuple const&>);
  static_assert(!cuda::std::__can_apply<ApplyConstRvalueInt, Tuple const&>);

  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, Tuple&&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, Tuple&&>);
  static_assert(cuda::std::__can_apply<ApplyRvalueInt, Tuple&&>);
  static_assert(cuda::std::__can_apply<ApplyConstRvalueInt, Tuple&&>);

  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, Tuple const&&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, Tuple const&&>);
  static_assert(!cuda::std::__can_apply<ApplyRvalueInt, Tuple const&&>);
  static_assert(cuda::std::__can_apply<ApplyConstRvalueInt, Tuple const&&>);
}

template <class Tuple>
TEST_HOST_DEVICE_FUNC constexpr void test_can_apply_two_value_elements()
{
  static_assert(cuda::std::__can_apply<ApplyLvalueIntInt, Tuple&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueIntInt, Tuple&>);
  static_assert(!cuda::std::__can_apply<ApplyRvalueIntInt, Tuple&>);
  static_assert(!cuda::std::__can_apply<ApplyConstRvalueIntInt, Tuple&>);

  static_assert(!cuda::std::__can_apply<ApplyLvalueIntInt, Tuple const&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueIntInt, Tuple const&>);
  static_assert(!cuda::std::__can_apply<ApplyRvalueIntInt, Tuple const&>);
  static_assert(!cuda::std::__can_apply<ApplyConstRvalueIntInt, Tuple const&>);

  static_assert(!cuda::std::__can_apply<ApplyLvalueIntInt, Tuple&&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueIntInt, Tuple&&>);
  static_assert(cuda::std::__can_apply<ApplyRvalueIntInt, Tuple&&>);
  static_assert(cuda::std::__can_apply<ApplyConstRvalueIntInt, Tuple&&>);

  static_assert(!cuda::std::__can_apply<ApplyLvalueIntInt, Tuple const&&>);
  static_assert(cuda::std::__can_apply<ApplyConstLvalueIntInt, Tuple const&&>);
  static_assert(!cuda::std::__can_apply<ApplyRvalueIntInt, Tuple const&&>);
  static_assert(cuda::std::__can_apply<ApplyConstRvalueIntInt, Tuple const&&>);
}

TEST_HOST_DEVICE_FUNC constexpr void test_can_apply_cvref()
{
  test_can_apply_value_element<cuda::std::tuple<int>>();
  test_can_apply_value_element<cuda::std::array<int, 1>>();
  test_can_apply_two_value_elements<cuda::std::tuple<int, int>>();
  test_can_apply_two_value_elements<cuda::std::pair<int, int>>();
  test_can_apply_two_value_elements<cuda::std::array<int, 2>>();

  static_assert(cuda::std::__can_apply<ApplyNullary, cuda::std::tuple<>&>);
  static_assert(cuda::std::__can_apply<ApplyNullary, cuda::std::tuple<> const&>);
  static_assert(cuda::std::__can_apply<ApplyNullary, cuda::std::tuple<>&&>);
  static_assert(cuda::std::__can_apply<ApplyNullary, cuda::std::tuple<> const&&>);
  static_assert(cuda::std::__can_apply<ApplyNullary, cuda::std::array<int, 0>&>);
  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, cuda::std::tuple<>&>);
  static_assert(!cuda::std::__can_apply<ApplyNullary, cuda::std::tuple<int>&>);

  // Reference elements keep their own value category. The tuple's cv-ref does not rebind them.
  {
    using RefTup = cuda::std::tuple<int&>;
    static_assert(cuda::std::__can_apply<ApplyLvalueInt, RefTup&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueInt, RefTup const&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueInt, RefTup&&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueInt, RefTup const&&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueInt, RefTup&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueInt, RefTup&&>);
    static_assert(!cuda::std::__can_apply<ApplyConstRvalueInt, RefTup const&&>);
  }
  {
    using CRefTup = cuda::std::tuple<int const&>;
    static_assert(!cuda::std::__can_apply<ApplyLvalueInt, CRefTup&>);
    static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, CRefTup&>);
    static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, CRefTup const&>);
    static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, CRefTup&&>);
    static_assert(cuda::std::__can_apply<ApplyConstLvalueInt, CRefTup const&&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueInt, CRefTup&&>);
    static_assert(!cuda::std::__can_apply<ApplyConstRvalueInt, CRefTup&&>);
  }
  {
    using RRefTup = cuda::std::tuple<int&&>;
    static_assert(cuda::std::__can_apply<ApplyLvalueInt, RRefTup&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueInt, RRefTup const&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueInt, RRefTup&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueInt, RRefTup const&>);
    static_assert(!cuda::std::__can_apply<ApplyLvalueInt, RRefTup&&>);
    static_assert(!cuda::std::__can_apply<ApplyLvalueInt, RRefTup const&&>);
    static_assert(cuda::std::__can_apply<ApplyRvalueInt, RRefTup&&>);
    static_assert(cuda::std::__can_apply<ApplyRvalueInt, RRefTup const&&>);
  }
  {
    using RefPair = cuda::std::pair<int&, int const&>;
    static_assert(cuda::std::__can_apply<ApplyLvalueConstLvalue, RefPair&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueConstLvalue, RefPair const&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueConstLvalue, RefPair&&>);
    static_assert(cuda::std::__can_apply<ApplyLvalueConstLvalue, RefPair const&&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueConstRvalue, RefPair&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueConstRvalue, RefPair&&>);
    static_assert(!cuda::std::__can_apply<ApplyRvalueConstRvalue, RefPair const&&>);
  }

  // Same element mix as test_call_quals_and_arg_types. Each tuple cv-ref must select a different callable.
  {
    using Mixed = cuda::std::tuple<int, int const&, unsigned&&>;
    static_assert(cuda::std::__can_apply<ApplyMixedLvalue, Mixed&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedLvalue, Mixed const&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedLvalue, Mixed&&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedLvalue, Mixed const&&>);

    static_assert(cuda::std::__can_apply<ApplyMixedConstLvalue, Mixed&>);
    static_assert(cuda::std::__can_apply<ApplyMixedConstLvalue, Mixed const&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedConstLvalue, Mixed&&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedConstLvalue, Mixed const&&>);

    static_assert(!cuda::std::__can_apply<ApplyMixedRvalue, Mixed&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedRvalue, Mixed const&>);
    static_assert(cuda::std::__can_apply<ApplyMixedRvalue, Mixed&&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedRvalue, Mixed const&&>);

    static_assert(!cuda::std::__can_apply<ApplyMixedConstRvalue, Mixed&>);
    static_assert(!cuda::std::__can_apply<ApplyMixedConstRvalue, Mixed const&>);
    static_assert(cuda::std::__can_apply<ApplyMixedConstRvalue, Mixed&&>);
    static_assert(cuda::std::__can_apply<ApplyMixedConstRvalue, Mixed const&&>);
  }

  {
    using VolTup = cuda::std::tuple<int volatile>;
    static_assert(cuda::std::__can_apply<ApplyVolatileLvalue, VolTup&>);
    static_assert(!cuda::std::__can_apply<ApplyVolatileRvalue, VolTup&>);
    static_assert(!cuda::std::__can_apply<ApplyLvalueInt, VolTup&>);
    static_assert(cuda::std::__can_apply<ApplyVolatileRvalue, VolTup&&>);
    static_assert(!cuda::std::__can_apply<ApplyVolatileLvalue, VolTup&&>);
    static_assert(cuda::std::__can_apply<ApplyConstVolatileLvalue, VolTup const&>);
    static_assert(!cuda::std::__can_apply<ApplyVolatileLvalue, VolTup const&>);
    static_assert(!cuda::std::__can_apply<ApplyVolatileRvalue, VolTup const&&>);
  }

  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, int>);
  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, int&>);
  static_assert(!cuda::std::__can_apply<ApplyNullary, int>);
  static_assert(!cuda::std::__can_apply<ApplyLvalueInt, cuda::std::tuple<int, int>&>);
}

TEST_HOST_DEVICE_FUNC void test_return_type()
{
  using ReturnTypeTest::test;
  test<0, void>();
  test<1, int>();
  test<2, int&>();
  test<3, int const&>();
  test<4, int volatile&>();
  test<5, int const volatile&>();
  test<6, int&&>();
  test<7, int const&&>();
  test<8, int volatile&&>();
  test<9, int const volatile&&>();
  test<10, int*>();
  test<11, int const*>();
  test<12, int volatile*>();
  test<13, int const volatile*>();
}

int main(int, char**)
{
  test_constexpr_evaluation();
  test_call_quals_and_arg_types();
  test_return_type();
  test_noexcept();
  test_can_apply_cvref();

  return 0;
}
