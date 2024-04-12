//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: nvrtc
// UNSUPPORTED: gcc-6

// <cuda/std/tuple>

// template <class T, class Tuple> constexpr T make_from_tuple(Tuple&&);

#include <cuda/std/array>
#include <cuda/std/tuple>
#include <cuda/std/utility>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>
#endif
#include <cuda/std/cassert>

#include "test_macros.h"
#include "type_id.h"

// std::array is explicitly allowed to be initialized with A a = { init-list };.
// Disable the missing braces warning for this reason.
#include "disable_missing_braces_warning.h"

template <class Tuple>
struct ConstexprConstructibleFromTuple
{
  template <class... Args>
  __host__ __device__ explicit constexpr ConstexprConstructibleFromTuple(Args&&... xargs)
      : args{cuda::std::forward<Args>(xargs)...}
  {}
  Tuple args;
};

template <class TupleLike>
struct ConstructibleFromTuple;

template <template <class...> class Tuple, class... Types>
struct ConstructibleFromTuple<Tuple<Types...>>
{
  template <class... Args>
  __host__ __device__ explicit ConstructibleFromTuple(Args&&... xargs)
      : args(xargs...)
      , arg_types(&makeArgumentID<Args&&...>())
  {}
  Tuple<cuda::std::decay_t<Types>...> args;
  TypeID const* arg_types;
};

template <class Tp, size_t N>
struct ConstructibleFromTuple<cuda::std::array<Tp, N>>
{
  template <class... Args>
  __host__ __device__ explicit ConstructibleFromTuple(Args&&... xargs)
      : args{xargs...}
      , arg_types(&makeArgumentID<Args&&...>())
  {}
  cuda::std::array<Tp, N> args;
  TypeID const* arg_types;
};

template <class Tuple>
__host__ __device__ constexpr bool do_constexpr_test(Tuple&& tup)
{
  using RawTuple = cuda::std::decay_t<Tuple>;
  using Tp       = ConstexprConstructibleFromTuple<RawTuple>;
  return cuda::std::make_from_tuple<Tp>(cuda::std::forward<Tuple>(tup)).args == tup;
}

// Needed by do_forwarding_test() since it compares pairs of different types.
template <class T1, class T2, class U1, class U2>
__host__ __device__ inline bool operator==(const cuda::std::pair<T1, T2>& lhs, const cuda::std::pair<U1, U2>& rhs)
{
  return lhs.first == rhs.first && lhs.second == rhs.second;
}

template <class... ExpectTypes, class Tuple>
__host__ __device__ bool do_forwarding_test(Tuple&& tup)
{
  using RawTuple = cuda::std::decay_t<Tuple>;
  using Tp       = ConstructibleFromTuple<RawTuple>;
  const Tp value = cuda::std::make_from_tuple<Tp>(cuda::std::forward<Tuple>(tup));
  return value.args == tup && value.arg_types == &makeArgumentID<ExpectTypes...>();
}

__host__ __device__ void test_constexpr_construction()
{
  {
    constexpr cuda::std::tuple<> tup;
    static_assert(do_constexpr_test(tup), "");
  }
  {
    constexpr cuda::std::tuple<int> tup(42);
    static_assert(do_constexpr_test(tup), "");
  }
  {
    constexpr cuda::std::tuple<int, long, void*> tup(42, 101, nullptr);
    static_assert(do_constexpr_test(tup), "");
  }
  {
    constexpr cuda::std::pair<int, const char*> p(42, "hello world");
    static_assert(do_constexpr_test(p), "");
  }
  {
    using Tuple             = cuda::std::array<int, 3>;
    using ValueTp           = ConstexprConstructibleFromTuple<Tuple>;
    constexpr Tuple arr     = {42, 101, -1};
    constexpr ValueTp value = cuda::std::make_from_tuple<ValueTp>(arr);
    static_assert(value.args[0] == arr[0] && value.args[1] == arr[1] && value.args[2] == arr[2], "");
  }
}

__host__ __device__ void test_perfect_forwarding()
{
  {
    using Tup = cuda::std::tuple<>;
    Tup tup;
    Tup const& ctup = tup;
    assert(do_forwarding_test<>(tup));
    assert(do_forwarding_test<>(ctup));
  }
  {
    using Tup = cuda::std::tuple<int>;
    Tup tup(42);
    Tup const& ctup = tup;
    assert(do_forwarding_test<int&>(tup));
    assert(do_forwarding_test<int const&>(ctup));
    assert(do_forwarding_test<int&&>(cuda::std::move(tup)));
    assert(do_forwarding_test<int const&&>(cuda::std::move(ctup)));
  }
  {
    using Tup  = cuda::std::tuple<int&, const char*, unsigned&&>;
    int x      = 42;
    unsigned y = 101;
    Tup tup(x, "hello world", cuda::std::move(y));
    Tup const& ctup = tup;
    assert((do_forwarding_test<int&, const char*&, unsigned&>(tup)));
    assert((do_forwarding_test<int&, const char* const&, unsigned&>(ctup)));
    assert((do_forwarding_test<int&, const char*&&, unsigned&&>(cuda::std::move(tup))));
    assert((do_forwarding_test<int&, const char* const&&, unsigned&&>(cuda::std::move(ctup))));
  }
  // test with pair<T, U>
  {
    using Tup = cuda::std::pair<int&, const char*>;
    int x     = 42;
    Tup tup(x, "hello world");
    Tup const& ctup = tup;
    assert((do_forwarding_test<int&, const char*&>(tup)));
    assert((do_forwarding_test<int&, const char* const&>(ctup)));
    assert((do_forwarding_test<int&, const char*&&>(cuda::std::move(tup))));
    assert((do_forwarding_test<int&, const char* const&&>(cuda::std::move(ctup))));
  }
  // test with array<T, I>
  {
    using Tup       = cuda::std::array<int, 3>;
    Tup tup         = {42, 101, -1};
    Tup const& ctup = tup;
    assert((do_forwarding_test<int&, int&, int&>(tup)));
    assert((do_forwarding_test<int const&, int const&, int const&>(ctup)));
    assert((do_forwarding_test<int&&, int&&, int&&>(cuda::std::move(tup))));
    assert((do_forwarding_test<int const&&, int const&&, int const&&>(cuda::std::move(ctup))));
  }
}

__host__ __device__ void test_noexcept()
{
  struct NothrowMoveable
  {
    NothrowMoveable() = default;
    __host__ __device__ NothrowMoveable(NothrowMoveable const&) {}
    __host__ __device__ NothrowMoveable(NothrowMoveable&&) noexcept {}
  };
  struct TestType
  {
    __host__ __device__ TestType(int, NothrowMoveable) noexcept {}
    __host__ __device__ TestType(int, int, int) noexcept(false) {}
    __host__ __device__ TestType(long, long, long) noexcept {}
  };
  {
    using Tuple = cuda::std::tuple<int, NothrowMoveable>;
    Tuple tup;
    unused(tup);
    Tuple const& ctup = tup;
    unused(ctup);
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    ASSERT_NOT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(ctup));
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(cuda::std::move(tup)));
  }
  {
    using Tuple = cuda::std::pair<int, NothrowMoveable>;
    Tuple tup;
    unused(tup);
    Tuple const& ctup = tup;
    unused(ctup);
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    ASSERT_NOT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(ctup));
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(cuda::std::move(tup)));
  }
#ifndef TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  {
    using Tuple = cuda::std::tuple<int, int, int>;
    Tuple tup;
    unused(tup);
    ASSERT_NOT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(tup));
    unused(tup);
  }
  {
    using Tuple = cuda::std::tuple<long, long, long>;
    Tuple tup;
    unused(tup);
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(tup));
  }
  {
    using Tuple = cuda::std::array<int, 3>;
    Tuple tup;
    unused(tup);
    ASSERT_NOT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(tup));
  }
#endif // TEST_COMPILER_BROKEN_SMF_NOEXCEPT
  {
    using Tuple = cuda::std::array<long, 3>;
    Tuple tup;
    unused(tup);
    LIBCPP_ASSERT_NOEXCEPT(cuda::std::make_from_tuple<TestType>(tup));
  }
}

int main(int, char**)
{
  test_constexpr_construction();
  test_perfect_forwarding();
  test_noexcept();

  return 0;
}
