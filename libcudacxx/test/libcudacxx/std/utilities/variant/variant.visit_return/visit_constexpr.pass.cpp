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
// template <class R, class Visitor, class... Variants>
// constexpr R visit(Visitor&& vis, Variants&&... vars);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

__host__ __device__ void test_constexpr_void()
{
  constexpr ReturnFirst obj{};
  constexpr ReturnArity aobj{};
  {
    using V = cuda::std::variant<int>;
    constexpr V v(42);
    static_assert((cuda::std::visit<void>(obj, v), 42) == 42, "");
  }
  {
    using V = cuda::std::variant<short, long, char>;
    constexpr V v(42l);
    static_assert((cuda::std::visit<void>(obj, v), 42) == 42, "");
  }
  {
    using V1 = cuda::std::variant<int>;
    using V2 = cuda::std::variant<int, char*, long long>;
    using V3 = cuda::std::variant<bool, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert((cuda::std::visit<void>(aobj, v1, v2, v3), 3) == 3, "");
  }
  {
    using V1 = cuda::std::variant<int>;
    using V2 = cuda::std::variant<int, char*, long long>;
    using V3 = cuda::std::variant<void*, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert((cuda::std::visit<void>(aobj, v1, v2, v3), 3) == 3, "");
  }
  {
    using V = cuda::std::variant<int, long, double, int*>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert((cuda::std::visit<void>(aobj, v1, v2, v3, v4), 4) == 4, "");
  }
  {
    using V = cuda::std::variant<int, long, double, long long, int*>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert((cuda::std::visit<void>(aobj, v1, v2, v3, v4), 4) == 4, "");
  }
}

__host__ __device__ void test_constexpr_int()
{
  constexpr ReturnFirst obj{};
  constexpr ReturnArity aobj{};
  {
    using V = cuda::std::variant<int>;
    constexpr V v(42);
    static_assert(cuda::std::visit<int>(obj, v) == 42, "");
  }
  {
    using V = cuda::std::variant<short, long, char>;
    constexpr V v(42l);
    static_assert(cuda::std::visit<int>(obj, v) == 42, "");
  }
  {
    using V1 = cuda::std::variant<int>;
    using V2 = cuda::std::variant<int, char*, long long>;
    using V3 = cuda::std::variant<bool, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert(cuda::std::visit<int>(aobj, v1, v2, v3) == 3, "");
  }
  {
    using V1 = cuda::std::variant<int>;
    using V2 = cuda::std::variant<int, char*, long long>;
    using V3 = cuda::std::variant<void*, int, int>;
    constexpr V1 v1;
    constexpr V2 v2(nullptr);
    constexpr V3 v3;
    static_assert(cuda::std::visit<int>(aobj, v1, v2, v3) == 3, "");
  }
  {
    using V = cuda::std::variant<int, long, double, int*>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert(cuda::std::visit<int>(aobj, v1, v2, v3, v4) == 4, "");
  }
  {
    using V = cuda::std::variant<int, long, double, long long, int*>;
    constexpr V v1(42l), v2(101), v3(nullptr), v4(1.1);
    static_assert(cuda::std::visit<int>(aobj, v1, v2, v3, v4) == 4, "");
  }
}

struct visitor_side_effects
{
  int arg_;
  __host__ __device__ constexpr visitor_side_effects(int arg) noexcept
      : arg_(arg)
  {}
  __host__ __device__ constexpr void operator()(int& x) const noexcept
  {
    x = arg_;
  }
};
__host__ __device__ constexpr int test_constexpr_explicit_side_effect()
{
  cuda::std::variant<int> v = 101;
  cuda::std::visit<void>(visitor_side_effects{202}, v);
  return cuda::std::get<int>(v);
}
static_assert(test_constexpr_explicit_side_effect() == 202, "");

int main(int, char**)
{
  test_constexpr_void();
  test_constexpr_int();

  return 0;
}
