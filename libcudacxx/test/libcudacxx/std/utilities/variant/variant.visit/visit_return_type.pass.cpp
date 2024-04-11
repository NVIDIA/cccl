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
// template <class Visitor, class... Variants>
// constexpr see below visit(Visitor&& vis, Variants&&... vars);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
// #include <cuda/std/string>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/std/variant>

#include "test_macros.h"
#include "variant_test_helpers.h"

struct almost_string
{
  const char* ptr;

  __host__ __device__ almost_string(const char* ptr)
      : ptr(ptr)
  {}

  __host__ __device__ friend bool operator==(const almost_string& lhs, const almost_string& rhs)
  {
    return lhs.ptr == rhs.ptr;
  }
};

__host__ __device__ void test_return_type()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;
  unused(cobj);
  { // test call operator forwarding - no variant
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj)), Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj)), const Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj))), Fn&&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj))), const Fn&&>, "");
  }
  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    V v(42);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v)), Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v)), const Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v)), Fn&&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v)), const Fn&&>, "");
    unused(v);
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    V v(42l);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v)), Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v)), const Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v)), Fn&&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v)), const Fn&&>, "");
    unused(v);
  }
  { // test call operator forwarding - multi variant, multi arg
    using V  = cuda::std::variant<int, long, double>;
    using V2 = cuda::std::variant<int*, almost_string>;
    V v(42l);
    V2 v2("hello");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v, v2)), Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v, v2)), const Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v, v2)), Fn&&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v, v2)), const Fn&&>, "");
    unused(v, v2);
  }
  {
    using V = cuda::std::variant<int, long, double, almost_string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v1, v2, v3, v4)), Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v1, v2, v3, v4)), const Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v1, v2, v3, v4)), Fn&&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v1, v2, v3, v4)), const Fn&&>,
                  "");
    unused(v1, v2, v3, v4);
  }
  {
    using V = cuda::std::variant<int, long, double, int*, almost_string>;
    V v1(42l), v2("hello"), v3(nullptr), v4(1.1);
    unused(v1, v2, v3, v4);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(obj, v1, v2, v3, v4)), Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cobj, v1, v2, v3, v4)), const Fn&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(obj), v1, v2, v3, v4)), Fn&&>, "");
    static_assert(cuda::std::is_same_v<decltype(cuda::std::visit(cuda::std::move(cobj), v1, v2, v3, v4)), const Fn&&>,
                  "");
  }
}

int main(int, char**)
{
  test_return_type();

  return 0;
}
