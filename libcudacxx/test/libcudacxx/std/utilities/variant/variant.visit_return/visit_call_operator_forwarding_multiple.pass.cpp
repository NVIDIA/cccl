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
// UNSUPPORTED: gcc-6

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

template <typename ReturnType>
__host__ __device__ void test_call_operator_forwarding()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;
  { // test call operator forwarding - multi variant, multi arg
    using V  = cuda::std::variant<int, long, double>;
    using V2 = cuda::std::variant<int*, almost_string>;
    V v(42l);
    V2 v2("hello");
    cuda::std::visit<int>(obj, v, v2);
    assert((Fn::check_call<long&, almost_string&>(CT_NonConst | CT_LValue)));
    cuda::std::visit<ReturnType>(cobj, v, v2);
    assert((Fn::check_call<long&, almost_string&>(CT_Const | CT_LValue)));
    cuda::std::visit<ReturnType>(cuda::std::move(obj), v, v2);
    assert((Fn::check_call<long&, almost_string&>(CT_NonConst | CT_RValue)));
    cuda::std::visit<ReturnType>(cuda::std::move(cobj), v, v2);
    assert((Fn::check_call<long&, almost_string&>(CT_Const | CT_RValue)));
  }
  {
    using V = cuda::std::variant<int, long, double, almost_string>;
    V v1(42l), v2("hello"), v3(101), v4(1.1);
    cuda::std::visit<ReturnType>(obj, v1, v2, v3, v4);
    assert((Fn::check_call<long&, almost_string&, int&, double&>(CT_NonConst | CT_LValue)));
    cuda::std::visit<ReturnType>(cobj, v1, v2, v3, v4);
    assert((Fn::check_call<long&, almost_string&, int&, double&>(CT_Const | CT_LValue)));
    cuda::std::visit<ReturnType>(cuda::std::move(obj), v1, v2, v3, v4);
    assert((Fn::check_call<long&, almost_string&, int&, double&>(CT_NonConst | CT_RValue)));
    cuda::std::visit<ReturnType>(cuda::std::move(cobj), v1, v2, v3, v4);
    assert((Fn::check_call<long&, almost_string&, int&, double&>(CT_Const | CT_RValue)));
  }
}

int main(int, char**)
{
  test_call_operator_forwarding<void>();
  test_call_operator_forwarding<int>();

  return 0;
}
