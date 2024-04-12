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

__host__ __device__ void test_call_operator_forwarding()
{
  using Fn = ForwardingCallObject;
  Fn obj{};
  const Fn& cobj = obj;
  { // test call operator forwarding - no variant
    cuda::std::visit(obj);
    assert(Fn::check_call<>(CT_NonConst | CT_LValue));
    cuda::std::visit(cobj);
    assert(Fn::check_call<>(CT_Const | CT_LValue));
    cuda::std::visit(cuda::std::move(obj));
    assert(Fn::check_call<>(CT_NonConst | CT_RValue));
    cuda::std::visit(cuda::std::move(cobj));
    assert(Fn::check_call<>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, single arg
    using V = cuda::std::variant<int>;
    V v(42);
    cuda::std::visit(obj, v);
    assert(Fn::check_call<int&>(CT_NonConst | CT_LValue));
    cuda::std::visit(cobj, v);
    assert(Fn::check_call<int&>(CT_Const | CT_LValue));
    cuda::std::visit(cuda::std::move(obj), v);
    assert(Fn::check_call<int&>(CT_NonConst | CT_RValue));
    cuda::std::visit(cuda::std::move(cobj), v);
    assert(Fn::check_call<int&>(CT_Const | CT_RValue));
  }
  { // test call operator forwarding - single variant, multi arg
    using V = cuda::std::variant<int, long, double>;
    V v(42l);
    cuda::std::visit(obj, v);
    assert(Fn::check_call<long&>(CT_NonConst | CT_LValue));
    cuda::std::visit(cobj, v);
    assert(Fn::check_call<long&>(CT_Const | CT_LValue));
    cuda::std::visit(cuda::std::move(obj), v);
    assert(Fn::check_call<long&>(CT_NonConst | CT_RValue));
    cuda::std::visit(cuda::std::move(cobj), v);
    assert(Fn::check_call<long&>(CT_Const | CT_RValue));
  }
}

int main(int, char**)
{
  test_call_operator_forwarding();

  return 0;
}
