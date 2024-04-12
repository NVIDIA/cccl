// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SUPPORT_VARIANT_TEST_HELPERS_H
#define SUPPORT_VARIANT_TEST_HELPERS_H

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"
#include "type_id.h"

#if TEST_STD_VER <= 2011
#  error This file requires C++14
#endif

// FIXME: Currently the variant<T&> tests are disabled using this macro.
#define TEST_VARIANT_HAS_NO_REFERENCES
#ifdef _LIBCPP_ENABLE_NARROWING_CONVERSIONS_IN_VARIANT
#  define TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
#endif

#ifdef TEST_VARIANT_ALLOWS_NARROWING_CONVERSIONS
constexpr bool VariantAllowsNarrowingConversions = true;
#else
constexpr bool VariantAllowsNarrowingConversions = false;
#endif

#ifndef TEST_HAS_NO_EXCEPTIONS
struct CopyThrows
{
  CopyThrows() = default;
  CopyThrows(CopyThrows const&)
  {
    throw 42;
  }
  CopyThrows& operator=(CopyThrows const&)
  {
    throw 42;
  }
};

struct MoveThrows
{
  static int alive;
  MoveThrows()
  {
    ++alive;
  }
  MoveThrows(MoveThrows const&)
  {
    ++alive;
  }
  MoveThrows(MoveThrows&&)
  {
    throw 42;
  }
  MoveThrows& operator=(MoveThrows const&)
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
  MakeEmptyT(MakeEmptyT const&)
  {
    ++alive;
    // Don't throw from the copy constructor since variant's assignment
    // operator performs a copy before committing to the assignment.
  }
  MakeEmptyT(MakeEmptyT&&)
  {
    throw 42;
  }
  MakeEmptyT& operator=(MakeEmptyT const&)
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
static_assert(cuda::std::is_swappable_v<MakeEmptyT>, ""); // required for test

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
#endif // !TEST_HAS_NO_EXCEPTIONS

enum CallType : unsigned
{
  CT_None,
  CT_NonConst = 1,
  CT_Const    = 2,
  CT_LValue   = 4,
  CT_RValue   = 8
};

__host__ __device__ inline constexpr CallType operator|(CallType LHS, CallType RHS)
{
  return static_cast<CallType>(static_cast<unsigned>(LHS) | static_cast<unsigned>(RHS));
}

struct ForwardingCallObject
{
  template <class... Args>
  __host__ __device__ ForwardingCallObject& operator()(Args&&...) &
  {
    set_call<Args&&...>(CT_NonConst | CT_LValue);
    return *this;
  }

  template <class... Args>
  __host__ __device__ const ForwardingCallObject& operator()(Args&&...) const&
  {
    set_call<Args&&...>(CT_Const | CT_LValue);
    return *this;
  }

  template <class... Args>
  __host__ __device__ ForwardingCallObject&& operator()(Args&&...) &&
  {
    set_call<Args&&...>(CT_NonConst | CT_RValue);
    return cuda::std::move(*this);
  }

  template <class... Args>
  __host__ __device__ const ForwardingCallObject&& operator()(Args&&...) const&&
  {
    set_call<Args&&...>(CT_Const | CT_RValue);
    return cuda::std::move(*this);
  }

  template <class... Args>
  __host__ __device__ static void set_call(CallType type)
  {
    assert(last_call_type() == CT_None);
    assert(last_call_args() == nullptr);
    last_call_type() = type;
    last_call_args() = cuda::std::addressof(makeArgumentID<Args...>());
  }

  template <class... Args>
  __host__ __device__ static bool check_call(CallType type)
  {
    bool result      = last_call_type() == type && last_call_args() && *last_call_args() == makeArgumentID<Args...>();
    last_call_type() = CT_None;
    last_call_args() = nullptr;
    return result;
  }

  // To check explicit return type for visit<R>
  __host__ __device__ constexpr operator int() const
  {
    return 0;
  }

  STATIC_MEMBER_VAR(last_call_type, CallType);
  STATIC_MEMBER_VAR(last_call_args, const TypeID*);
};

struct ReturnFirst
{
  template <class... Args>
  __host__ __device__ constexpr int operator()(int f, Args&&...) const
  {
    return f;
  }
};

struct ReturnArity
{
  template <class... Args>
  __host__ __device__ constexpr int operator()(Args&&...) const
  {
    return sizeof...(Args);
  }
};

#endif // SUPPORT_VARIANT_TEST_HELPERS_H
