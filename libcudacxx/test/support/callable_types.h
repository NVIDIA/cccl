//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_CALLABLE_TYPES_H
#define TEST_CALLABLE_TYPES_H

#include "test_macros.h"
// #include "type_id.h"

///////////////////////////////////////////////////////////////////////////////
//                       CALLABLE TEST TYPES
///////////////////////////////////////////////////////////////////////////////

__host__ __device__ constexpr bool returns_true()
{
  return true;
}

template <class Ret>
struct MoveOnlyCallable
{
  MoveOnlyCallable(MoveOnlyCallable const&) = delete;
  __host__ __device__ constexpr MoveOnlyCallable(MoveOnlyCallable&& other)
      : value(other.value)
  {
    other.value = !other.value;
  }

  template <class... Args>
  __host__ __device__ constexpr Ret operator()(Args&&...)
  {
    return Ret{value};
  }

  __host__ __device__ constexpr explicit MoveOnlyCallable(bool x)
      : value(x)
  {}
  Ret value;
};

template <class Ret>
struct CopyCallable
{
  __host__ __device__ constexpr CopyCallable(CopyCallable const& other)
      : value(other.value)
  {}

  __host__ __device__ constexpr CopyCallable(CopyCallable&& other)
      : value(other.value)
  {
    other.value = !other.value;
  }

  template <class... Args>
  __host__ __device__ constexpr Ret operator()(Args&&...)
  {
    return Ret{value};
  }

  __host__ __device__ constexpr explicit CopyCallable(bool x)
      : value(x)
  {}
  Ret value;
};

template <class Ret>
struct ConstCallable
{
  __host__ __device__ constexpr ConstCallable(ConstCallable const& other)
      : value(other.value)
  {}

  __host__ __device__ constexpr ConstCallable(ConstCallable&& other)
      : value(other.value)
  {
    other.value = !other.value;
  }

  template <class... Args>
  __host__ __device__ constexpr Ret operator()(Args&&...) const
  {
    return Ret{value};
  }

  __host__ __device__ constexpr explicit ConstCallable(bool x)
      : value(x)
  {}
  Ret value;
};

template <class Ret>
struct NoExceptCallable
{
  __host__ __device__ constexpr NoExceptCallable(NoExceptCallable const& other)
      : value(other.value)
  {}

  template <class... Args>
  __host__ __device__ constexpr Ret operator()(Args&&...) noexcept
  {
    return Ret{value};
  }

  template <class... Args>
  __host__ __device__ constexpr Ret operator()(Args&&...) const noexcept
  {
    return Ret{value};
  }

  __host__ __device__ constexpr explicit NoExceptCallable(bool x)
      : value(x)
  {}
  Ret value;
};

struct CopyAssignableWrapper
{
  constexpr CopyAssignableWrapper(CopyAssignableWrapper const&)            = default;
  constexpr CopyAssignableWrapper(CopyAssignableWrapper&&)                 = default;
  constexpr CopyAssignableWrapper& operator=(CopyAssignableWrapper const&) = default;
  constexpr CopyAssignableWrapper& operator=(CopyAssignableWrapper&&)      = default;

  template <class... Args>
  __host__ __device__ constexpr bool operator()(Args&&...)
  {
    return value;
  }

  __host__ __device__ constexpr explicit CopyAssignableWrapper(bool x)
      : value(x)
  {}
  bool value;
};

struct MoveAssignableWrapper
{
  constexpr MoveAssignableWrapper(MoveAssignableWrapper const&)            = delete;
  constexpr MoveAssignableWrapper(MoveAssignableWrapper&&)                 = default;
  constexpr MoveAssignableWrapper& operator=(MoveAssignableWrapper const&) = delete;
  constexpr MoveAssignableWrapper& operator=(MoveAssignableWrapper&&)      = default;

  template <class... Args>
  __host__ __device__ constexpr bool operator()(Args&&...)
  {
    return value;
  }

  __host__ __device__ constexpr explicit MoveAssignableWrapper(bool x)
      : value(x)
  {}
  bool value;
};

struct MemFunCallable
{
  __host__ __device__ constexpr explicit MemFunCallable(bool x)
      : value(x)
  {}

  __host__ __device__ constexpr bool return_value() const
  {
    return value;
  }
  __host__ __device__ constexpr bool return_value_nc()
  {
    return value;
  }
  bool value;
};

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
#if 0
struct ForwardingCallObject {
  struct State {
    CallType      last_call_type = CT_None;
    TypeID const& (*last_call_args)() = nullptr;

    template <class ...Args>
    __host__ __device__ constexpr void set_call(CallType type) {
      assert(last_call_type == CT_None);
      assert(last_call_args == nullptr);
      last_call_type = type;
      last_call_args = &makeArgumentID<Args...>;
    }

    template <class ...Args>
    __host__ __device__ constexpr bool check_call(CallType type) {
      bool result =
           last_call_type == type
        && last_call_args
        && *last_call_args == &makeArgumentID<Args...>;
      last_call_type = CT_None;
      last_call_args = nullptr;
      return result;
    }
  };

  State *st_;

  __host__ __device__ explicit constexpr ForwardingCallObject(State& st) : st_(&st) {}

  template <class ...Args>
  __host__ __device__ constexpr bool operator()(Args&&...) & {
      st_->set_call<Args&&...>(CT_NonConst | CT_LValue);
      return true;
  }

  template <class ...Args>
  __host__ __device__ constexpr bool operator()(Args&&...) const & {
      st_->set_call<Args&&...>(CT_Const | CT_LValue);
      return true;
  }

  // Don't allow the call operator to be invoked as an rvalue.
  template <class ...Args>
  __host__ __device__ constexpr bool operator()(Args&&...) && {
      st_->set_call<Args&&...>(CT_NonConst | CT_RValue);
      return true;
  }

  template <class ...Args>
  __host__ __device__ constexpr bool operator()(Args&&...) const && {
      st_->set_call<Args&&...>(CT_Const | CT_RValue);
      return true;
  }
};
#endif

#endif // TEST_CALLABLE_TYPES_H
