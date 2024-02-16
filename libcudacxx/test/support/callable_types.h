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
//#include "type_id.h"

///////////////////////////////////////////////////////////////////////////////
//                       CALLABLE TEST TYPES
///////////////////////////////////////////////////////////////////////////////

TEST_HOST_DEVICE constexpr bool returns_true() { return true; }

template <class Ret>
struct MoveOnlyCallable {
  MoveOnlyCallable(MoveOnlyCallable const&) = delete;
  TEST_HOST_DEVICE constexpr MoveOnlyCallable(MoveOnlyCallable&& other)
      : value(other.value)
  { other.value = !other.value; }

  template <class ...Args>
  TEST_HOST_DEVICE constexpr Ret operator()(Args&&...) { return Ret{value}; }

  TEST_HOST_DEVICE constexpr explicit MoveOnlyCallable(bool x) : value(x) {}
  Ret value;
};

template <class Ret>
struct CopyCallable {
  TEST_HOST_DEVICE constexpr CopyCallable(CopyCallable const& other)
      : value(other.value) {}

  TEST_HOST_DEVICE constexpr CopyCallable(CopyCallable&& other)
      : value(other.value) { other.value = !other.value; }

  template <class ...Args>
  TEST_HOST_DEVICE constexpr Ret operator()(Args&&...) { return Ret{value}; }

  TEST_HOST_DEVICE constexpr explicit CopyCallable(bool x) : value(x)  {}
  Ret value;
};


template <class Ret>
struct ConstCallable {
  TEST_HOST_DEVICE constexpr ConstCallable(ConstCallable const& other)
      : value(other.value) {}

  TEST_HOST_DEVICE constexpr ConstCallable(ConstCallable&& other)
      : value(other.value) { other.value = !other.value; }

  template <class ...Args>
  TEST_HOST_DEVICE constexpr Ret operator()(Args&&...) const { return Ret{value}; }

  TEST_HOST_DEVICE constexpr explicit ConstCallable(bool x) : value(x)  {}
  Ret value;
};



template <class Ret>
struct NoExceptCallable {
  TEST_HOST_DEVICE constexpr NoExceptCallable(NoExceptCallable const& other)
      : value(other.value) {}

  template <class ...Args>
  TEST_HOST_DEVICE constexpr Ret operator()(Args&&...) noexcept { return Ret{value}; }

  template <class ...Args>
  TEST_HOST_DEVICE constexpr Ret operator()(Args&&...) const noexcept { return Ret{value}; }

  TEST_HOST_DEVICE constexpr explicit NoExceptCallable(bool x) : value(x)  {}
  Ret value;
};

struct CopyAssignableWrapper {
  constexpr CopyAssignableWrapper(CopyAssignableWrapper const&) = default;
  constexpr CopyAssignableWrapper(CopyAssignableWrapper&&) = default;
  constexpr CopyAssignableWrapper& operator=(CopyAssignableWrapper const&) = default;
  constexpr CopyAssignableWrapper& operator=(CopyAssignableWrapper &&) = default;

  template <class ...Args>
  TEST_HOST_DEVICE constexpr bool operator()(Args&&...) { return value; }

  TEST_HOST_DEVICE constexpr explicit CopyAssignableWrapper(bool x) : value(x) {}
  bool value;
};


struct MoveAssignableWrapper {
  constexpr MoveAssignableWrapper(MoveAssignableWrapper const&) = delete;
  constexpr MoveAssignableWrapper(MoveAssignableWrapper&&) = default;
  constexpr MoveAssignableWrapper& operator=(MoveAssignableWrapper const&) = delete;
  constexpr MoveAssignableWrapper& operator=(MoveAssignableWrapper &&) = default;

  template <class ...Args>
  TEST_HOST_DEVICE constexpr bool operator()(Args&&...) { return value; }

  TEST_HOST_DEVICE constexpr explicit MoveAssignableWrapper(bool x) : value(x) {}
  bool value;
};

struct MemFunCallable {
  TEST_HOST_DEVICE constexpr explicit MemFunCallable(bool x) : value(x) {}

  TEST_HOST_DEVICE constexpr bool return_value() const { return value; }
  TEST_HOST_DEVICE constexpr bool return_value_nc() { return value; }
  bool value;
};

enum CallType : unsigned {
  CT_None,
  CT_NonConst = 1,
  CT_Const = 2,
  CT_LValue = 4,
  CT_RValue = 8
};

TEST_HOST_DEVICE inline constexpr CallType operator|(CallType LHS, CallType RHS) {
    return static_cast<CallType>(static_cast<unsigned>(LHS) | static_cast<unsigned>(RHS));
}
#if 0
struct ForwardingCallObject {
  struct State {
    CallType      last_call_type = CT_None;
    TypeID const& (*last_call_args)() = nullptr;

    template <class ...Args>
    TEST_HOST_DEVICE constexpr void set_call(CallType type) {
      assert(last_call_type == CT_None);
      assert(last_call_args == nullptr);
      last_call_type = type;
      last_call_args = &makeArgumentID<Args...>;
    }

    template <class ...Args>
    TEST_HOST_DEVICE constexpr bool check_call(CallType type) {
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

  TEST_HOST_DEVICE explicit constexpr ForwardingCallObject(State& st) : st_(&st) {}

  template <class ...Args>
  TEST_HOST_DEVICE constexpr bool operator()(Args&&...) & {
      st_->set_call<Args&&...>(CT_NonConst | CT_LValue);
      return true;
  }

  template <class ...Args>
  TEST_HOST_DEVICE constexpr bool operator()(Args&&...) const & {
      st_->set_call<Args&&...>(CT_Const | CT_LValue);
      return true;
  }

  // Don't allow the call operator to be invoked as an rvalue.
  template <class ...Args>
  TEST_HOST_DEVICE constexpr bool operator()(Args&&...) && {
      st_->set_call<Args&&...>(CT_NonConst | CT_RValue);
      return true;
  }

  template <class ...Args>
  TEST_HOST_DEVICE constexpr bool operator()(Args&&...) const && {
      st_->set_call<Args&&...>(CT_Const | CT_RValue);
      return true;
  }
};
#endif

#endif // TEST_CALLABLE_TYPES_H
