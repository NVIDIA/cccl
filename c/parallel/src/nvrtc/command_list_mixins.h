//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

template <typename... Types>
struct mixin_next
{};

template <typename T>
struct mixin_next<T> : public T
{
  template <typename Context>
  constexpr mixin_next(Context& c)
      : T(c)
  {}
};

template <typename T, typename... Types>
struct mixin_next<T, Types...>
    : public T
    , public mixin_next<Types...>
{
  template <typename Context>
  constexpr mixin_next(Context& c)
      : T(c)
      , mixin_next<Types...>(c)
  {}
};

template <typename... Types>
struct mixin_all : public mixin_next<Types...>
{
  constexpr auto operator->()
  {
    return this;
  }
};

/*
Context: The context type that will be passed as a reference to the underlying node(s)
Types: The mixins are inherited into a parent type - A single mixin can have multiple functions if desired.

Usage: node_list<Context, NodeA, NodeB,...>{std::move(context)}->|NodeA::fun1|NodeB::fun2|...
*/
template <typename Context, typename... Types>
struct node_list
{
private:
  Context c;

public:
  constexpr node_list() = default;
  constexpr node_list(Context&& c_)
      : c(std::forward<Context>(c_))
  {}
  constexpr node_list(const Context& c_)
      : c(c_)
  {}

  template <typename T = void>
  constexpr mixin_all<Types...> operator->()
  {
    return mixin_all<Types...>{c};
  }
};
