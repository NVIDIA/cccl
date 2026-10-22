//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// __referenceable<Tp>
//
// [defns.referenceable] defines "a referenceable type" as:
// An object type, a function type that does not have cv-qualifiers
//    or a ref-qualifier, or a reference type.
//

#include <cuda/std/concepts>

#include "test_macros.h"

struct Foo
{};

static_assert(!cuda::std::__referenceable<void>);
static_assert(cuda::std::__referenceable<int>);
static_assert(cuda::std::__referenceable<int[3]>);
static_assert(cuda::std::__referenceable<int[]>);
static_assert(cuda::std::__referenceable<int&>);
static_assert(cuda::std::__referenceable<const int&>);
static_assert(cuda::std::__referenceable<int*>);
static_assert(cuda::std::__referenceable<const int*>);
static_assert(cuda::std::__referenceable<Foo>);
static_assert(cuda::std::__referenceable<const Foo>);
static_assert(cuda::std::__referenceable<Foo&>);
static_assert(cuda::std::__referenceable<const Foo&>);

// Functions without cv-qualifiers are referenceable
static_assert(cuda::std::__referenceable<void()>);

static_assert(cuda::std::__referenceable<void(int)>);

static_assert(cuda::std::__referenceable<void(int, float)>);

static_assert(cuda::std::__referenceable<void(int, float, Foo&)>);

static_assert(cuda::std::__referenceable<void(...)>);

static_assert(cuda::std::__referenceable<void(int, ...)>);

static_assert(cuda::std::__referenceable<void(int, float, ...)>);

static_assert(cuda::std::__referenceable<void(int, float, Foo&, ...)>);

// member functions with or without cv-qualifiers are referenceable
static_assert(cuda::std::__referenceable<void (Foo::*)()>);
static_assert(cuda::std::__referenceable<void (Foo::*)() const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(int)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(int) const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(int, float)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(int, float) const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(int, float, Foo&)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(int, float, Foo&) const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(...)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(...) const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(int, ...)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(int, ...) const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(int, float, ...)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(int, float, ...) const>);

static_assert(cuda::std::__referenceable<void (Foo::*)(int, float, Foo&, ...)>);
static_assert(cuda::std::__referenceable<void (Foo::*)(int, float, Foo&, ...) const>);

int main(int, char**)
{
  return 0;
}
