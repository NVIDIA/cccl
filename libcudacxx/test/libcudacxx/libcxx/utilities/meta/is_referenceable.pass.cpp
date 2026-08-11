//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// __cccl_is_referenceable<Tp>
//
// [defns.referenceable] defines "a referenceable type" as:
// An object type, a function type that does not have cv-qualifiers
//    or a ref-qualifier, or a reference type.
//

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Foo
{};

static_assert(!cuda::std::__can_reference<void>);
static_assert(cuda::std::__can_reference<int>);
static_assert(cuda::std::__can_reference<int[3]>);
static_assert(cuda::std::__can_reference<int[]>);
static_assert(cuda::std::__can_reference<int&>);
static_assert(cuda::std::__can_reference<const int&>);
static_assert(cuda::std::__can_reference<int*>);
static_assert(cuda::std::__can_reference<const int*>);
static_assert(cuda::std::__can_reference<Foo>);
static_assert(cuda::std::__can_reference<const Foo>);
static_assert(cuda::std::__can_reference<Foo&>);
static_assert(cuda::std::__can_reference<const Foo&>);
static_assert(cuda::std::__can_reference<Foo&&>);
static_assert(cuda::std::__can_reference<const Foo&&>);

#if !TEST_COMPILER(MSVC)
static_assert(cuda::std::__can_reference<int __attribute__((__vector_size__(8)))>);
static_assert(cuda::std::__can_reference<const int __attribute__((__vector_size__(8)))>);
static_assert(cuda::std::__can_reference<float __attribute__((__vector_size__(16)))>);
static_assert(cuda::std::__can_reference<const float __attribute__((__vector_size__(16)))>);
#endif // !TEST_COMPILER(MSVC)

// Functions without cv-qualifiers are referenceable
static_assert(cuda::std::__can_reference<void()>);
static_assert(!cuda::std::__can_reference<void() const>);
static_assert(!cuda::std::__can_reference<void() &>);
static_assert(!cuda::std::__can_reference<void() const&>);
static_assert(!cuda::std::__can_reference<void() &&>);
static_assert(!cuda::std::__can_reference<void() const&&>);

static_assert(cuda::std::__can_reference<void(int)>);
static_assert(!cuda::std::__can_reference<void(int) const>);
static_assert(!cuda::std::__can_reference<void(int) &>);
static_assert(!cuda::std::__can_reference<void(int) const&>);
static_assert(!cuda::std::__can_reference<void(int) &&>);
static_assert(!cuda::std::__can_reference<void(int) const&&>);

static_assert(cuda::std::__can_reference<void(int, float)>);
static_assert(!cuda::std::__can_reference<void(int, float) const>);
static_assert(!cuda::std::__can_reference<void(int, float) &>);
static_assert(!cuda::std::__can_reference<void(int, float) const&>);
static_assert(!cuda::std::__can_reference<void(int, float) &&>);
static_assert(!cuda::std::__can_reference<void(int, float) const&&>);

static_assert(cuda::std::__can_reference<void(int, float, Foo&)>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&) const>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&) &>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&) const&>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&) &&>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&) const&&>);

static_assert(cuda::std::__can_reference<void(...)>);
static_assert(!cuda::std::__can_reference<void(...) const>);
static_assert(!cuda::std::__can_reference<void(...) &>);
static_assert(!cuda::std::__can_reference<void(...) const&>);
static_assert(!cuda::std::__can_reference<void(...) &&>);
static_assert(!cuda::std::__can_reference<void(...) const&&>);

static_assert(cuda::std::__can_reference<void(int, ...)>);
static_assert(!cuda::std::__can_reference<void(int, ...) const>);
static_assert(!cuda::std::__can_reference<void(int, ...) &>);
static_assert(!cuda::std::__can_reference<void(int, ...) const&>);
static_assert(!cuda::std::__can_reference<void(int, ...) &&>);
static_assert(!cuda::std::__can_reference<void(int, ...) const&&>);

static_assert(cuda::std::__can_reference<void(int, float, ...)>);
static_assert(!cuda::std::__can_reference<void(int, float, ...) const>);
static_assert(!cuda::std::__can_reference<void(int, float, ...) &>);
static_assert(!cuda::std::__can_reference<void(int, float, ...) const&>);
static_assert(!cuda::std::__can_reference<void(int, float, ...) &&>);
static_assert(!cuda::std::__can_reference<void(int, float, ...) const&&>);

static_assert(cuda::std::__can_reference<void(int, float, Foo&, ...)>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&, ...) const>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&, ...) &>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&, ...) const&>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&, ...) &&>);
static_assert(!cuda::std::__can_reference<void(int, float, Foo&, ...) const&&>);

// member functions with or without cv-qualifiers are referenceable
static_assert(cuda::std::__can_reference<void (Foo::*)()>);
static_assert(cuda::std::__can_reference<void (Foo::*)() const>);
static_assert(cuda::std::__can_reference<void (Foo::*)() &>);
static_assert(cuda::std::__can_reference<void (Foo::*)() const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)() &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)() const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(int)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int) const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(int, float)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float) const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&) const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(...)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(...) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(...) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(...) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(...) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(...) const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(int, ...)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, ...) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, ...) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, ...) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, ...) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, ...) const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, ...)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, ...) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, ...) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, ...) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, ...) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, ...) const&&>);

static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&, ...)>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&, ...) const>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&, ...) &>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&, ...) const&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&, ...) &&>);
static_assert(cuda::std::__can_reference<void (Foo::*)(int, float, Foo&, ...) const&&>);

int main(int, char**)
{
  return 0;
}
