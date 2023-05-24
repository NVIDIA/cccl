//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// constexpr optional() noexcept;

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "archetypes.h"

using cuda::std::optional;

template <class Opt>
__host__ __device__
void
test_constexpr()
{
    static_assert(cuda::std::is_nothrow_default_constructible<Opt>::value, "");
    static_assert(cuda::std::is_trivially_destructible<Opt>::value, "");
    static_assert(cuda::std::is_trivially_destructible<typename Opt::value_type>::value, "");

    constexpr Opt opt;
    static_assert(static_cast<bool>(opt) == false, "");

    struct test_constexpr_ctor
        : public Opt
    {
        __host__ __device__
        constexpr test_constexpr_ctor() {}
    };
}

template <class Opt>
__host__ __device__
void
test()
{
    static_assert(cuda::std::is_nothrow_default_constructible<Opt>::value, "");
    static_assert(!cuda::std::is_trivially_destructible<Opt>::value, "");
    static_assert(!cuda::std::is_trivially_destructible<typename Opt::value_type>::value, "");
    {
        Opt opt;
        assert(static_cast<bool>(opt) == false);
    }
    {
        const Opt opt;
        assert(static_cast<bool>(opt) == false);
    }

    struct test_constexpr_ctor
        : public Opt
    {
        __host__ __device__
        constexpr test_constexpr_ctor() {}
    };
}

int main(int, char**)
{
    test_constexpr<optional<int>>();
    test_constexpr<optional<int*>>();
#if !defined(TEST_COMPILER_C1XX) || TEST_STD_VER >= 17
    test_constexpr<optional<ImplicitTypes::NoCtors>>();
    test_constexpr<optional<NonTrivialTypes::NoCtors>>();
    test_constexpr<optional<NonConstexprTypes::NoCtors>>();
#endif
    test<optional<NonLiteralTypes::NoCtors>>();
    // EXTENSIONS
#if defined(_LIBCPP_VERSION) && 0 // FIXME these extensions are currently disabled.
    test_constexpr<optional<int&>>();
    test_constexpr<optional<const int&>>();
    test_constexpr<optional<int&>>();
    test_constexpr<optional<NonLiteralTypes::NoCtors&>>();
    test_constexpr<optional<NonLiteralTypes::NoCtors&&>>();
#endif

  return 0;
}
