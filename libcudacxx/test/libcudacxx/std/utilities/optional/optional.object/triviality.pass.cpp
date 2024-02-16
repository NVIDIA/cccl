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

// The following special member functions should propagate the triviality of
// the element held in the optional (see P0602R4):
//
// constexpr optional(const optional& rhs);
// constexpr optional(optional&& rhs) noexcept(see below);
// constexpr optional<T>& operator=(const optional& rhs);
// constexpr optional<T>& operator=(optional&& rhs) noexcept(see below);


#include <cuda/std/optional>
#include <cuda/std/type_traits>

#if !defined(_LIBCUDACXX_ADDRESSOF)
#define TEST_WORKAROUND_NO_ADDRESSOF
#endif

#include "archetypes.h"

#include "test_macros.h"


TEST_HOST_DEVICE
constexpr bool implies(bool p, bool q) {
    return !p || q;
}

template <class T>
struct SpecialMemberTest {
    using O = cuda::std::optional<T>;

#ifndef TEST_COMPILER_ICC
    static_assert(implies(cuda::std::is_trivially_copy_constructible_v<T>,
                          cuda::std::is_trivially_copy_constructible_v<O>),
        "optional<T> is trivially copy constructible if T is trivially copy constructible.");

    static_assert(implies(cuda::std::is_trivially_move_constructible_v<T>,
                          cuda::std::is_trivially_move_constructible_v<O>),
        "optional<T> is trivially move constructible if T is trivially move constructible");
#endif // TEST_COMPILER_ICC

    static_assert(implies(cuda::std::is_trivially_copy_constructible_v<T> &&
                          cuda::std::is_trivially_copy_assignable_v<T> &&
                          cuda::std::is_trivially_destructible_v<T>,

                          cuda::std::is_trivially_copy_assignable_v<O>),
        "optional<T> is trivially copy assignable if T is "
        "trivially copy constructible, "
        "trivially copy assignable, and "
        "trivially destructible");

    static_assert(implies(cuda::std::is_trivially_move_constructible_v<T> &&
                          cuda::std::is_trivially_move_assignable_v<T> &&
                          cuda::std::is_trivially_destructible_v<T>,

                          cuda::std::is_trivially_move_assignable_v<O>),
        "optional<T> is trivially move assignable if T is "
        "trivially move constructible, "
        "trivially move assignable, and"
        "trivially destructible.");
};

template <class ...Args> TEST_HOST_DEVICE static void sink(Args&&...) {}

template <class ...TestTypes>
struct DoTestsMetafunction {
    TEST_HOST_DEVICE
    DoTestsMetafunction() { sink(SpecialMemberTest<TestTypes>{}...); }
};

struct TrivialMoveNonTrivialCopy {
    TrivialMoveNonTrivialCopy() = default;
    TEST_HOST_DEVICE
    TrivialMoveNonTrivialCopy(const TrivialMoveNonTrivialCopy&) {}
    TrivialMoveNonTrivialCopy(TrivialMoveNonTrivialCopy&&) = default;
    TEST_HOST_DEVICE
    TrivialMoveNonTrivialCopy& operator=(const TrivialMoveNonTrivialCopy&) { return *this; }
    TrivialMoveNonTrivialCopy& operator=(TrivialMoveNonTrivialCopy&&) = default;
};

struct TrivialCopyNonTrivialMove {
    TrivialCopyNonTrivialMove() = default;
    TrivialCopyNonTrivialMove(const TrivialCopyNonTrivialMove&) = default;
    TEST_HOST_DEVICE
    TrivialCopyNonTrivialMove(TrivialCopyNonTrivialMove&&) {}
    TrivialCopyNonTrivialMove& operator=(const TrivialCopyNonTrivialMove&) = default;
    TEST_HOST_DEVICE
    TrivialCopyNonTrivialMove& operator=(TrivialCopyNonTrivialMove&&) { return *this; }
};

int main(int, char**) {
    sink(
        ImplicitTypes::ApplyTypes<DoTestsMetafunction>{},
        ExplicitTypes::ApplyTypes<DoTestsMetafunction>{},
        NonLiteralTypes::ApplyTypes<DoTestsMetafunction>{},
        NonTrivialTypes::ApplyTypes<DoTestsMetafunction>{},
        DoTestsMetafunction<TrivialMoveNonTrivialCopy, TrivialCopyNonTrivialMove>{}
    );
    return 0;
}
