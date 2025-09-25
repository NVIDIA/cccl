//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// template <class _Iterator>
// struct __bounded_iter;
//
// Arithmetic operators

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "types.h"

template <class... Ts>
using box = cuda::std::__compressed_movable_box<Ts...>;

template <class... Ts>
__host__ __device__ TEST_CONSTEXPR_CXX20 void test(const int expected)
{
  box<Ts...> input{1337};
  box<Ts...> b{42};
  b = cuda::std::move(input);
  assert(b.template __get<0>() == expected);
  constexpr bool is_noexcept =
    (cuda::std::movable<Ts> && ...)
      ? ((cuda::std::is_nothrow_move_constructible_v<Ts> && cuda::std::is_nothrow_move_assignable_v<Ts>) && ...)
      : (cuda::std::is_nothrow_move_constructible_v<Ts> && ...);
  static_assert(noexcept(b = cuda::std::move(input)) == is_noexcept);
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  { // single element
    { // trivial empty type
      box<TrivialEmpty> input{};
      box<TrivialEmpty> b{};
      b = cuda::std::move(input);
      assert(b.__get<0>() == 42);
      static_assert(noexcept(b = cuda::std::move(input)));
    }

    { // trivial nonempty type
      test<int>(1337);
    }

    { // non-trivial empty type
      test<NotTriviallyMoveAssignableEmpty<13>>(13);
    }

    { // non-trivial nonempty type
      test<NotTriviallyMoveAssignable<42>>(1337);
    }

    { // non-trivial empty type, not noexcept
      test<NotTriviallyMoveAssignableEmpty<MayThrow>>(MayThrow);
    }

    { // non-trivial nonempty type, not noexcept
      test<NotTriviallyMoveAssignable<MayThrow>>(1337);
    }

    { // nonempty not copy assignable
      test<NotMoveAssignable<42>>(1337);
    }

    { // not copy assignable,
      test<NotMoveAssignableEmpty<13>>(13);
    }

    { // nonempty not copy assignable
      test<NotMoveAssignable<MayThrow>>(1337);
    }

    { // empty not copy assignable,
      test<NotMoveAssignableEmpty<MayThrow>>(MayThrow);
    }
  }

  { // two elements
    { // trivial empty type
      box<TrivialEmpty, int> input{};
      box<TrivialEmpty, int> b{};
      b = cuda::std::move(input);
      assert(b.__get<0>() == 42);
      static_assert(noexcept(b = cuda::std::move(input)));
    }

    { // trivial nonempty type
      test<int, int>(1337);
    }

    { // non-trivial empty type
      test<NotTriviallyMoveAssignableEmpty<13>, int>(13);
    }

    { // non-trivial nonempty type
      test<NotTriviallyMoveAssignable<42>, int>(1337);
    }

    { // non-trivial empty type, not noexcept
      test<NotTriviallyMoveAssignableEmpty<MayThrow>, int>(MayThrow);
    }

    { // non-trivial nonempty type, not noexcept
      test<NotTriviallyMoveAssignable<MayThrow>, int>(1337);
    }

    { // nonempty not copy assignable
      test<NotMoveAssignable<42>, int>(1337);
    }

    { // not copy assignable,
      test<NotMoveAssignableEmpty<13>, int>(13);
    }

    { // nonempty not copy assignable
      test<NotMoveAssignable<MayThrow>, int>(1337);
    }

    { // empty not copy assignable,
      test<NotMoveAssignableEmpty<MayThrow>, int>(MayThrow);
    }
  }

  { // three elements
    { // trivial empty type
      box<TrivialEmpty, int> input{};
      box<TrivialEmpty, int> b{};
      b = cuda::std::move(input);
      assert(b.__get<0>() == 42);
      static_assert(noexcept(b = cuda::std::move(input)));
    }

    { // trivial nonempty type
      test<int, int, int>(1337);
    }

    { // non-trivial empty type
      test<NotTriviallyMoveAssignableEmpty<13>, int, int>(13);
    }

    { // non-trivial nonempty type
      test<NotTriviallyMoveAssignable<42>, int, int>(1337);
    }

    { // non-trivial empty type, not noexcept
      test<NotTriviallyMoveAssignableEmpty<MayThrow>, int, int>(MayThrow);
    }

    { // non-trivial nonempty type, not noexcept
      test<NotTriviallyMoveAssignable<MayThrow>, int, int>(1337);
    }

    { // nonempty not copy assignable
      test<NotMoveAssignable<42>, int, int>(1337);
    }

    { // not copy assignable,
      test<NotMoveAssignableEmpty<13>, int, int>(13);
    }

    { // nonempty not copy assignable
      test<NotMoveAssignable<MayThrow>, int, int>(1337);
    }

    { // empty not copy assignable,
      test<NotMoveAssignableEmpty<MayThrow>, int, int>(MayThrow);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test());
#endif // TEST_STD_VER >= 2020

  return 0;
}
