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

template <class... T>
using box = cuda::std::__compressed_movable_box<T...>;

__host__ __device__ constexpr bool test()
{
  { // single element
    { // trivial nonempty type
      box<int> b{};
      assert(b.__get<0>() == 0);
      static_assert(noexcept(box<int>{}));
    }

    { // non-trivial empty type
      [[maybe_unused]] box<NotTriviallyDefaultConstructibleEmpty<42>> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyDefaultConstructibleEmpty<42>>{}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyDefaultConstructible<42>> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyDefaultConstructible<42>>{}));
    }

    { // non-trivial empty type, not noexcept
      [[maybe_unused]] box<NotTriviallyDefaultConstructibleEmpty<MayThrow>> b{};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyDefaultConstructibleEmpty<MayThrow>>{}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyDefaultConstructible<MayThrow>> b{};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyDefaultConstructible<MayThrow>>{}));
    }

    { // not default constructible
      static_assert(!cuda::std::is_default_constructible_v<NotDefaultConstructible>);
      static_assert(!cuda::std::is_default_constructible_v<box<NotDefaultConstructible>>);
    }

    { // Not copyable
      box<NotCopyConstructible<42>> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotCopyConstructible<42>>{}));
    }

    { // Not copy-assignable
      box<NotCopyAssignable<42>, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotCopyAssignable<42>, int>{}));
    }

    { // Not move-assignable
      box<NotMoveAssignable<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotMoveAssignable<42>, int>{}));
    }
  }

  { // two elements
    { // trivial nonempty type
      box<int, int> b{};
      assert(b.__get<0>() == 0);
      static_assert(noexcept(box<int, int>{}));
    }

    { // non-trivial empty type
      [[maybe_unused]] box<NotTriviallyDefaultConstructibleEmpty<42>, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyDefaultConstructibleEmpty<42>, int>{}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyDefaultConstructible<42>, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyDefaultConstructible<42>, int>{}));
    }

    { // non-trivial empty type, not noexcept
      [[maybe_unused]] box<NotTriviallyDefaultConstructibleEmpty<MayThrow>, int> b{};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyDefaultConstructibleEmpty<MayThrow>, int>{}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyDefaultConstructible<MayThrow>, int> b{};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyDefaultConstructible<MayThrow>, int>{}));
    }

    { // not default constructible
      static_assert(!cuda::std::is_default_constructible_v<NotDefaultConstructible>);
      static_assert(!cuda::std::is_default_constructible_v<box<NotDefaultConstructible, int>>);
    }

    { // Not copyable
      box<NotCopyConstructible<42>, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotCopyConstructible<42>, int>{}));
    }

    { // Not copy-assignable
      box<NotCopyAssignable<42>, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotCopyAssignable<42>, int>{}));
    }

    { // Not move-assignable
      box<NotMoveAssignable<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotMoveAssignable<42>, int>{}));
    }
  }

  { // three elements
    { // trivial nonempty type
      box<int, int, int> b{};
      assert(b.__get<0>() == 0);
      static_assert(noexcept(box<int, int, int>{}));
    }

    { // non-trivial empty type
      [[maybe_unused]] box<NotTriviallyDefaultConstructibleEmpty<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyDefaultConstructibleEmpty<42>, int, int>{}));
    }

    { // non-trivial nonempty type
      box<NotTriviallyDefaultConstructible<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotTriviallyDefaultConstructible<42>, int, int>{}));
    }

    { // non-trivial empty type, not noexcept
      [[maybe_unused]] box<NotTriviallyDefaultConstructibleEmpty<MayThrow>, int, int> b{};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyDefaultConstructibleEmpty<MayThrow>, int, int>{}));
    }

    { // non-trivial nonempty type, not noexcept
      box<NotTriviallyDefaultConstructible<MayThrow>, int, int> b{};
      assert(b.__get<0>() == MayThrow);
      static_assert(!noexcept(box<NotTriviallyDefaultConstructible<MayThrow>, int, int>{}));
    }

    { // not default constructible
      static_assert(!cuda::std::is_default_constructible_v<NotDefaultConstructible>);
      static_assert(!cuda::std::is_default_constructible_v<box<NotDefaultConstructible, int, int>>);
    }

    { // Not copyable
      box<NotCopyConstructible<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotCopyConstructible<42>, int, int>{}));
    }

    { // Not copy-assignable
      box<NotCopyAssignable<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotCopyAssignable<42>, int, int>{}));
    }

    { // Not move-assignable
      box<NotMoveAssignable<42>, int, int> b{};
      assert(b.__get<0>() == 42);
      static_assert(noexcept(box<NotMoveAssignable<42>, int, int>{}));
    }
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
