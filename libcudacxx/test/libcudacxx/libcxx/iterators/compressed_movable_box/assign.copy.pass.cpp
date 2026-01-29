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

template <class T>
__host__ __device__ TEST_CONSTEXPR_CXX20 void test(const int expected)
{
  constexpr bool is_noexcept =
    cuda::std::copyable<T>
      ? (cuda::std::is_nothrow_copy_constructible_v<T> && cuda::std::is_nothrow_copy_assignable_v<T>)
      : cuda::std::is_nothrow_copy_constructible_v<T>;
  { // single item
    const box<T> input{1337};
    box<T> b{42};
    b = input;
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_assignable_v<box<T>> == is_noexcept);
  }
  { // two items
    const box<T, int> input{1337};
    box<T, int> b{42};
    b = input;
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_assignable_v<box<T, int>> == is_noexcept);
  }
  { // three items
    const box<T, int, int> input{1337};
    box<T, int, int> b{42};
    b = input;
    assert(b.template __get<0>() == expected);
    static_assert(cuda::std::is_nothrow_copy_assignable_v<box<T, int, int>> == is_noexcept);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  using cuda::std::__compressed_box_choose;
  using cuda::std::__compressed_box_copy_assign_available;
  using cuda::std::__compressed_box_specialization;
  using cuda::std::__smf_availability;

  { // trivial empty type
    using T = TrivialEmpty;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__trivial);

    const box<T> input{};
    box<T> b{};
    b = input;
    assert(b.__get<0>() == 42);
    static_assert(noexcept(b = input));
  }

  { // trivial nonempty type
    using T = int;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__trivial);
    test<T>(1337);
  }

  { // non-trivial empty type
    using T = NotTriviallyCopyAssignableEmpty<13>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(13);
  }

  { // non-trivial nonempty type
    using T = NotTriviallyCopyAssignable<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // non-trivial empty type, not noexcept
    using T = NotTriviallyCopyAssignableEmpty<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(MayThrow);
  }

  { // non-trivial nonempty type, not noexcept
    using T = NotTriviallyCopyAssignable<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // nonempty not copy assignable
    using T = NotCopyAssignable<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // nonempty not copy assignable
    using T = NotCopyAssignableNotDefaultConstructible<42>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // not copy assignable
    using T = NotCopyAssignableEmpty<13>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(13);
  }

  { // not copy assignable
    using T = NotCopyAssignableNotDefaultConstructibleEmpty<13>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(13);
  }

  { // nonempty not copy assignable
    using T = NotCopyAssignable<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // nonempty not copy assignable
    using T = NotCopyAssignableNotDefaultConstructible<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(1337);
  }

  { // empty not copy assignable
    using T = NotCopyAssignableEmpty<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(MayThrow);
  }

  { // empty not copy assignable
    using T = NotCopyAssignableNotDefaultConstructibleEmpty<MayThrow>;
    static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__with_engaged);
    static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__available);
    test<T>(MayThrow);
  }

  { // neither copy constructible nor copy assignable
    {
      using T = NotCopyConstructibleOrAssignable;
      static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__store_inline);
      static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__deleted);
      static_assert(!cuda::std::is_copy_assignable_v<box<T>>);
    }

    {
      using T = NotCopyConstructibleOrAssignableEmpty;
      static_assert(__compressed_box_choose<T>() == __compressed_box_specialization::__empty_non_final);
      static_assert(__compressed_box_copy_assign_available<T> == __smf_availability::__deleted);
      static_assert(!cuda::std::is_copy_assignable_v<box<T>>);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
#  if !TEST_COMPILER(GCC, >=, 14) && !TEST_COMPILER(MSVC)
  // GCC:  error: destroying ‘b’ outside its lifetime (on the })
  // MSVC: error: read of an uninitialized symbol
  static_assert(test());
#  endif // !TEST_COMPILER(GCC, >=, 14) && !TEST_COMPILER(MSVC)
#endif // TEST_STD_VER >= 2020

  return 0;
}
