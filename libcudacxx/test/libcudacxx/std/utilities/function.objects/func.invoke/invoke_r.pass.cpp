//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/functional>

// template<class R, class F, class... Args>
// constexpr R invoke_r(F&& f, Args&&... args)
//     noexcept(is_nothrow_invocable_r_v<R, F, Args...>);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/functional>
#include <cuda/std/type_traits>
#include <cuda/std/utility> // declval

#include "test_macros.h"

template <class R, class F, class, class... Args>
struct can_invoke_r_impl : cuda::std::false_type
{};

template <class R, class F, class... Args>
struct can_invoke_r_impl<
  R,
  F,
  cuda::std::void_t<decltype(cuda::std::invoke_r<R>(cuda::std::declval<F>(), cuda::std::declval<Args>()...))>,
  Args...>
    : cuda::std::is_same<R, decltype(cuda::std::invoke_r<R>(cuda::std::declval<F>(), cuda::std::declval<Args>()...))>
{};

template <class R, class F, class... Args>
using can_invoke_r = typename can_invoke_r_impl<R, F, void, Args...>::type;

TEST_FUNC constexpr bool test()
{
  // Make sure basic functionality works (i.e. we actually call the function and return the right result).
  {
    struct F
    {
      TEST_FUNC constexpr int operator()(int i) const
      {
        return i + 3;
      }
    };
    assert(cuda::std::invoke_r<int>(F{}, 4) == 7);
  }

  // Make sure invoke_r is SFINAE-friendly
  {
    struct F
    {
      TEST_FUNC char* operator()(int) const;
    };

    static_assert(can_invoke_r<char*, F, int>::value);
    static_assert(can_invoke_r<void*, F, int>::value);

    // discard return type
    static_assert(can_invoke_r<void, F, int>::value);

    // wrong argument type
    static_assert(!can_invoke_r<char*, F, void*>::value);

    // missing argument
    static_assert(!can_invoke_r<char*, F>::value);

    // incompatible return type
    static_assert(!can_invoke_r<int*, F, int>::value);

    // discard return type, invalid argument type
    static_assert(!can_invoke_r<void, F, void*>::value);
  }

  // Make sure invoke_r has the right noexcept specification
  {
    struct F
    {
      TEST_FUNC char* operator()(int) const noexcept(true);
    };

    struct G
    {
      TEST_FUNC char* operator()(int) const noexcept(false);
    };

    struct ConversionNotNoexcept
    {
      TEST_FUNC ConversionNotNoexcept(char*) noexcept(false);
    };

    static_assert(noexcept(cuda::std::invoke_r<char*>(cuda::std::declval<F>(), 0)));

    // function call is not noexcept
    static_assert(!noexcept(cuda::std::invoke_r<char*>(cuda::std::declval<G>(), 0)));

    // function call is noexcept, conversion isn't
    static_assert(!noexcept(cuda::std::invoke_r<ConversionNotNoexcept>(cuda::std::declval<F>(), 0)));

    // function call and conversion are both not noexcept
    static_assert(!noexcept(cuda::std::invoke_r<ConversionNotNoexcept>(cuda::std::declval<G>(), 0)));
  }

  // Make sure invoke_r works with void return type
  {
    struct F
    {
      TEST_FUNC constexpr char* operator()(int) const
      {
        was_called = true;
        return nullptr;
      }

      bool& was_called;
    };

    bool was_called = false;
    cuda::std::invoke_r<void>(F{was_called}, 3);
    assert(was_called);
    static_assert(cuda::std::is_void<decltype(cuda::std::invoke_r<void>(F{was_called}, 3))>::value);
  }

// https://developercommunity.visualstudio.com/t/ICE-when-forwarding-a-function-to-invoke/10806827
#if !TEST_COMPILER(MSVC)
  // Make sure invoke_r works with const void return type
  {
    struct F
    {
      TEST_FUNC constexpr char* operator()(int) const
      {
        was_called = true;
        return nullptr;
      }

      bool& was_called;
    };

    bool was_called = false;
    cuda::std::invoke_r<const void>(F{was_called}, 3);
    assert(was_called);
    static_assert(cuda::std::is_void<decltype(cuda::std::invoke_r<const void>(F{was_called}, 3))>::value);
  }
#endif // !TEST_COMPILER(MSVC)

  // Make sure invoke_r forwards its arguments
  {
    struct NonCopyable
    {
      NonCopyable()                   = default;
      NonCopyable(NonCopyable const&) = delete;
      NonCopyable(NonCopyable&&)      = default;
    };
    // Forward argument, with void return
    {
      struct F
      {
        TEST_FUNC constexpr void operator()(NonCopyable) const
        {
          was_called = true;
        }

        bool& was_called;
      };

      bool was_called = false;
      cuda::std::invoke_r<void>(F{was_called}, NonCopyable());
      assert(was_called);
    }
    // Forward argument, with non-void return
    {
      struct F
      {
        TEST_FUNC constexpr int operator()(NonCopyable) const
        {
          was_called = true;
          return 0;
        }

        bool& was_called;
      };

      bool was_called = false;
      unused(cuda::std::invoke_r<int>(F{was_called}, NonCopyable()));
      assert(was_called);
    }
    // Forward function object, with void return
    {
      struct MoveOnlyVoidFunction
      {
        bool& was_called;
        TEST_FUNC constexpr void operator()() &&
        {
          was_called = true;
        }
      };
      bool was_called = false;
      cuda::std::invoke_r<void>(MoveOnlyVoidFunction{was_called});
      assert(was_called);
    }
    // Forward function object, with non-void return
    {
      struct MoveOnlyIntFunction
      {
        bool& was_called;
        TEST_FUNC constexpr int operator()() &&
        {
          was_called = true;
          return 0;
        }
      };
      bool was_called = false;
      unused(cuda::std::invoke_r<int>(MoveOnlyIntFunction{was_called}));
      assert(was_called);
    }
  }

  // Make sure invoke_r performs an implicit conversion of the result
  {
    struct Convertible
    {
      TEST_FUNC constexpr operator int() const
      {
        return 42;
      }
    };

    struct F
    {
      TEST_FUNC constexpr Convertible operator()() const
      {
        return Convertible{};
      }
    };
    int result = cuda::std::invoke_r<int>(F{});
    assert(result == 42);
  }

  // Note: We don't test that `std::invoke_r` works with all kinds of callable types here,
  //       since that is extensively tested in the `std::invoke` tests.
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
