//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// std::rethrow_if_nested needs RTTI on the host and the harness disables it by default; the RTTI-enabling
// flags are requested per host compiler (unsupported ones are filtered), and the cases that need RTTI are
// also guarded, so the test still runs where the flag does not take.
// ADDITIONAL_COMPILE_OPTIONS_HOST: -frtti, --rtti, /GR

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__host_stdlib/exception>
#include <cuda/std/cassert>

#include <nv/target>

#include "test_macros.h"

// This test checks that the nested-exception macros behave like std::throw_with_nested and
// std::rethrow_if_nested on host, and that they compile in device code. Device code is not ran, because it
// traps and CUDA is left in undefined state.

TEST_FUNC constexpr int low_value()
{
  return 7;
}

TEST_FUNC constexpr int high_value()
{
  return 42;
}

// Polymorphic on purpose: std::rethrow_if_nested looks for the nested cause with a dynamic_cast.
struct Low
{
  int value = low_value();

  // virtual so the type is polymorphic, as rethrow_if_nested requires; TEST_FUNC for NVRTC, where an unannotated
  // member is a host function
  [[nodiscard]] TEST_FUNC virtual const char* what() const noexcept
  {
    return "Low";
  }
};

struct High
{
  int value = high_value();

  // virtual so the type is polymorphic, as rethrow_if_nested requires; TEST_FUNC for NVRTC, where an unannotated
  // member is a host function
  [[nodiscard]] TEST_FUNC virtual const char* what() const noexcept
  {
    return "High";
  }
};

// Names only the user's types, so it compiles everywhere the macros do, NVRTC included (no <exception>
// there, hence no std::nested_exception to catch).
TEST_FUNC void test_macros_compile_everywhere()
{
  // a. maybe-with-nested outside a handler is caught as the type thrown
  _CCCL_TRY
  {
    _CCCL_THROW_MAYBE_WITH_NESTED(High);
  }
  _CCCL_CATCH (const High& e)
  {
    assert(e.value == high_value());
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // b. throw-with-nested inside a handler is caught as the type thrown
  _CCCL_TRY
  {
    _CCCL_TRY
    {
      _CCCL_THROW(Low);
    }
    _CCCL_CATCH ([[maybe_unused]] const Low& e)
    {
      _CCCL_THROW_WITH_NESTED(High);
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  }
  _CCCL_CATCH (const High& e)
  {
    assert(e.value == high_value());
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}

#if _CCCL_HOSTED()
// The cause is reachable through std::nested_exception itself, which needs no RTTI: the thrown object
// derives from both the user's type and std::nested_exception.
TEST_FUNC void test_nesting_without_rtti()
{
  // 1. throwing with the active exception nested, then unwinding the chain through the base class
  [[maybe_unused]] bool saw_nested = false;
  [[maybe_unused]] bool saw_low    = false;
  _CCCL_TRY
  {
    _CCCL_TRY
    {
      _CCCL_THROW(Low);
    }
    _CCCL_CATCH ([[maybe_unused]] const Low& e)
    {
      _CCCL_THROW_WITH_NESTED(High);
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  }
  _CCCL_CATCH ([[maybe_unused]] const ::std::nested_exception& ne)
  {
    NV_IF_TARGET(NV_IS_HOST, (saw_nested = true;))
    _CCCL_TRY
    {
      NV_IF_TARGET(NV_IS_HOST, (ne.rethrow_nested();)) // [[noreturn]]: a nested cause was present
    }
    _CCCL_CATCH (const Low& cause)
    {
      NV_IF_TARGET(NV_IS_HOST, (saw_low = true;))
      assert(cause.value == low_value());
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
  NV_IF_TARGET(NV_IS_HOST, (assert(saw_nested); assert(saw_low);))

  // 2. maybe-with-nested inside a handler nests too
  saw_nested = false;
  _CCCL_TRY
  {
    _CCCL_TRY
    {
      _CCCL_THROW(Low);
    }
    _CCCL_CATCH ([[maybe_unused]] const Low& e)
    {
      _CCCL_THROW_MAYBE_WITH_NESTED(High);
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  }
  _CCCL_CATCH ([[maybe_unused]] const ::std::nested_exception& ne)
  {
    NV_IF_TARGET(NV_IS_HOST, (saw_nested = true;))
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
  NV_IF_TARGET(NV_IS_HOST, (assert(saw_nested);))

  // 3. maybe-with-nested outside any handler is a plain throw: not a std::nested_exception
  _CCCL_TRY
  {
    _CCCL_THROW_MAYBE_WITH_NESTED(High);
  }
  _CCCL_CATCH ([[maybe_unused]] const ::std::nested_exception& ne)
  {
    assert(false);
  }
  _CCCL_CATCH (const High& e)
  {
    assert(e.value == high_value());
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}
#endif // _CCCL_HOSTED()

#if _CCCL_HOSTED() && !defined(_CCCL_NO_RTTI)
TEST_FUNC void test_rethrow_if_nested()
{
  // 4. rethrow-if-nested recovers the cause
  [[maybe_unused]] bool saw_high = false;
  [[maybe_unused]] bool saw_low  = false;
  _CCCL_TRY
  {
    _CCCL_TRY
    {
      _CCCL_THROW(Low);
    }
    _CCCL_CATCH ([[maybe_unused]] const Low& e)
    {
      _CCCL_THROW_WITH_NESTED(High);
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  }
  _CCCL_CATCH (const High& e)
  {
    NV_IF_TARGET(NV_IS_HOST, (saw_high = true;))
    assert(e.value == high_value());
    _CCCL_TRY
    {
      _CCCL_RETHROW_IF_NESTED(e);
      assert(false); // a nested cause was present, so this must have thrown
    }
    _CCCL_CATCH (const Low& cause)
    {
      NV_IF_TARGET(NV_IS_HOST, (saw_low = true;))
      assert(cause.value == low_value());
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
  NV_IF_TARGET(NV_IS_HOST, (assert(saw_high); assert(saw_low);))

  // 5. rethrow-if-nested on an exception with no nested cause has no effect
  _CCCL_TRY
  {
    _CCCL_THROW(High);
  }
  _CCCL_CATCH (const High& e)
  {
    _CCCL_RETHROW_IF_NESTED(e);
    assert(e.value == high_value());
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 6. rethrow-if-nested outside any handler, on a plain object: no effect
  const Low plain{};
  _CCCL_RETHROW_IF_NESTED(plain);
  assert(plain.value == low_value());

  // 7. rethrow-if-nested after maybe-with-nested outside a handler: a no-op, not the std::terminate that
  // an empty nested cause would produce
  _CCCL_TRY
  {
    _CCCL_THROW_MAYBE_WITH_NESTED(High);
  }
  _CCCL_CATCH (const High& e)
  {
    _CCCL_RETHROW_IF_NESTED(e);
    assert(e.value == high_value());
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}
#endif // _CCCL_HOSTED() && !_CCCL_NO_RTTI

TEST_FUNC void test()
{
  test_macros_compile_everywhere();
#if _CCCL_HOSTED()
  test_nesting_without_rtti();
#  if !defined(_CCCL_NO_RTTI)
  test_rethrow_if_nested();
#  endif // !_CCCL_NO_RTTI
#endif // _CCCL_HOSTED()
}

__global__ void test_kernel()
{
  // compile only on device
  test();
}

int main(int, char**)
{
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
