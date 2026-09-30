//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// std::rethrow_if_nested needs RTTI on the host; the harness disables it by default. Unsupported flags are
// filtered per compiler.
// ADDITIONAL_COMPILE_OPTIONS_HOST: -frtti --rtti /GR

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/cassert>

#include <nv/target>

#include "test_macros.h"

// This test checks that _CCCL_THROW_WITH_NESTED and _CCCL_RETHROW_IF_NESTED behave like
// std::throw_with_nested and std::rethrow_if_nested on host, and that they compile in device code.
// Device code is not ran, because it traps and CUDA is left in undefined state.

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

  TEST_FUNC virtual ~Low() = default;

  [[nodiscard]] TEST_FUNC static const char* what() noexcept
  {
    return "Low";
  }
};

struct High
{
  int value = high_value();

  TEST_FUNC virtual ~High() = default;

  [[nodiscard]] TEST_FUNC static const char* what() noexcept
  {
    return "High";
  }
};

TEST_FUNC void test()
{
  // 1. throwing with the active exception nested, then unwinding the chain
  bool saw_high = false;
  bool saw_low  = false;
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
    saw_high = true;
    assert(e.value == high_value());
    _CCCL_TRY
    {
      _CCCL_RETHROW_IF_NESTED(e);
      assert(false); // a nested cause was present, so this must have thrown
    }
    _CCCL_CATCH (const Low& cause)
    {
      saw_low = true;
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

  // 2. rethrow-if-nested on an exception with no nested cause has no effect
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

  // 3. rethrow-if-nested outside any handler, on a plain object: no effect
  const Low plain{};
  _CCCL_RETHROW_IF_NESTED(plain);
  assert(plain.value == low_value());

  // 4. maybe-with-nested inside a handler: the active exception is nested
  saw_high = false;
  saw_low  = false;
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
  _CCCL_CATCH (const High& e)
  {
    saw_high = true;
    _CCCL_TRY
    {
      _CCCL_RETHROW_IF_NESTED(e);
      assert(false);
    }
    _CCCL_CATCH ([[maybe_unused]] const Low& cause)
    {
      saw_low = true;
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

  // 5. maybe-with-nested outside any handler: a plain throw, and rethrow-if-nested on the result is a
  // no-op rather than the std::terminate a null nested cause would produce
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
