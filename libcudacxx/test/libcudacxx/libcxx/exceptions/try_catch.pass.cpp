//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/cassert>

#include <nv/target>

// This test checks whether catching/throwing exceptions works correctly on host and whether it compiles in device code.
// Device code is not ran, because it traps and CUDA is left in undefined state.

__host__ __device__ constexpr int exception_value()
{
  return 42;
}

struct ExceptionBase
{
  int value = exception_value();
};

struct Exception : ExceptionBase
{};

// forward declaration
struct HostException;

#if !_CCCL_COMPILER(NVRTC)
struct HostException
{
  int value;

  HostException()
      : value(exception_value())
  {}
};
#endif // !_CCCL_COMPILER(NVRTC)

__host__ __device__ void test_rethrow_helper()
{
  _CCCL_TRY
  {
    _CCCL_THROW(Exception());
  }
  _CCCL_CATCH (Exception e)
  {
    assert(e.value == exception_value());
    _CCCL_RETHROW;
  }
  _CCCL_CATCH (ExceptionBase e)
  {
    assert(e.value == exception_value());
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }
}

__host__ __device__ void test()
{
  // 1. test catch by value
  _CCCL_TRY
  {
    _CCCL_THROW(Exception());
  }
  _CCCL_CATCH (Exception e)
  {
    assert(e.value == exception_value());
    assert(true);
  }
  _CCCL_CATCH (ExceptionBase e)
  {
    assert(e.value == exception_value());
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 2. test catch by lvalue reference
  _CCCL_TRY
  {
    _CCCL_THROW(Exception());
  }
  _CCCL_CATCH (Exception & e)
  {
    assert(e.value == exception_value());
    assert(true);
  }
  _CCCL_CATCH (ExceptionBase & e)
  {
    assert(e.value == exception_value());
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 3. test catch by const lvalue reference
  _CCCL_TRY
  {
    _CCCL_THROW(Exception());
  }
  _CCCL_CATCH (const Exception& e)
  {
    assert(e.value == exception_value());
    assert(true);
  }
  _CCCL_CATCH (const ExceptionBase& e)
  {
    assert(e.value == exception_value());
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 4. test rethrow
  _CCCL_TRY
  {
    test_rethrow_helper();
  }
  _CCCL_CATCH (const Exception& e)
  {
    assert(e.value == exception_value());
    assert(true);
  }
  _CCCL_CATCH (const ExceptionBase& e)
  {
    assert(e.value == exception_value());
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 5. test pathological case (try/catch inside an if without braces)
  // clang-format off
  if (true)
    _CCCL_TRY
    {
      assert(true);
    }
    _CCCL_CATCH (const Exception& e)
    {
      assert(e.value == 0);
      assert(false);
    }
    _CCCL_CATCH (const ExceptionBase& e)
    {
      assert(e.value == 0);
      assert(false);
    }
    _CCCL_CATCH_ALL
    {
      assert(false);
    }
  else
    assert(false);
  // clang-format on

  // 6. test throwing host-only exceptions
  _CCCL_TRY
  {
    _CCCL_THROW(HostException());
  }
  _CCCL_CATCH (const HostException& e)
  {
    assert(e.value == exception_value());
    assert(true);
  }
  _CCCL_CATCH_FALLTHROUGH
}

__global__ void test_kernel()
{
  test();
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
