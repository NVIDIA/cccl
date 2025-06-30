//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#define CCCL_DISABLE_EXCEPTIONS

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/cassert>

#include <nv/target>

struct ExceptionBase
{
  int value;
};

struct Exception : ExceptionBase
{};

__host__ __device__ void test()
{
  // 1. test catch by value
  _CCCL_TRY
  {
    assert(true);
  }
  _CCCL_CATCH (Exception e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH (ExceptionBase e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 2. test catch by lvalue reference
  _CCCL_TRY
  {
    assert(true);
  }
  _CCCL_CATCH (Exception & e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH (ExceptionBase & e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 3. test catch by const lvalue reference
  _CCCL_TRY
  {
    assert(true);
  }
  _CCCL_CATCH (const Exception & e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH (const ExceptionBase & e)
  {
    assert(e.value == 0);
    assert(false);
  }
  _CCCL_CATCH_ALL
  {
    assert(false);
  }

  // 4. test pathological case (try/catch inside an if without braces)

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
}

int main(int, char**)
{
  test();
  return 0;
}
