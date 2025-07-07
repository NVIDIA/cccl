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

struct ExceptionBase
{
  int value;
};

struct Exception : ExceptionBase
{};

#if !_CCCL_COMPILER(NVRTC)
void test_host()
{
  // 1. test catch by value
  _CCCL_TRY
  {
    throw Exception{};
  }
  _CCCL_CATCH (Exception e)
  {
    assert(e.value == 0);
    assert(true);
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
    throw Exception{};
  }
  _CCCL_CATCH (Exception & e)
  {
    assert(e.value == 0);
    assert(true);
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
    throw Exception{};
  }
  _CCCL_CATCH (const Exception & e)
  {
    assert(e.value == 0);
    assert(true);
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
}
#endif // !_CCCL_COMPILER(NVRTC)

__device__ void test_device()
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
}

int main(int, char**)
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (test_host();), (test_device();))
  return 0;
}
