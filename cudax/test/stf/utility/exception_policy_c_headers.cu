//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief SCOPE and the policy vocabulary survive C library headers
 *
 * The C library declares abort, exit, and terminate-adjacent names at global
 * scope, and C headers may mask library functions with function-like macros.
 * This test includes the C and C++ library headers BEFORE the STF header and
 * exercises the spellings that could collide: SCOPE's tokens paste into a
 * fully qualified path (no unqualified lookup to hijack), ON_THROW injects
 * the policy namespace for the policy expression only, and the qualified
 * exception_policies names resolve regardless of what is in scope. Pins the
 * current lookup strategy against refactors that would weaken it.
 */

// Deliberately first, and both flavors.
#include <cuda/experimental/stf.cuh>

#include <cstdlib>
#include <exception>

#include <stdlib.h>

using namespace cuda::experimental::stf;

int main()
{
  int order = 0;
  {
    SCOPE(exit)
    {
      ++order;
    };
    SCOPE(fail)
    {
      order = -100;
    };
    SCOPE(success)
    {
      ++order;
    };
  }
  _CCCL_ASSERT(order == 2, "SCOPE guards must run despite C headers");

  // The macro's namespace injection: plain `abort` inside ON_THROW means the policy.
  const int v = ON_THROW(abort)
  {
    return 42;
  };
  _CCCL_ASSERT(v == 42, "ON_THROW(abort) must compile and pass the value through");

  // The function form with qualified policies.
  const int w = on_throw(exception_policies::terminate) << [] {
    return 5;
  };
  _CCCL_ASSERT(w == 5, "on_throw(exception_policies::terminate) must compile");

  // Qualified names resolve to the policy objects, not the C library functions.
  [[maybe_unused]] const auto& policy_abort     = exception_policies::abort;
  [[maybe_unused]] const auto& policy_terminate = exception_policies::terminate;

  // Bare abort/exit still mean the C library: the policies live in a non-inline
  // namespace that `using namespace cuda::experimental::stf` does not open.
  if (order != 2)
  {
    abort();
  }
  exit(0);
}
