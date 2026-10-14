// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <new>

namespace c2h
{
// Call only while handling an exception. Recognized allocation failures return;
// all other exceptions propagate to the caller.
inline void rethrow_unless_allocation_failure()
{
  try
  {
    throw;
  }
  catch (const std::bad_alloc&)
  {}
  catch (const THRUST_NS_QUALIFIER::system_error& e)
  {
    if (e.code()
        != THRUST_NS_QUALIFIER::system::error_code{cudaErrorMemoryAllocation, THRUST_NS_QUALIFIER::cuda_category()})
    {
      throw;
    }
  }
}
} // namespace c2h
