//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#include <cuda.h>

namespace cccl::detail
{
// An error that maps to a specific CUresult at the C API boundary. guard()
// turns it into the return value and records the message.
class cccl_error : public std::runtime_error
{
public:
  cccl_error(CUresult code, const std::string& message)
      : std::runtime_error(message)
      , code_(code)
  {}

  CUresult code() const noexcept
  {
    return code_;
  }

private:
  CUresult code_;
};

// Message for the most recent failure on this thread, read back through
// cccl_serialization_v2_last_error().
inline std::string& last_error_storage()
{
  thread_local std::string message;
  return message;
}

inline void set_last_error(const char* message) noexcept
{
  try
  {
    last_error_storage() = message;
  }
  catch (...)
  {
    last_error_storage().clear();
  }
}

inline void clear_last_error() noexcept
{
  last_error_storage().clear();
}

// Runs `body` at a C API boundary: clears the last-error message on entry, and
// maps whatever it throws to a CUresult, leaving the description behind as the
// last-error message. A cccl_error carries its own code; any other exception is
// CUDA_ERROR_UNKNOWN. Failures are also printed to stderr unless PrintToStderr
// is false, which suits entry points whose failures are an expected answer.
template <bool PrintToStderr = true, typename Body>
CUresult guard(const char* entry_point, Body&& body) noexcept
{
  clear_last_error();
  auto fail = [&](CUresult code, const char* message) noexcept {
    if constexpr (PrintToStderr)
    {
      std::fprintf(stderr, "\nERROR in %s(): %s\n", entry_point, message);
    }
    set_last_error(message);
    return code;
  };
  try
  {
    body();
    return CUDA_SUCCESS;
  }
  catch (const cccl_error& error)
  {
    return fail(error.code(), error.what());
  }
  catch (const std::exception& error)
  {
    return fail(CUDA_ERROR_UNKNOWN, error.what());
  }
  catch (...)
  {
    return fail(CUDA_ERROR_UNKNOWN, "unknown exception");
  }
}
} // namespace cccl::detail
