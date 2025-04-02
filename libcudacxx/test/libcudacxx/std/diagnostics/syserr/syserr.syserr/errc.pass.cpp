//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

#include <cuda/std/__system_error/errc.h>

#include <cerrno>

int main(int, char**)
{
  static_assert(static_cast<int>(cuda::std::errc::invalid_argument) == EINVAL);
  static_assert(static_cast<int>(cuda::std::errc::result_out_of_range) == ERANGE);
  static_assert(static_cast<int>(cuda::std::errc::value_too_large) == EOVERFLOW);

  return 0;
}
