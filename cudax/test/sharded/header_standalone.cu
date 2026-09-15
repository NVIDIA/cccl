//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Header hygiene: every `__sharded` header must compile on its own.
 *
 * Compiled once per header, with `CUDAX_SHARDED_HEADER` naming the header
 * under test and nothing else included before it. A header that relies on a
 * name another header happens to introduce — a using-declaration pulled in
 * by an earlier include, say — compiles through the umbrella and fails here,
 * which is exactly the ordering dependency this test exists to prevent.
 */

#ifndef CUDAX_SHARDED_HEADER
#  error "CUDAX_SHARDED_HEADER must name the header under test"
#endif

#include CUDAX_SHARDED_HEADER

int main()
{
  return 0;
}
