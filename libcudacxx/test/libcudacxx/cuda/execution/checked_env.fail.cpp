//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/execution>

struct query_t
{};

struct invalid_env
{
  int query(query_t)
  {
    return 42;
  }
};

int main(int, char**)
{
  // A checked environment must answer each advertised query on a const object.
  auto env = cuda::checked_env<query_t>(invalid_env{});
  (void) env;

  return 0;
}
