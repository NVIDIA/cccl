//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70

#include "group_memcpy_async.h"

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, cuda_thread_count = 4;)

  test_select_source<storage<uint16_t>>();

  return 0;
}
