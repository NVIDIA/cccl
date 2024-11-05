//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <stdexcept>

#include "errors.h"

bool try_push_context()
{
  CUcontext context = nullptr;

  check(cuCtxGetCurrent(&context));

  if (context == nullptr)
  {
    const int default_device = 0;
    check(cuDevicePrimaryCtxRetain(&context, default_device));
    check(cuCtxPushCurrent(context));

    return true;
  }

  return false;
}
