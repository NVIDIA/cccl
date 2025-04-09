//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: libcpp-has-no-threads

// <cuda/ptx>

#include <cuda/ptx>
#include <cuda/std/utility>

#include "generated/fence.h"
#include "generated/fence_mbarrier_init.h"
#include "generated/fence_proxy_alias.h"
#include "generated/fence_proxy_async.h"
#include "generated/fence_proxy_async_generic_sync_restrict.h"
#include "generated/fence_proxy_tensormap_generic.h"
#include "generated/fence_sync_restrict.h"

int main(int, char**)
{
  return 0;
}
