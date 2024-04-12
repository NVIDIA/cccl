//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/chrono>

// system_clock

// time_t to_time_t(const time_point& t);

#include <cuda/std/chrono>

#include "test_macros.h"
#include <nv/target>

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST,
               (typedef ::std::chrono::system_clock C; cuda::std::time_t t1 = C::to_time_t(C::now()); unused(t1);));
  return 0;
}
