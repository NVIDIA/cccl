//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-70

// <mutex>

// struct once_flag;

// once_flag(const once_flag&) = delete;

#include<cuda/std/mutex>

int main(int, char**)
{
    cuda::std::once_flag f;
    cuda::std::once_flag f2(f);

  return 0;
}
