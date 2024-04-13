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

#include "utils.h"

int main(int argc, char** argv)
{
  cuda::access_property ap(cuda::access_property::persisting{});
  int* array0 = new int[9];
  cuda::annotated_ptr<int, cuda::access_property> array_anno_ptr{array0, ap};
  cuda::annotated_ptr<int, cuda::access_property::shared> shared_ptr;

  array_anno_ptr = shared_ptr; //  fail to compile, as expected

  return 0;
}
