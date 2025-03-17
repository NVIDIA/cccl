//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cstdlib>
#include <string>
#include <variant>

#include <cccl/c/types.h>

// For each kernel accepts a user operator that contains both iterator and user operator state
// This declaration is used as blueprint for aligned_storage, but is only *valid* in the generated NVRTC program.
struct for_each_default
{
  // Defaults:
  void* iterator; // A pointer for iterator
  void* user_op; // A pointer for user data
};

struct for_each_kernel_state
{
  std::variant<for_each_default, std::unique_ptr<char[]>> for_each_arg;
  size_t user_op_offset;

  // Get address of argument for kernel
  void* get();
};

std::string get_for_kernel(cccl_op_t user_op, cccl_iterator_t iter);

for_each_kernel_state make_for_kernel_state(cccl_op_t user_op, cccl_iterator_t iterator);
