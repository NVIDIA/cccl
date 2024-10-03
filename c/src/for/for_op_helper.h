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

#include <memory>
#include <string>
#include <variant>

#include <cccl/c/types.h>
#include <util/small_storage.h>

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
  std::variant<for_each_default, std::unique_ptr<void>> for_each_arg;
  size_t user_op_offset;

  void* get();
};

std::string get_for_kernel_op_declaration(cccl_op_t user_op, std::string state_identifier);
std::string get_for_kernel_op_invocation(cccl_op_t user_op, std::string state_var);

for_each_kernel_state make_for_kernel_state(cccl_op_t user_op, cccl_iterator_t iterator);

struct cccl_for_op_helper
{
  std::string op_decl;
  std::string op_invoke;
  std::string op_state_var;
  std::string for_each_kernel;

  cccl_for_op_helper(cccl_op_t user_op,
                     cccl_iterator_t iter,
                     std::string state_decl = "state_storage_t",
                     std::string state_var  = "state")
      : op_decl(get_for_kernel_op_declaration(user_op, state_decl))
      , op_invoke(get_for_kernel_op_invocation(user_op, state_var))
      , op_state_var()
      , for_each_kernel()
  {}
};

/*
template <typename IterT, typename UserOpT>
struct for_each_state {
  IterT iter_state
  UserOpT user_op
}

void for_each_state::operator()(idx) {
  auto iter_proxy = this->iter_state(idx)
  user_op(iter_proxy);
}

kernel(num_items, for_each_state)
*/
