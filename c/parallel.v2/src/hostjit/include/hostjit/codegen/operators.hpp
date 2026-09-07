//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <string>

#include <cccl/c/types.h>

namespace hostjit::codegen
{
// Result of generating operator code.
struct OperatorCode
{
  std::string preamble; // extern decl + functor struct (goes at file scope)
  std::string setup_code; // initialization inside function body
  std::string local_var; // e.g., "op_0"
};

// Generate code for a binary operator (reduce, scan).
// Produces an extern "C" device function declaration (or inline for well-known ops)
// and a functor struct that wraps it.
OperatorCode make_binary_op(
  cccl_op_t op,
  const std::string& accum_type, // C++ type name for operands
  const std::string& functor_name, // e.g., "ReduceOp"
  const std::string& var_name, // e.g., "op_0"
  const std::string& state_param, // e.g., "op_0_state" (void* param name)
  bool has_bitcode);

// Generate code for a unary operator (transform).
// Produces a functor with operator()(const in_type& a) const -> out_type.
OperatorCode make_unary_op(
  cccl_op_t op,
  const std::string& in_type, // C++ type name for input operand
  const std::string& out_type, // C++ type name for result
  const std::string& functor_name, // e.g., "UnaryOp"
  const std::string& var_name, // e.g., "op_0"
  const std::string& state_param, // e.g., "op_0_state" (void* param name)
  bool has_bitcode);

// Generate code for a comparison operator (sort).
// Same as binary op but the functor returns bool.
OperatorCode make_comparison_op(
  cccl_op_t op,
  const std::string& key_type, // C++ type name for keys
  const std::string& functor_name, // e.g., "CompareOp"
  const std::string& var_name, // e.g., "cmp_0"
  const std::string& state_param, // e.g., "cmp_0_state"
  bool has_bitcode);

// Generate code for a for_each operator. Adapts c.parallel's user-op contract
// (`void op(T*)`) to the contract that cub::DeviceFor::ForEachN expects
// (`void op(T&)`). Functor is stateless for non-stateful ops; for stateful
// ops it embeds the state bytes inline so they ride along into device
// constant memory via the kernel-arg copy.
OperatorCode make_for_each_op(
  cccl_op_t op,
  const std::string& elem_type, // C++ type name for the iterator's element
  const std::string& functor_name, // e.g., "ForEachOp"
  const std::string& var_name, // e.g., "op_0"
  const std::string& state_param, // e.g., "op_0_state"
  bool has_bitcode);
} // namespace hostjit::codegen
