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

// Generate a well-known binary operation body (e.g., CCCL_PLUS → "*out = *a + *b").
// Returns "" for unknown ops.
std::string get_well_known_op_body(cccl_op_kind_t kind, const std::string& type_name);

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
} // namespace hostjit::codegen
