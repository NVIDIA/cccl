//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <format>
#include <string_view>
#include <unordered_set>

#include "cccl/c/types.h"
#include <jit_templates/templates/operation.h>
#include <kernels/operators.h>
#include <util/errors.h>
#include <util/types.h>

std::unordered_set<std::string_view> primitive_types = {
  "::cuda::std::int8_t",
  "::cuda::std::uint8_t",
  "::cuda::std::int16_t",
  "::cuda::std::uint16_t",
  "::cuda::std::int32_t",
  "::cuda::std::uint32_t",
  "::cuda::std::int64_t",
  "::cuda::std::uint64_t",
  "__half",
  "float",
  "double",
  "bool",
};

std::string format_state_bytes(const void* state, size_t size)
{
  if (state == nullptr || size == 0)
  {
    return "0";
  }

  const auto* bytes = static_cast<const unsigned char*>(state);
  std::string result;
  result.reserve(size * 5);
  for (size_t i = 0; i < size; ++i)
  {
    if (i > 0)
    {
      result += ", ";
    }
    result += std::format("0x{:02x}", bytes[i]);
  }
  return result;
}

constexpr std::string_view binary_op_template = R"XXX(
#define LHS_T {0}
#define RHS_T {1}
#define OP_NAME {2}
#define OP_ALIGNMENT {3}
#define OP_SIZE {4}
#define OP_STATE_BYTES {5}

// Source
{6}

#undef LHS_T
#undef RHS_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_STATE_BYTES
#undef OP_SIZE
)XXX";

constexpr std::string_view stateless_binary_op_template = R"XXX(
extern "C" __device__ void OP_NAME(const void* lhs, const void* rhs, void* out);
struct op_wrapper {{
  __device__ {0} operator()(const LHS_T& lhs, const RHS_T& rhs) const {{
    {0} ret;
    OP_NAME(&lhs, &rhs, &ret);
    return ret;
  }}
}};
)XXX";

constexpr std::string_view stateful_binary_op_template = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {{
  char data[OP_SIZE];
}};
__device__ __constant__ op_state op_state_instance = {{OP_STATE_BYTES}};
extern "C" __device__ void OP_NAME(void* state, const void* lhs, const void* rhs, void* out);
struct op_wrapper {{
  __device__ {0} operator()(const LHS_T& lhs, const RHS_T& rhs) {{
    {0} ret;
    OP_NAME(const_cast<op_state*>(&op_state_instance), &lhs, &rhs, &ret);
    return ret;
  }}
}};
)XXX";

std::string make_kernel_binary_operator_full_source(
  std::string_view lhs_t, std::string_view rhs_t, cccl_op_t operation, std::string_view return_type)
{
  if (lhs_t == rhs_t && (return_type == lhs_t || return_type == "bool"))
  {
    auto desc = user_operation_traits::well_known_operation_description(operation.type);
    if (desc && (desc->symbol || primitive_types.contains(lhs_t)))
    {
      if (desc->check != user_operation_traits::binary_predicate_matcher
          && desc->check != user_operation_traits::binary_matcher)
      {
        throw std::runtime_error(
          std::format("c.parallel: invalid well-known operation '{}' specified for a binary cccl_op_t.",
                      std::format(desc->name, +"")));
      }

      std::string ret =
        std::format("#include <cuda/std/functional>\nusing op_wrapper = {};\n", std::format(desc->name, lhs_t.data()));
      if (!primitive_types.contains(lhs_t))
      {
        std::string_view type_names[] = {return_type, lhs_t, rhs_t};
        ret += user_operation_traits::binary_builder(cuda::std::span(type_names), *desc->symbol, operation.name);
      }
      return ret;
    }
  }

  const bool is_stateful           = operation.type == cccl_op_kind_t::CCCL_STATEFUL;
  const std::string op_alignment   = is_stateful ? std::format("{}", operation.alignment) : "";
  const std::string op_size        = is_stateful ? std::format("{}", operation.size) : "";
  const std::string op_state_bytes = is_stateful ? format_state_bytes(operation.state, operation.size) : "0";
  const std::string op_src =
    std::vformat(std::string(is_stateful ? stateful_binary_op_template : stateless_binary_op_template),
                 std::make_format_args(return_type));

  return std::format(binary_op_template, lhs_t, rhs_t, operation.name, op_alignment, op_size, op_state_bytes, op_src);
}

std::string make_kernel_user_binary_operator(
  std::string_view lhs_t, std::string_view rhs_t, std::string_view output_t, cccl_op_t operation)
{
  return make_kernel_binary_operator_full_source(lhs_t, rhs_t, operation, output_t);
}

std::string make_kernel_user_comparison_operator(std::string_view input_t, cccl_op_t operation)
{
  return make_kernel_binary_operator_full_source(input_t, input_t, operation, "bool");
}

std::string make_kernel_user_unary_operator(std::string_view input_t, std::string_view output_t, cccl_op_t operation)
{
  constexpr std::string_view unary_op_template = R"XXX(
#define INPUT_T {0}
#define OUTPUT_T {1}
#define OP_NAME {2}
#define OP_ALIGNMENT {3}
#define OP_SIZE {4}
#define OP_STATE_BYTES {5}

// Source
{6}

#undef INPUT_T
#undef OUTPUT_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_STATE_BYTES
#undef OP_SIZE
)XXX";

  constexpr std::string_view stateless_op = R"XXX(
extern "C" __device__  void OP_NAME(const void* val, void* result);
struct op_wrapper {
  __device__ OUTPUT_T operator()(const INPUT_T& val) const {
    OUTPUT_T out;
    OP_NAME(&val, &out);
    return out;
  }
};
)XXX";

  constexpr std::string_view stateful_op = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {
  char data[OP_SIZE];
};
__device__ __constant__ op_state op_state_instance = {OP_STATE_BYTES};
extern "C" __device__ void OP_NAME(op_state* state, const void* val, void* result);
struct op_wrapper
{
  __device__ OUTPUT_T operator()(const INPUT_T& val)
  {
    OUTPUT_T out;
    OP_NAME(const_cast<op_state*>(&op_state_instance), &val, &out);
    return out;
  }
};

)XXX";

  if (output_t == input_t || output_t == "bool")
  {
    auto desc = user_operation_traits::well_known_operation_description(operation.type);
    if (desc && (desc->symbol || primitive_types.contains(input_t)))
    {
      if (desc->check != user_operation_traits::unary_predicate_matcher
          && desc->check != user_operation_traits::unary_matcher)
      {
        throw std::runtime_error(
          std::format("c.parallel: invalid well-known operation '{}' specified for a unary cccl_op_t.",
                      std::format(desc->name, +"")));
      }

      std::string_view type_names[] = {output_t, input_t};
      std::string ret               = std::format(
        "#include <cuda/std/functional>\nusing op_wrapper = {};\n", std::format(desc->name, input_t.data()));
      if (!primitive_types.contains(input_t))
      {
        ret += user_operation_traits::unary_builder(cuda::std::span(type_names), *desc->symbol, operation.name);
      }
      return ret;
    }
  }

  const bool is_stateful           = operation.type == cccl_op_kind_t::CCCL_STATEFUL;
  const std::string op_state_bytes = is_stateful ? format_state_bytes(operation.state, operation.size) : "0";
  const std::string op_src         = is_stateful ? std::string(stateful_op) : std::string(stateless_op);

  return is_stateful
         ? std::format(
             unary_op_template,
             input_t,
             output_t,
             operation.name,
             operation.alignment,
             operation.size,
             op_state_bytes,
             op_src)
         : std::format(unary_op_template, input_t, output_t, operation.name, "", "", op_state_bytes, op_src);
}
