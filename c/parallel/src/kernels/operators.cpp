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

#include "cccl/c/types.h"
#include <kernels/operators.h>
#include <util/errors.h>
#include <util/types.h>

constexpr std::string_view binary_op_template = R"XXX(
#define LHS_T {0}
#define RHS_T {1}
#define OP_NAME {2}
#define OP_ALIGNMENT {3}
#define OP_SIZE {4}

// Source
{5}

#undef LHS_T
#undef RHS_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_SIZE
)XXX";

constexpr std::string_view stateless_binary_op_template = R"XXX(
extern "C" __device__ {0} OP_NAME(LHS_T lhs, RHS_T rhs);
struct op_wrapper {{
  __device__ {0} operator()(LHS_T lhs, RHS_T rhs) const {{
    return OP_NAME(lhs, rhs);
  }}
}};
)XXX";

constexpr std::string_view stateful_binary_op_template = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {{
  char data[OP_SIZE];
}};
extern "C" __device__ {0} OP_NAME(op_state *state, LHS_T lhs, RHS_T rhs);
struct op_wrapper {{
  op_state state;
  __device__ {0} operator()(LHS_T lhs, RHS_T rhs) {{
    return OP_NAME(&state, lhs, rhs);
  }}
}};
)XXX";

std::string make_kernel_binary_operator_full_source(
  std::string_view lhs_t, std::string_view rhs_t, cccl_op_t operation, std::string_view return_type)
{
  const std::string op_alignment =
    operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::format("{}", operation.alignment);
  const std::string op_size = operation.type == cccl_op_kind_t::CCCL_STATELESS ? "" : std::format("{}", operation.size);

  return std::format(
    binary_op_template,
    lhs_t,
    rhs_t,
    operation.name,
    op_alignment,
    op_size,
    operation.type == cccl_op_kind_t::CCCL_STATELESS
      ? std::format(stateless_binary_op_template, return_type)
      : std::format(stateful_binary_op_template, return_type));
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

// Source
{5}

#undef INPUT_T
#undef OUTPUT_T
#undef OP_NAME
#undef OP_ALIGNMENT
#undef OP_SIZE
)XXX";

  constexpr std::string_view stateless_op = R"XXX(
extern "C" __device__ OUTPUT_T OP_NAME(INPUT_T val);
struct op_wrapper {
  __device__ OUTPUT_T operator()(INPUT_T val) const {
    return OP_NAME(val);
  }
};
)XXX";

  constexpr std::string_view stateful_op = R"XXX(
struct __align__(OP_ALIGNMENT) op_state {
  char data[OP_SIZE];
};
extern "C" __device__ OUTPUT_T OP_NAME(op_state* state, INPUT_T val);
struct op_wrapper
{
  op_state state;
  __device__ OUTPUT_T operator()(INPUT_T val)
  {
    return OP_NAME(&state, val);
  }
};

)XXX";

  return (operation.type == cccl_op_kind_t::CCCL_STATELESS)
         ? std::format(unary_op_template, input_t, output_t, operation.name, "", "", stateless_op)
         : std::format(
             unary_op_template, input_t, output_t, operation.name, operation.alignment, operation.size, stateful_op);
}
