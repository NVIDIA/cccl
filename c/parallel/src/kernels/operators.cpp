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
  "float",
  "double",
  "bool",
};

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
extern "C" __device__ void OP_NAME(const void* lhs, const void* rhs, void* out);
struct op_wrapper {{
  __device__ {0} operator()(LHS_T lhs, RHS_T rhs) const {{
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
extern "C" __device__ void OP_NAME(void* state, const void* lhs, const void* rhs, void* out);
struct op_wrapper {{
  op_state state;
  __device__ {0} operator()(LHS_T lhs, RHS_T rhs) {{
    {0} ret;
    OP_NAME(&state, &lhs, &rhs, &ret);
    return ret;
  }}
}};
)XXX";

std::string make_kernel_binary_operator_full_source(
  std::string_view lhs_t, std::string_view rhs_t, cccl_op_t operation, std::string_view return_type)
{
  if (primitive_types.contains(lhs_t) && primitive_types.contains(rhs_t) && primitive_types.contains(return_type))
  {
    switch (operation.type)
    {
      case CCCL_PLUS:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::plus<>;\n";
      case CCCL_MINUS:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::minus<>;\n";
      case CCCL_MULTIPLIES:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::multiplies<>;\n";
      case CCCL_DIVIDES:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::divides<>;\n";
      case CCCL_MODULUS:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::modulus<>;\n";
      case CCCL_EQUAL_TO:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::equal_to<>;\n";
      case CCCL_NOT_EQUAL_TO:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::not_equal_to<>;\n";
      case CCCL_GREATER:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::greater<>;\n";
      case CCCL_LESS:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::less<>;\n";
      case CCCL_GREATER_EQUAL:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::greater_equal<>;\n";
      case CCCL_LESS_EQUAL:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::less_equal<>;\n";
      case CCCL_LOGICAL_AND:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::logical_and<>;\n";
      case CCCL_LOGICAL_OR:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::logical_or<>;\n";
      case CCCL_BIT_AND:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::bit_and<>;\n";
      case CCCL_BIT_OR:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::bit_or<>;\n";
      case CCCL_BIT_XOR:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::bit_xor<>;\n";
      case CCCL_STATEFUL:
        [[fallthrough]];
      case CCCL_STATELESS:
        break;
      default:
        throw std::runtime_error("c.parallel: invalid well-known operation specified for a binary cccl_op_t.");
    }
  }

  const std::string op_alignment =
    operation.type == cccl_op_kind_t::CCCL_STATEFUL ? std::format("{}", operation.alignment) : "";
  const std::string op_size = operation.type == cccl_op_kind_t::CCCL_STATEFUL ? std::format("{}", operation.size) : "";

  return std::format(
    binary_op_template,
    lhs_t,
    rhs_t,
    operation.name,
    op_alignment,
    op_size,
    operation.type == cccl_op_kind_t::CCCL_STATEFUL
      ? std::format(stateful_binary_op_template, return_type)
      : std::format(stateless_binary_op_template, return_type));
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
extern "C" __device__  void OP_NAME(const void* val, void* result);
struct op_wrapper {
  __device__ OUTPUT_T operator()(INPUT_T val) const {
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
extern "C" __device__ void OP_NAME(op_state* state, const void* val, void* result);
struct op_wrapper
{
  op_state state;
  __device__ OUTPUT_T operator()(INPUT_T val)
  {
    OUTPUT_T out;
    OP_NAME(&state, &val, &out);
    return out;
  }
};

)XXX";

  if (primitive_types.contains(input_t) && primitive_types.contains(output_t))
  {
    switch (operation.type)
    {
      case CCCL_LOGICAL_NOT:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::logical_not<>;\n";
      case CCCL_BIT_NOT:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::bit_not<>;\n";
      case CCCL_IDENTITY:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::identity<>;\n";
      case CCCL_NEGATE:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::std::negate<>;\n";
      case CCCL_MINIMUM:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::minimum<>;\n";
      case CCCL_MAXIMUM:
        return "#include <cuda/std/functional>\nusing op_wrapper = ::cuda::maximum<>;\n";
      case CCCL_STATEFUL:
        [[fallthrough]];
      case CCCL_STATELESS:
        break;
      default:
        throw std::runtime_error("c.parallel: invalid well-known operation specified for a unary cccl_op_t.");
    }
  }

  return (operation.type == cccl_op_kind_t::CCCL_STATEFUL)
         ? std::format(
             unary_op_template, input_t, output_t, operation.name, operation.alignment, operation.size, stateful_op)
         : std::format(unary_op_template, input_t, output_t, operation.name, "", "", stateless_op);
}
