// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/source_common.h"

#include <unordered_set>

namespace cuda::coop::tile::detail::provider
{
std::string replace_all(std::string source, const std::string& from, const std::string& to)
{
  size_t position = 0;
  while ((position = source.find(from, position)) != std::string::npos)
  {
    source.replace(position, from.size(), to);
    position += to.size();
  }
  return source;
}

std::string build_include_block(const std::vector<std::string>& includes)
{
  std::unordered_set<std::string> seen;
  std::string result;
  for (const auto& include : includes)
  {
    if (include.empty() || !seen.insert(include).second)
    {
      continue;
    }
    result += "#include <" + include + ">\n";
  }
  return result;
}

bool is_source_symbol_identifier(const std::string& symbol) noexcept
{
  const auto is_alpha_or_underscore = [](unsigned char ch) {
    return ch == '_' || (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z');
  };
  const auto is_digit = [](unsigned char ch) {
    return ch >= '0' && ch <= '9';
  };
  if (symbol.empty() || !is_alpha_or_underscore(static_cast<unsigned char>(symbol.front())))
  {
    return false;
  }
  for (size_t i = 1; i < symbol.size(); ++i)
  {
    const unsigned char ch = static_cast<unsigned char>(symbol[i]);
    if (!is_alpha_or_underscore(ch) && !is_digit(ch))
    {
      return false;
    }
  }
  return true;
}

bool is_integer_dtype_suffix(const std::string& dtype_suffix)
{
  return dtype_suffix == "i8" || dtype_suffix == "u8" || dtype_suffix == "i16" || dtype_suffix == "u16"
      || dtype_suffix == "i32" || dtype_suffix == "u32" || dtype_suffix == "i64" || dtype_suffix == "u64";
}

// Map the validated dtype token to C++ spelling, headers, and a typed zero.
// Half types need CUDA conversion expressions; integer signedness comes from
// the token rather than TileIR's signless element type. On false, ignore all
// output fields because only successful branches fill the complete set.
bool decode_dtype_suffix(
  const std::string& dtype_suffix, std::string* cpp_type, std::string* extra_include, std::string* zero_expression)
{
  if (!cpp_type || !extra_include || !zero_expression)
  {
    return false;
  }
  if (dtype_suffix == "f16")
  {
    *cpp_type        = "__half";
    *extra_include   = "cuda_fp16.h";
    *zero_expression = "__float2half_rn(0.0f)";
    return true;
  }
  if (dtype_suffix == "bf16")
  {
    *cpp_type        = "__nv_bfloat16";
    *extra_include   = "cuda_bf16.h";
    *zero_expression = "__float2bfloat16_rn(0.0f)";
    return true;
  }
  if (dtype_suffix == "f32")
  {
    *cpp_type        = "float";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "f64")
  {
    *cpp_type        = "double";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "i8")
  {
    *cpp_type        = "signed char";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "u8")
  {
    *cpp_type        = "unsigned char";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "i16")
  {
    *cpp_type        = "short";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "u16")
  {
    *cpp_type        = "unsigned short";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "i32")
  {
    *cpp_type        = "int";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "u32")
  {
    *cpp_type        = "unsigned int";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "i64")
  {
    *cpp_type        = "long long";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  if (dtype_suffix == "u64")
  {
    *cpp_type        = "unsigned long long";
    *extra_include   = "";
    *zero_expression = "static_cast<ValueT>(0)";
    return true;
  }
  return false;
}

int bitwidth_for_dtype_suffix(const std::string& dtype_suffix)
{
  if (dtype_suffix == "i8" || dtype_suffix == "u8")
  {
    return 8;
  }
  if (dtype_suffix == "i16" || dtype_suffix == "u16" || dtype_suffix == "f16" || dtype_suffix == "bf16")
  {
    return 16;
  }
  if (dtype_suffix == "i32" || dtype_suffix == "u32" || dtype_suffix == "f32")
  {
    return 32;
  }
  if (dtype_suffix == "i64" || dtype_suffix == "u64" || dtype_suffix == "f64")
  {
    return 64;
  }
  return 0;
}

std::string resolve_reduce_algorithm_token(const std::string& token)
{
  if (token == "raking_commutative_only")
  {
    return "cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY";
  }
  if (token == "raking")
  {
    return "cub::BLOCK_REDUCE_RAKING";
  }
  if (token == "warp_reductions")
  {
    return "cub::BLOCK_REDUCE_WARP_REDUCTIONS";
  }
  return "";
}

std::string resolve_scan_algorithm_token(const std::string& token)
{
  if (token == "raking")
  {
    return "cub::BLOCK_SCAN_RAKING";
  }
  if (token == "raking_memoize")
  {
    return "cub::BLOCK_SCAN_RAKING_MEMOIZE";
  }
  if (token == "warp_scans")
  {
    return "cub::BLOCK_SCAN_WARP_SCANS";
  }
  return "";
}

std::string resolve_smem_config_token(const std::string& token)
{
  if (token.empty() || token == "default" || token == "four_byte" || token == "4")
  {
    return "cudaSharedMemBankSizeFourByte";
  }
  if (token == "eight_byte" || token == "8")
  {
    return "cudaSharedMemBankSizeEightByte";
  }
  return "";
}

bool resolve_binary_op_token(const std::string& token, std::string* expression)
{
  if (!expression)
  {
    return false;
  }
  if (token == "mul")
  {
    *expression = "a * b";
    return true;
  }
  if (token == "min")
  {
    *expression = "(b < a) ? b : a";
    return true;
  }
  if (token == "max")
  {
    *expression = "(a < b) ? b : a";
    return true;
  }
  if (token == "bit_and")
  {
    *expression = "a & b";
    return true;
  }
  if (token == "bit_or")
  {
    *expression = "a | b";
    return true;
  }
  if (token == "bit_xor")
  {
    *expression = "a ^ b";
    return true;
  }
  if (token == "sum")
  {
    *expression = "a + b";
    return true;
  }
  return false;
}
} // namespace cuda::coop::tile::detail::provider
