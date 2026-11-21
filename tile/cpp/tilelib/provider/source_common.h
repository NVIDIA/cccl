// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <string>
#include <vector>

namespace cuda::coop::tile::detail::provider
{
[[nodiscard]] std::string replace_all(std::string source, const std::string& from, const std::string& to);
[[nodiscard]] std::string build_include_block(const std::vector<std::string>& includes);
[[nodiscard]] bool is_source_symbol_identifier(const std::string& symbol) noexcept;
[[nodiscard]] bool is_integer_dtype_suffix(const std::string& dtype_suffix);
[[nodiscard]] bool decode_dtype_suffix(
  const std::string& dtype_suffix, std::string* cpp_type, std::string* extra_include, std::string* zero_expression);
[[nodiscard]] int bitwidth_for_dtype_suffix(const std::string& dtype_suffix);
[[nodiscard]] std::string resolve_reduce_algorithm_token(const std::string& token);
[[nodiscard]] std::string resolve_scan_algorithm_token(const std::string& token);
[[nodiscard]] std::string resolve_smem_config_token(const std::string& token);
[[nodiscard]] bool resolve_binary_op_token(const std::string& token, std::string* expression);
} // namespace cuda::coop::tile::detail::provider
