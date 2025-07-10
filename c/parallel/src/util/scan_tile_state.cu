//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <format>
#include <regex>

#include "scan_tile_state.h"

// TODO: NVRTC doesn't currently support extracting basic type
// information (e.g., type sizes and alignments) from compiled
// LTO-IR. So we separately compile a small PTX file that defines the
// necessary types and constants and grep it for the required
// information. If/when NVRTC adds these features, we can remove this
// extra compilation step and get the information directly from the
// LTO-IR.
static constexpr auto ptx_u64_assignment_regex = R"(\.visible\s+\.global\s+\.align\s+\d+\s+\.u64\s+{}\s*=\s*(\d+);)";

std::optional<size_t> find_size_t(char* ptx, std::string_view name)
{
  std::regex regex(std::format(ptx_u64_assignment_regex, name));
  std::cmatch match;
  if (std::regex_search(ptx, match, regex))
  {
    auto result = std::stoi(match[1].str());
    return result;
  }
  return std::nullopt;
}

std::pair<size_t, size_t> get_tile_state_bytes_per_tile(
  cccl_type_info accum_t,
  const std::string& accum_cpp,
  const char** ptx_args,
  size_t num_ptx_args,
  const std::string& arch)
{
  constexpr size_t num_ptx_lto_args       = 3;
  const char* ptx_lopts[num_ptx_lto_args] = {"-lto", arch.c_str(), "-ptx"};

  constexpr std::string_view ptx_src_template = R"XXX(
        #include <cub/agent/single_pass_scan_operators.cuh>
        #include <cub/util_type.cuh>
        struct __align__({1}) storage_t {{
           char data[{0}];
        }};
        __device__ size_t description_bytes_per_tile = cub::ScanTileState<{2}>::description_bytes_per_tile;
        __device__ size_t payload_bytes_per_tile = cub::ScanTileState<{2}>::payload_bytes_per_tile;
        )XXX";

  const std::string ptx_src = std::format(ptx_src_template, accum_t.size, accum_t.alignment, accum_cpp);
  auto compile_result =
    begin_linking_nvrtc_program(num_ptx_lto_args, ptx_lopts)
      ->add_program(nvrtc_translation_unit{ptx_src.c_str(), "tile_state_info"})
      ->compile_program({ptx_args, num_ptx_args})
      ->link_program()
      ->finalize_program();
  auto ptx_code = compile_result.data.get();

  size_t description_bytes_per_tile;
  size_t payload_bytes_per_tile;
  auto maybe_description_bytes_per_tile = find_size_t(ptx_code, "description_bytes_per_tile");
  if (maybe_description_bytes_per_tile)
  {
    description_bytes_per_tile = maybe_description_bytes_per_tile.value();
  }
  else
  {
    throw std::runtime_error("Failed to find description_bytes_per_tile in PTX");
  }
  payload_bytes_per_tile = find_size_t(ptx_code, "payload_bytes_per_tile").value_or(0);

  return {description_bytes_per_tile, payload_bytes_per_tile};
}
