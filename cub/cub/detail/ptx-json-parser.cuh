// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#include <cuda/std/__cccl/algorithm_wrapper.h>

#include <format>
#include <string_view>

#include <nlohmann/json.hpp>

CUB_NAMESPACE_BEGIN

namespace detail::ptx_json
{
inline nlohmann::json parse(std::string_view tag, std::span<const char> cubin)
{
  auto const open_tag      = std::format("cccl.ptx_json.begin(\"{}\")", tag);
  auto const open_location = std::ranges::search(cubin, open_tag);
  if (std::ranges::size(open_location) != open_tag.size())
  {
    return nullptr;
  }

  auto const close_tag      = std::format("cccl.ptx_json.end(\"{}\")", tag);
  auto const close_location = std::ranges::search(cubin, close_tag);
  if (std::ranges::size(close_location) != close_location.size())
  {
    return nullptr;
  }

  return nlohmann::json::parse(std::ranges::end(open_location), std::ranges::begin(close_location), nullptr, true, true);
}
} // namespace detail::ptx_json

CUB_NAMESPACE_END
