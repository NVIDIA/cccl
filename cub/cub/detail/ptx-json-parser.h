/******************************************************************************
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#include <thrust/detail/algorithm_wrapper.h>

#include <format>
#include <string_view>

#include <nlohmann/json.hpp>

CUB_NAMESPACE_BEGIN

namespace detail::ptx_json
{
inline nlohmann::json parse(std::string_view tag, std::string_view ptx_stream)
{
  auto const open_tag      = std::format("cccl.ptx_json.begin({})", tag);
  auto const open_location = std::ranges::search(ptx_stream, open_tag);
  if (std::ranges::size(open_location) != open_tag.size())
  {
    return nullptr;
  }

  auto const close_tag      = std::format("cccl.ptx_json.end({})", tag);
  auto const close_location = std::ranges::search(ptx_stream, close_tag);
  if (std::ranges::size(close_location) != close_location.size())
  {
    return nullptr;
  }

  return nlohmann::json::parse(std::ranges::end(open_location), std::ranges::begin(close_location), nullptr, true, true);
}
} // namespace detail::ptx_json

CUB_NAMESPACE_END
