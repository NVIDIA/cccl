#include <algorithm>
#include <format>
#include <iostream>

#include <cccl/ptx-json-parser.h>

nlohmann::json ptx_json::parser::parse(std::string_view id, std::string_view ptx_stream)
{
  auto const open_tag      = std::format("cccl.ptx_json.begin({})", id);
  auto const open_location = std::ranges::search(ptx_stream, open_tag);
  if (std::ranges::size(open_location) != open_tag.size())
  {
    return nullptr;
  }

  auto const close_tag      = std::format("cccl.ptx_json.end({})", id);
  auto const close_location = std::ranges::search(ptx_stream, close_tag);
  if (std::ranges::size(close_location) != close_location.size())
  {
    return nullptr;
  }

  std::cerr << "asdf\n";
  return nlohmann::json::parse(std::ranges::end(open_location), std::ranges::begin(close_location), nullptr, true, true);
}
