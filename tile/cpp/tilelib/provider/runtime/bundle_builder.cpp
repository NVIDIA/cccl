// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "provider/runtime/runtime.h"
#include "provider/spec_identity.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
[[nodiscard]] bool starts_with(const std::string& value, const std::string& prefix)
{
  return prefix.size() <= value.size() && value.compare(0, prefix.size(), prefix) == 0;
}

[[nodiscard]] std::string replace_all(std::string source, const std::string& from, const std::string& to)
{
  size_t pos = 0;
  while ((pos = source.find(from, pos)) != std::string::npos)
  {
    source.replace(pos, from.size(), to);
    pos += to.size();
  }
  return source;
}

[[nodiscard]] bool rewrite_macro_define_value(std::string* source, const std::string& macro, int value)
{
  if (!source || macro.empty())
  {
    return false;
  }
  const std::string needle = "#define " + macro + " ";
  size_t search            = 0;
  bool replaced            = false;
  while (true)
  {
    const size_t pos = source->find(needle, search);
    if (pos == std::string::npos)
    {
      break;
    }
    const size_t line_end  = source->find('\n', pos);
    const std::string line = needle + std::to_string(value);
    if (line_end == std::string::npos)
    {
      source->replace(pos, source->size() - pos, line);
      replaced = true;
      break;
    }
    source->replace(pos, line_end - pos, line);
    search   = pos + line.size();
    replaced = true;
  }
  return replaced;
}
} // namespace

// Give one source fragment private topology macros and concrete values.
// Mutate the source in place before combining wrappers that may use different
// worker widths. A false result can leave a partially rewritten string; callers
// must discard that fragment instead of compiling it.
bool specialize_source_for_bundle(std::string* source, const CallMetadata& meta, size_t fragment_index)
{
  if (!source || !meta.request)
  {
    return false;
  }
  const ImplementationSpec& request = *meta.request;
  const std::string identity        = semantic_identity(request);
  const std::string block_macro     = "CUDA_COOP_TILE_BLOCK_THREADS_BUNDLE_" + std::to_string(fragment_index);
  const std::string items_macro     = "CUDA_COOP_TILE_ITEMS_PER_THREAD_BUNDLE_" + std::to_string(fragment_index);

  *source = replace_all(std::move(*source), "CUDA_COOP_TILE_BLOCK_THREADS", block_macro);
  *source = replace_all(std::move(*source), "CUDA_COOP_TILE_ITEMS_PER_THREAD", items_macro);
  if (!rewrite_macro_define_value(source, block_macro, request.topology().block_threads))
  {
    emit_trace_line(
      "[coop_linkage_provider][trace] failed to specialize block_threads macro for semantic_key=" + identity);
    return false;
  }
  if (!rewrite_macro_define_value(source, items_macro, request.topology().items_per_thread))
  {
    emit_trace_line(
      "[coop_linkage_provider][trace] failed to specialize items_per_thread macro for semantic_key=" + identity);
    return false;
  }
  return true;
}

// Separate leading includes so a bundle can deduplicate them at file scope.
// Once the first body line appears, preserve subsequent lines in the body.
// This is the format emitted by family templates, not a general C++ parser;
// preprocessor structure later in a template must stay inside its fragment.
void split_source_includes_and_body(
  const std::string& source, std::vector<std::string>* out_includes, std::string* out_body)
{
  if (!out_includes || !out_body)
  {
    return;
  }
  out_includes->clear();
  out_body->clear();

  std::istringstream input(source);
  std::ostringstream body;
  std::string line;
  bool in_prefix = true;
  while (std::getline(input, line))
  {
    const size_t first_non_space = line.find_first_not_of(" \t");
    const std::string trimmed    = first_non_space == std::string::npos ? std::string() : line.substr(first_non_space);

    if (in_prefix && (trimmed.empty() || starts_with(trimmed, "#include")))
    {
      if (starts_with(trimmed, "#include"))
      {
        out_includes->push_back(trimmed);
      }
      continue;
    }

    in_prefix = false;
    body << line << '\n';
  }

  *out_body = body.str();
}

std::vector<std::string> collect_bundle_include_paths(const std::vector<CallMetadata>& metas)
{
  std::vector<std::string> include_paths;
  std::unordered_set<std::string> seen;
  for (const CallMetadata& meta : metas)
  {
    for (const std::string& path : meta.include_paths)
    {
      if (path.empty())
      {
        continue;
      }
      if (seen.insert(path).second)
      {
        include_paths.push_back(path);
      }
    }
  }
  return include_paths;
}
} // namespace cuda::coop::tile::detail::provider
