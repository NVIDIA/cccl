// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#include "provider/runtime/runtime.h"
#include "provider/spec_identity.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
inline constexpr size_t default_seen_artifact_cache_entries = 4096;
inline constexpr size_t default_bundle_cache_entries        = 128;
inline constexpr size_t default_resource_cache_entries      = 512;

[[nodiscard]] bool parse_int64_strict(const std::string& value, int64_t* out)
{
  if (!out || value.empty())
  {
    return false;
  }
  size_t consumed = 0;
  try
  {
    const long long parsed = std::stoll(value, &consumed, 10);
    if (consumed != value.size())
    {
      return false;
    }
    *out = static_cast<int64_t>(parsed);
    return true;
  }
  catch (...)
  {
    return false;
  }
}

[[nodiscard]] size_t cache_entry_limit_from_env(const char* env_key, size_t default_value)
{
  const char* const raw = std::getenv(env_key);
  if (!raw || raw[0] == '\0')
  {
    return default_value;
  }
  int64_t parsed = 0;
  if (!parse_int64_strict(raw, &parsed) || parsed < 0)
  {
    return default_value;
  }
  return static_cast<size_t>(parsed);
}
} // namespace

uint64_t fnv1a_hash(const std::string& value) noexcept
{
  uint64_t hash = 1469598103934665603ULL;
  for (unsigned char ch : value)
  {
    hash ^= static_cast<uint64_t>(ch);
    hash *= 1099511628211ULL;
  }
  return hash;
}

std::string hex_u64(uint64_t value)
{
  std::ostringstream oss;
  oss << std::hex << value;
  return oss.str();
}

size_t seen_artifact_cache_limit()
{
  return cache_entry_limit_from_env("CUDA_COOP_TILE_PROVIDER_SEEN_CACHE_ENTRIES", default_seen_artifact_cache_entries);
}

size_t bundle_cache_limit()
{
  return cache_entry_limit_from_env("CUDA_COOP_TILE_PROVIDER_BUNDLE_CACHE_ENTRIES", default_bundle_cache_entries);
}

size_t resource_cache_limit()
{
  return cache_entry_limit_from_env("CUDA_COOP_TILE_PROVIDER_RESOURCE_CACHE_ENTRIES", default_resource_cache_entries);
}

TargetArchitecture parse_sm_arch(TileIRStringView architecture)
{
  if (!architecture.data || architecture.len == 0)
  {
    return {};
  }

  std::string_view arch{architecture.data, architecture.len};
  constexpr std::string_view sm_prefix      = "sm_";
  constexpr std::string_view compute_prefix = "compute_";
  if (arch.rfind(sm_prefix, 0) == 0)
  {
    arch.remove_prefix(sm_prefix.size());
  }
  else if (arch.rfind(compute_prefix, 0) == 0)
  {
    arch.remove_prefix(compute_prefix.size());
  }

  if (arch.empty())
  {
    return {};
  }

  TargetArchitectureFamily family = TargetArchitectureFamily::none;
  if (arch.back() == 'a')
  {
    family = TargetArchitectureFamily::a;
    arch.remove_suffix(1);
  }
  else if (arch.back() == 'f')
  {
    family = TargetArchitectureFamily::f;
    arch.remove_suffix(1);
  }
  if (arch.empty())
  {
    return {};
  }

  int code          = 0;
  const auto parsed = std::from_chars(arch.data(), arch.data() + arch.size(), code, 10);
  if (parsed.ec != std::errc{} || parsed.ptr != arch.data() + arch.size() || code <= 0)
  {
    return {};
  }
  return TargetArchitecture{code, family};
}

int64_t parse_cache_epoch_from_env()
{
  const char* const epoch_env = std::getenv("CUDA_COOP_TILE_PROVIDER_CACHE_EPOCH");
  if (!epoch_env || epoch_env[0] == '\0')
  {
    return 0;
  }
  char* end_ptr          = nullptr;
  const long long parsed = std::strtoll(epoch_env, &end_ptr, 10);
  if (!end_ptr || *end_ptr != '\0')
  {
    return 0;
  }
  return static_cast<int64_t>(parsed);
}

// Identify emitted code by selected symbol, topology, target, and static options.
// Runtime scalar operands are absent because they do not change source. This
// key is used for trace deduplication and as a component of the bundle key.
std::string make_codegen_artifact_cache_key(const CallMetadata& meta, const std::string& callable_symbol)
{
  if (!meta.request || callable_symbol.empty())
  {
    return {};
  }

  const ImplementationSpec& request    = *meta.request;
  const std::string payload            = request_payload(request.primitive());
  const std::string primitive_metadata = codegen_metadata(request);

  std::ostringstream oss;
  oss << "kind=primitive|callable=" << callable_symbol << "|sm=" << request.target().sm_arch.sm_name()
      << "|epoch=" << request.target().cache_epoch << "|threads=" << request.topology().block_threads
      << "|ipt=" << request.topology().items_per_thread;
  oss << "|payload=" << payload;
  if (!primitive_metadata.empty())
  {
    oss << "|primitive_meta_hash=" << hex_u64(fnv1a_hash(primitive_metadata));
  }
  return oss.str();
}

std::string ordered_include_paths_hash_key(const std::vector<std::string>& include_paths)
{
  std::ostringstream oss;
  for (const std::string& path : include_paths)
  {
    oss << "|include=" << path;
  }
  return hex_u64(fnv1a_hash(oss.str()));
}

// Build an order-independent set-of-artifacts key with ordered include paths.
// Artifact order does not change the exported implementations; include search
// order can change which header is used. These keys do not hash header contents
// or NVRTC binaries. Development changes at the same paths need a new cache
// epoch or provider lifetime.
std::string make_bundle_cache_key(
  TargetArchitecture target_arch,
  int64_t target_epoch,
  const std::vector<std::string>& artifact_keys,
  const std::vector<std::string>& include_paths)
{
  std::vector<std::string> sorted_artifact_keys = artifact_keys;
  std::sort(sorted_artifact_keys.begin(), sorted_artifact_keys.end());
  std::ostringstream include_key_builder;
  for (const auto& path : include_paths)
  {
    include_key_builder << "|include=" << path;
  }

  std::ostringstream oss;
  oss << "sm=" << target_arch.sm_name() << "|epoch=" << target_epoch;
  for (const auto& key : sorted_artifact_keys)
  {
    oss << "|key=" << key;
  }
  if (!include_paths.empty())
  {
    oss << "|include_hash=" << hex_u64(fnv1a_hash(include_key_builder.str()));
  }
  return oss.str();
}
} // namespace cuda::coop::tile::detail::provider
