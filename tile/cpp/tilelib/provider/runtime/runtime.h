// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CUDA_COOP_TILE_PROVIDER_RUNTIME_H
#define CUDA_COOP_TILE_PROVIDER_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "provider/primitive_spec.h"
#include "tile_ir/linkage_provider/LinkageProviderAPI.h"

namespace cuda::coop::tile::detail::provider
{
inline constexpr uint64_t resource_query_sentinel   = 0x43554f5052455355ULL;
inline constexpr uint64_t resource_query_version    = 2;
inline constexpr size_t resource_query_header_words = 2;

// Own a validated implementation and the include paths used to compile it.
// An empty request denotes an unbound or failed call. No compiler-owned string
// views or operand pointers survive in this record.
struct CallMetadata
{
  std::optional<ImplementationSpec> request;
  std::vector<std::string> include_paths;
};

// Describe the generated wrapper's complete shared-memory allocation.
// TempStorage is CUB's own storage; extra_static_smem_bytes accounts for wrapper
// fields. The advertised size is their enclosing ScratchStorage, including
// padding, and its alignment is measured separately. modeled is true only
// after the compiled query results have passed validation.
struct PrimitiveResourceEstimate
{
  bool modeled                     = false;
  uint64_t temp_storage_bytes      = 0;
  uint64_t extra_static_smem_bytes = 0;
  uint64_t static_smem_bytes       = 0;
  uint64_t scratch_alignment_bytes = 0;
  std::string detail;
};

// Keep caches for one provider instance, protected by cache_mutex.
// Readers copy values while holding the lock, then release it before source
// generation or NVRTC compilation. Concurrent misses may duplicate compilation;
// these maps do not serialize an entire build. seen_artifact_keys records trace
// history only, whereas the other maps retain estimates and owned LTO-IR bytes.
struct RuntimeState
{
  std::mutex cache_mutex;
  std::unordered_set<std::string> seen_artifact_keys;
  std::unordered_map<std::string, std::vector<char>> bundle_ltoir_cache;
  std::unordered_map<std::string, PrimitiveResourceEstimate> resource_estimate_cache;
};

using ResourceSourceBuilder = bool (*)(const CallMetadata&, std::string*, std::string*);

[[nodiscard]] uint64_t fnv1a_hash(const std::string& value) noexcept;
[[nodiscard]] std::string hex_u64(uint64_t value);

[[nodiscard]] std::vector<std::string> split_path_list(const std::string& value);
[[nodiscard]] std::vector<std::string> split_encoded_path_list(const std::string& value);
[[nodiscard]] std::vector<std::string> resolve_provider_include_paths();
[[nodiscard]] bool validate_provider_configuration(std::string* out_error);
void append_unique_existing_include_path(
  const std::string& path, std::vector<std::string>* out, std::unordered_set<std::string>* seen);
[[nodiscard]] bool
include_paths_contain_file(const std::vector<std::string>& include_paths, const std::string& relative_path);

[[nodiscard]] size_t seen_artifact_cache_limit();
[[nodiscard]] size_t bundle_cache_limit();
[[nodiscard]] size_t resource_cache_limit();
[[nodiscard]] TargetArchitecture parse_sm_arch(TileIRStringView architecture);
[[nodiscard]] int64_t parse_cache_epoch_from_env();
[[nodiscard]] std::string make_codegen_artifact_cache_key(const CallMetadata& meta, const std::string& callable_symbol);
[[nodiscard]] std::string ordered_include_paths_hash_key(const std::vector<std::string>& include_paths);
[[nodiscard]] std::string make_bundle_cache_key(
  TargetArchitecture target_arch,
  int64_t target_epoch,
  const std::vector<std::string>& artifact_keys,
  const std::vector<std::string>& include_paths);

[[nodiscard]] bool trace_enabled();
void trace_const_params(const TileIRResolvedCall& call, size_t call_index);
void emit_trace_line(const std::string& line);
void trace_artifact_event(const char* event, const CallMetadata& meta, const std::string& cache_key);
void trace_bundle_event(const char* event, const std::string& bundle_key, size_t primitive_count);
void trace_resource_event(
  const char* event, const CallMetadata& meta, const PrimitiveResourceEstimate& estimate, uint64_t limit_bytes);
void trace_resource_query_batch_event(const char* event, TargetArchitecture target_arch, size_t primitive_count);
void record_nvrtc_compile_counter();

[[nodiscard]] bool compile_source_to_ltoir(
  const std::string& source,
  const std::string& entry_name,
  TargetArchitecture target_arch,
  int block_threads,
  int items_per_thread,
  const std::vector<std::string>& include_paths,
  std::vector<char>* out_ltoir,
  std::string* out_error = nullptr);
[[nodiscard]] bool compile_resource_query_to_sizes(
  const std::string& source,
  TargetArchitecture target_arch,
  const std::vector<std::string>& include_paths,
  size_t expected_count,
  std::vector<uint64_t>* out_sizes,
  std::string* out_error = nullptr);

[[nodiscard]] bool specialize_source_for_bundle(std::string* source, const CallMetadata& meta, size_t fragment_index);
void split_source_includes_and_body(
  const std::string& source, std::vector<std::string>* out_includes, std::string* out_body);
[[nodiscard]] std::vector<std::string> collect_bundle_include_paths(const std::vector<CallMetadata>& metas);

[[nodiscard]] bool estimate_primitive_resources(
  RuntimeState* state,
  const CallMetadata& meta,
  TargetArchitecture target_arch,
  ResourceSourceBuilder build_source,
  PrimitiveResourceEstimate* out_estimate);
[[nodiscard]] bool estimate_primitive_resources_batch(
  RuntimeState* state,
  const std::vector<CallMetadata>& metas,
  TargetArchitecture target_arch,
  ResourceSourceBuilder build_source,
  std::vector<PrimitiveResourceEstimate>* out_estimates,
  std::string* out_error = nullptr);

// Clear the complete cache before insertion when its nonzero limit is reached.
// The caller must hold cache_mutex. A zero limit means unbounded retention,
// not disabled caching; this is a count bound rather than an LRU policy.
template <typename Cache>
void clear_cache_if_at_limit(Cache* cache, size_t limit)
{
  if (cache && limit != 0 && cache->size() >= limit)
  {
    cache->clear();
  }
}
} // namespace cuda::coop::tile::detail::provider

#endif // CUDA_COOP_TILE_PROVIDER_RUNTIME_H
