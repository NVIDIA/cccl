// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "provider/runtime/runtime.h"
#include "provider/spec_identity.h"

namespace cuda::coop::tile::detail::provider
{
namespace
{
// Own one uncached query and the call positions sharing its exact resource key.
// Aliases avoid recompiling the same specialization within a probe batch.
struct PendingResourceProbe
{
  size_t meta_index = 0;
  std::string resource_key;
  std::string source;
  std::string detail;
  std::vector<size_t> aliases;
};

[[nodiscard]] std::string resource_cache_key(const CallMetadata& meta)
{
  if (!meta.request)
  {
    return {};
  }
  const ImplementationSpec& request = *meta.request;
  return "kind=resource|artifact=" + artifact_identity(request)
       + "|epoch=" + std::to_string(request.target().cache_epoch)
       + "|include_hash=" + ordered_include_paths_hash_key(meta.include_paths);
}

[[nodiscard]] bool validate_resource_sizes(
  const CallMetadata& meta,
  const std::vector<uint64_t>& sizes,
  size_t offset,
  const std::string& detail,
  PrimitiveResourceEstimate* out_estimate)
{
  if (!meta.request || !out_estimate || offset + 3 > sizes.size())
  {
    return false;
  }

  const ImplementationSpec& request = *meta.request;
  const std::string identity        = semantic_identity(request);
  const std::string payload         = request_payload(request.primitive());
  const uint64_t scratch_bytes      = sizes[offset];
  const uint64_t scratch_alignment  = sizes[offset + 1];
  const uint64_t temp_bytes         = sizes[offset + 2];
  if (scratch_bytes == 0 || temp_bytes == 0 || temp_bytes > scratch_bytes || scratch_alignment == 0
      || (scratch_alignment & (scratch_alignment - 1)) != 0)
  {
    std::ostringstream trace;
    trace << "[coop_linkage_provider][trace] invalid CUB scratch layout size=" << scratch_bytes
          << " alignment=" << scratch_alignment << " temp_storage=" << temp_bytes << " semantic_key=" << identity
          << " payload=" << payload;
    emit_trace_line(trace.str());
    return false;
  }

  PrimitiveResourceEstimate estimate;
  estimate.modeled                 = true;
  estimate.temp_storage_bytes      = temp_bytes;
  estimate.extra_static_smem_bytes = scratch_bytes - temp_bytes;
  estimate.static_smem_bytes       = scratch_bytes;
  estimate.scratch_alignment_bytes = scratch_alignment;
  estimate.detail                  = detail;
  *out_estimate                    = std::move(estimate);
  return true;
}

[[nodiscard]] bool build_batched_resource_query(
  const std::vector<CallMetadata>& metas,
  const std::vector<PendingResourceProbe>& pending,
  const std::vector<size_t>& probe_indices,
  std::string* out_source)
{
  if (!out_source || probe_indices.empty())
  {
    return false;
  }

  std::unordered_set<std::string> include_lines_seen;
  std::string query_includes;
  std::string query_fragments;
  std::ostringstream query_values;
  for (size_t fragment_index = 0; fragment_index < probe_indices.size(); ++fragment_index)
  {
    const PendingResourceProbe& probe = pending[probe_indices[fragment_index]];
    const CallMetadata& meta          = metas[probe.meta_index];
    std::string source                = probe.source;
    if (!specialize_source_for_bundle(&source, meta, fragment_index))
    {
      return false;
    }

    std::vector<std::string> source_includes;
    std::string source_body;
    split_source_includes_and_body(source, &source_includes, &source_body);
    if (source_body.empty())
    {
      emit_trace_line("[coop_linkage_provider][trace] generated empty resource-query source body for semantic_key="
                      + semantic_identity(*meta.request));
      return false;
    }
    for (const std::string& include_line : source_includes)
    {
      if (!include_line.empty() && include_lines_seen.insert(include_line).second)
      {
        query_includes += include_line;
        query_includes.push_back('\n');
      }
    }

    const std::string scope_name = "cuda_coop_tile_resource_scope_" + std::to_string(fragment_index);
    query_fragments += "\n// ---- coop resource fragment " + std::to_string(fragment_index) + " ----\n";
    query_fragments += "namespace " + scope_name + " {\n";
    query_fragments += source_body;
    if (!query_fragments.empty() && query_fragments.back() != '\n')
    {
      query_fragments.push_back('\n');
    }
    query_fragments += "} // namespace " + scope_name + "\n";

    query_values
      << "    static_cast<unsigned long long>(sizeof(" << scope_name << "::ScratchStorage)),\n"
      << "    static_cast<unsigned long long>(alignof(" << scope_name << "::ScratchStorage)),\n"
      << "    static_cast<unsigned long long>(sizeof(decltype(" << scope_name << "::ScratchStorage::temp_storage)))";
    if (fragment_index + 1 != probe_indices.size())
    {
      query_values << ',';
    }
    query_values << '\n';
  }

  std::ostringstream query;
  query
    << query_includes << '\n'
    << query_fragments << '\n'
    << "extern \"C\" __device__ __constant__ unsigned long long "
       "cuda_coop_tile_resource_sizes[] = {\n"
    << "    " << resource_query_sentinel << "ULL,\n"
    << "    " << resource_query_version << "ULL,\n"
    << query_values.str() << "};\n";
  *out_source = query.str();
  return true;
}
} // namespace

// Resolve exact wrapper sizes, preserving the order of the requested calls.
// Copy cache hits under the mutex, deduplicate misses, and group compilation by
// identical ordered include paths. Each query returns ScratchStorage size,
// alignment, and its CUB TempStorage size. Cache each successfully validated
// group after compilation. On failure, earlier groups and output entries may
// already be populated; callers must reject the batch rather than use a partial
// estimate vector. NVRTC runs outside the cache lock.
bool estimate_primitive_resources_batch(
  RuntimeState* state,
  const std::vector<CallMetadata>& metas,
  TargetArchitecture target_arch,
  ResourceSourceBuilder build_source,
  std::vector<PrimitiveResourceEstimate>* out_estimates,
  std::string* out_error)
{
  if (out_error)
  {
    out_error->clear();
  }
  if (!state || !target_arch.valid() || !build_source || !out_estimates)
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] invalid resource-probe input contract";
    }
    return false;
  }
  out_estimates->assign(metas.size(), PrimitiveResourceEstimate{});
  if (metas.empty())
  {
    return true;
  }

  std::vector<PendingResourceProbe> pending;
  std::unordered_map<std::string, size_t> pending_by_key;
  size_t cache_hits = 0;
  for (size_t meta_index = 0; meta_index < metas.size(); ++meta_index)
  {
    const CallMetadata& meta = metas[meta_index];
    if (!meta.request)
    {
      if (out_error)
      {
        *out_error = "[LP-PROVIDER-INTERNAL-001] resource probe received untyped metadata at call index="
                   + std::to_string(meta_index);
      }
      return false;
    }
    if (meta.request->target().sm_arch != target_arch)
    {
      const std::string detail = "[LP-TARGET-001] resource batch target mismatch: batch=" + target_arch.sm_name()
                               + " call=" + meta.request->target().sm_arch.sm_name();
      emit_trace_line("[coop_linkage_provider][trace] " + detail);
      if (out_error)
      {
        *out_error = detail;
      }
      return false;
    }
    const std::string resource_key = resource_cache_key(meta);
    bool cache_hit                 = false;
    PrimitiveResourceEstimate cached_estimate;
    {
      const std::lock_guard<std::mutex> lock(state->cache_mutex);
      if (const auto it = state->resource_estimate_cache.find(resource_key); it != state->resource_estimate_cache.end())
      {
        cached_estimate = it->second;
        cache_hit       = true;
      }
    }
    if (cache_hit)
    {
      (*out_estimates)[meta_index] = std::move(cached_estimate);
      ++cache_hits;
      continue;
    }

    if (const auto duplicate = pending_by_key.find(resource_key); duplicate != pending_by_key.end())
    {
      pending[duplicate->second].aliases.push_back(meta_index);
      continue;
    }

    PendingResourceProbe probe;
    probe.meta_index   = meta_index;
    probe.resource_key = resource_key;
    if (!build_source(meta, &probe.source, &probe.detail))
    {
      if (out_error)
      {
        *out_error =
          "[LP-PROVIDER-INTERNAL-001] failed to build resource source for " + semantic_identity(*meta.request);
      }
      return false;
    }
    if (probe.source.empty())
    {
      if (out_error)
      {
        *out_error = "[LP-PROVIDER-INTERNAL-001] resource source builder returned empty source for "
                   + semantic_identity(*meta.request);
      }
      return false;
    }
    const size_t pending_index = pending.size();
    pending_by_key.emplace(resource_key, pending_index);
    pending.push_back(std::move(probe));
  }
  if (cache_hits != 0)
  {
    trace_resource_query_batch_event("resource_query_cache_hit", target_arch, cache_hits);
  }

  std::map<std::vector<std::string>, std::vector<size_t>> groups;
  for (size_t pending_index = 0; pending_index < pending.size(); ++pending_index)
  {
    groups[metas[pending[pending_index].meta_index].include_paths].push_back(pending_index);
  }

  for (const auto& [include_paths, probe_indices] : groups)
  {
    std::string query_source;
    if (!build_batched_resource_query(metas, pending, probe_indices, &query_source))
    {
      if (out_error)
      {
        *out_error = "[LP-PROVIDER-INTERNAL-001] failed to assemble resource query for " + target_arch.sm_name();
      }
      return false;
    }

    trace_resource_query_batch_event("resource_query_nvrtc_compile", target_arch, probe_indices.size());
    std::vector<uint64_t> sizes;
    const size_t expected_count = probe_indices.size() * 3;
    if (!compile_resource_query_to_sizes(query_source, target_arch, include_paths, expected_count, &sizes, out_error)
        || sizes.size() != expected_count)
    {
      if (out_error && out_error->empty())
      {
        *out_error = "[LP-PROVIDER-INTERNAL-001] resource query returned an unexpected number of size values";
      }
      for (const size_t probe_index : probe_indices)
      {
        const CallMetadata& meta          = metas[pending[probe_index].meta_index];
        const ImplementationSpec& request = *meta.request;
        emit_trace_line("[coop_linkage_provider][trace] failed to query exact CUB scratch layout for semantic_key="
                        + semantic_identity(request) + " payload=" + request_payload(request.primitive()));
      }
      return false;
    }

    std::vector<std::pair<std::string, PrimitiveResourceEstimate>> cache_entries;
    cache_entries.reserve(probe_indices.size());
    for (size_t group_index = 0; group_index < probe_indices.size(); ++group_index)
    {
      const PendingResourceProbe& probe = pending[probe_indices[group_index]];
      const CallMetadata& meta          = metas[probe.meta_index];
      PrimitiveResourceEstimate estimate;
      if (!validate_resource_sizes(meta, sizes, group_index * 3, probe.detail, &estimate))
      {
        if (out_error)
        {
          *out_error = "[LP-SCRATCH-001] invalid resource layout for " + semantic_identity(*meta.request);
        }
        return false;
      }
      (*out_estimates)[probe.meta_index] = estimate;
      for (const size_t alias : probe.aliases)
      {
        (*out_estimates)[alias] = estimate;
      }
      cache_entries.emplace_back(probe.resource_key, std::move(estimate));
    }

    {
      const std::lock_guard<std::mutex> lock(state->cache_mutex);
      for (auto& [resource_key, estimate] : cache_entries)
      {
        clear_cache_if_at_limit(&state->resource_estimate_cache, resource_cache_limit());
        state->resource_estimate_cache.insert_or_assign(resource_key, std::move(estimate));
      }
    }
  }
  return true;
}

bool estimate_primitive_resources(
  RuntimeState* state,
  const CallMetadata& meta,
  TargetArchitecture target_arch,
  ResourceSourceBuilder build_source,
  PrimitiveResourceEstimate* out_estimate)
{
  if (!out_estimate)
  {
    return false;
  }
  std::vector<PrimitiveResourceEstimate> estimates;
  if (!estimate_primitive_resources_batch(state, {meta}, target_arch, build_source, &estimates)
      || estimates.size() != 1)
  {
    return false;
  }
  *out_estimate = std::move(estimates.front());
  return true;
}
} // namespace cuda::coop::tile::detail::provider
