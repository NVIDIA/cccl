// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/spec_identity.h"

#include <sstream>
#include <type_traits>

#include "provider/registry.h"

namespace cuda::coop::tile::detail::provider
{
// This file converts a validated, typed ImplementationSpec into stable identity
// fragments. Those fragments let discovery candidates, resource probes,
// traces, and caches agree that they are talking about the same specialization
// without comparing frontend Tile IR directly. The compiler-selected
// implementation symbol is a separate output identity and is never invented
// here.
//
// These strings are internal key material, not a user-facing serialization.
// Field names, ordering, and omission rules must nevertheless remain
// deterministic: changing them deliberately invalidates process-local artifact
// identity and can stop otherwise-compatible declarations from sharing work.
namespace
{
template <class... Ts>
struct overloaded : Ts...
{
  using Ts::operator()...;
};

template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

int bool_value(bool value)
{
  return value ? 1 : 0;
}

void append_topology_prefix(std::ostringstream& out, const Topology& topology)
{
  out << "block_threads=" << topology.block_threads;
}

void append_items_per_thread(std::ostringstream& out, const Topology& topology)
{
  out << ";items_per_thread=" << topology.items_per_thread;
}

// Runtime operands are intentionally absent from ImplementationSpec, so this
// metadata contains exactly the values that select primitive semantics or
// change generated source.
std::string metadata_impl(const ImplementationSpec& request)
{
  return std::visit(
    overloaded{
      [&](const ReduceSpec& spec) {
        std::ostringstream out;
        out << "algorithm=" << spec.algorithm << ";";
        if (spec.generic)
        {
          out << "binary_op=" << spec.binary_op << ";";
        }
        out << "block_threads=" << request.topology().block_threads << ";dtype=" << spec.dtype;
        append_items_per_thread(out, request.topology());
        if (spec.batches > 0)
        {
          out << ";batches=" << spec.batches << ";extent=" << request.demand().extent();
          if (request.topology().reduction_batches_per_invocation > 1)
          {
            out << ";batches_per_invocation=" << request.topology().reduction_batches_per_invocation;
          }
        }
        return out.str();
      },
      [&](const ScanSpec& spec) {
        std::ostringstream out;
        out << "algorithm=" << spec.algorithm << ";block_threads=" << request.topology().block_threads
            << ";dtype=" << spec.dtype << ";initial_present=" << bool_value(spec.initial_present)
            << ";prefix_present=" << bool_value(spec.prefix_present);
        append_items_per_thread(out, request.topology());
        out << ";mode=" << spec.mode << ";scan_op=" << spec.scan_op;
        return out.str();
      },
      [&](const HistogramSpec& spec) {
        std::ostringstream out;
        out << "bins=" << spec.bins << ";block_threads=" << request.topology().block_threads << ";dtype=" << spec.dtype;
        append_items_per_thread(out, request.topology());
        out << ";output_dtype=i32";
        return out.str();
      },
      [&](const RunLengthDecodeSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";decoded_items_per_thread=" << request.topology().decoded_items_per_thread
            << ";length_dtype=" << spec.length_dtype << ";runs_per_thread=" << request.topology().items_per_thread
            << ";value_dtype=" << spec.value_dtype;
        return out.str();
      },
      [&](const DiscontinuitySpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";dtype=" << spec.dtype;
        append_items_per_thread(out, request.topology());
        out << ";mode=" << spec.mode;
        return out.str();
      },
      [&](const AdjacentDifferenceSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";dtype=" << spec.dtype;
        append_items_per_thread(out, request.topology());
        out << ";mode=" << spec.mode;
        return out.str();
      },
      [&](const ExchangeSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";dtype=" << spec.dtype;
        append_items_per_thread(out, request.topology());
        out << ";mode=" << spec.mode;
        return out.str();
      },
      [&](const ShuffleSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";dtype=" << spec.dtype;
        append_items_per_thread(out, request.topology());
        out << ";mode=" << spec.mode;
        return out.str();
      },
      [&](const RadixRankSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";descending=" << bool_value(spec.descending) << ";dtype=" << spec.dtype
            << ";inner_scan_algorithm=" << spec.inner_scan_algorithm;
        append_items_per_thread(out, request.topology());
        out << ";memoize_outer_scan=" << bool_value(spec.memoize_outer_scan) << ";radix_bits=" << spec.radix_bits
            << ";smem_config=" << spec.smem_config;
        return out.str();
      },
      [&](const MergeSortSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";descending=" << bool_value(spec.descending);
        if (spec.pairs)
        {
          append_items_per_thread(out, request.topology());
          out << ";key_dtype=" << spec.key_dtype << ";partial_tile=" << bool_value(spec.partial_tile)
              << ";value_dtype=" << spec.value_dtype;
        }
        else
        {
          out << ";dtype=" << spec.key_dtype;
          append_items_per_thread(out, request.topology());
          out << ";partial_tile=" << bool_value(spec.partial_tile);
        }
        return out.str();
      },
      [&](const RadixSortSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";descending=" << bool_value(spec.descending);
        if (spec.pairs)
        {
          out << ";inner_scan_algorithm=" << spec.inner_scan_algorithm;
          append_items_per_thread(out, request.topology());
          out << ";key_dtype=" << spec.key_dtype << ";memoize_outer_scan=" << bool_value(spec.memoize_outer_scan)
              << ";radix_bits=" << spec.radix_bits << ";smem_config=" << spec.smem_config
              << ";value_dtype=" << spec.value_dtype;
        }
        else
        {
          out << ";dtype=" << spec.key_dtype << ";inner_scan_algorithm=" << spec.inner_scan_algorithm;
          append_items_per_thread(out, request.topology());
          out << ";memoize_outer_scan=" << bool_value(spec.memoize_outer_scan) << ";radix_bits=" << spec.radix_bits
              << ";smem_config=" << spec.smem_config;
        }
        return out.str();
      },
      [&](const TopKSpec& spec) {
        std::ostringstream out;
        append_topology_prefix(out, request.topology());
        out << ";descending=" << bool_value(spec.descending);
        if (spec.pairs)
        {
          out << ";full_tile=" << bool_value(spec.full_tile);
          append_items_per_thread(out, request.topology());
          out << ";key_dtype=" << spec.key_dtype << ";value_dtype=" << spec.value_dtype;
        }
        else
        {
          out << ";dtype=" << spec.key_dtype << ";full_tile=" << bool_value(spec.full_tile);
          append_items_per_thread(out, request.topology());
        }
        return out.str();
      }},
    request.primitive());
}
} // namespace

// Used by cache keys to distinguish any primitive metadata that changes the
// generated source body. The runtime adds target architecture, an explicit
// cache epoch, and ordered include paths where needed. It does not fingerprint
// compiler binaries or header contents at these paths.
std::string codegen_metadata(const ImplementationSpec& request)
{
  return metadata_impl(request);
}

// Used for tracing, resource lookup, and semantic deduplication. Runtime
// operands are absent because their values do not change the primitive being
// requested.
std::string semantic_metadata(const ImplementationSpec& request)
{
  return metadata_impl(request);
}

// Anchor semantic identity on the one registry-owned declaration for the
// family. The declaration name is frontend-neutral and therefore is not a
// compatibility dimension.
std::string semantic_identity(const ImplementationSpec& request)
{
  const FamilyDescriptor& descriptor = family_descriptor(family_id(request.primitive()));
  return "coop::" + std::string(descriptor.decl_symbol) + "|" + semantic_metadata(request);
}

// Implementation/artifact identity extends semantic identity with the exact
// target family. This prevents sm_90 and sm_90a (or sm_100 and sm_100f) from
// sharing an implementation symbol even when their primitive semantics match.
// Runtime cache layers add the explicit cache epoch and include-path identity.
std::string artifact_identity(const ImplementationSpec& request)
{
  return artifact_identity(request, request.target().sm_arch);
}

std::string artifact_identity(const ImplementationSpec& request, TargetArchitecture architecture)
{
  const std::string target = architecture.sm_name();
  return semantic_identity(request) + "|tile_interface_abi=2|target=" + (target.empty() ? "<invalid>" : target);
}

// Payload selects the family source/resource template. Scan is the only family
// whose generic and fixed-sum forms share a FamilyId but require distinct
// payloads.
std::string request_payload(const PrimitiveSpec& primitive)
{
  if (const auto* scan = std::get_if<ScanSpec>(&primitive))
  {
    return scan->generic ? "scan_generic" : "scan_sum";
  }
  return request_payload(family_id(primitive));
}
} // namespace cuda::coop::tile::detail::provider
