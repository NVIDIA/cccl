// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Serve generated cuda.coop.tile implementations to TileIRAS through the
// LinkageProvider C ABI.
//
// LPv4 discovery consumes structured callsite records and returns an
// implementation-only catalog. Every admitted specialization carries a typed
// ImplementationSpec. Resource queries derive scratch from the exact generated
// wrapper. Instantiation emits each authenticated compiler-selected
// implementation symbol verbatim, and all selected specializations are
// compiled in one batched NVRTC LTO-IR module.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

#include "linkage_providers/ImplCatalogBuilder.h"
#include "linkage_providers/TileLinkageProvider.h"
#include "provider/families/emitter.h"
#include "provider/metadata_parser.h"
#include "provider/registry.h"
#include "provider/runtime/runtime.h"
#include "provider/source_common.h"
#include "provider/spec_identity.h"

namespace
{
namespace lp = cutile::linkage_provider;

using cuda::coop::tile::detail::provider::argument_is_scalar_by_value;
using cuda::coop::tile::detail::provider::artifact_identity;
using cuda::coop::tile::detail::provider::bind_call_request;
using cuda::coop::tile::detail::provider::bundle_cache_limit;
using cuda::coop::tile::detail::provider::CallMetadata;
using cuda::coop::tile::detail::provider::clear_cache_if_at_limit;
using cuda::coop::tile::detail::provider::collect_bundle_include_paths;
using cuda::coop::tile::detail::provider::compile_source_to_ltoir;
using cuda::coop::tile::detail::provider::declaration_symbol_matches_family;
using cuda::coop::tile::detail::provider::emit_family_source;
using cuda::coop::tile::detail::provider::emit_trace_line;
using cuda::coop::tile::detail::provider::estimate_primitive_resources_batch;
using cuda::coop::tile::detail::provider::family_candidate_preference_score;
using cuda::coop::tile::detail::provider::family_name;
using cuda::coop::tile::detail::provider::FamilyDescriptor;
using cuda::coop::tile::detail::provider::FamilyId;
using cuda::coop::tile::detail::provider::find_family_descriptor;
using cuda::coop::tile::detail::provider::find_family_descriptor_for_implementation;
using cuda::coop::tile::detail::provider::fnv1a_hash;
using cuda::coop::tile::detail::provider::hex_u64;
using cuda::coop::tile::detail::provider::ImplementationSpec;
using cuda::coop::tile::detail::provider::include_paths_contain_file;
using cuda::coop::tile::detail::provider::is_source_symbol_identifier;
using cuda::coop::tile::detail::provider::LogicalRequest;
using cuda::coop::tile::detail::provider::make_bundle_cache_key;
using cuda::coop::tile::detail::provider::make_codegen_artifact_cache_key;
using cuda::coop::tile::detail::provider::materialize_call_request;
using cuda::coop::tile::detail::provider::MetadataResult;
using cuda::coop::tile::detail::provider::MetadataStatus;
using cuda::coop::tile::detail::provider::parse_call_metadata;
using cuda::coop::tile::detail::provider::parse_sm_arch;
using cuda::coop::tile::detail::provider::PrimitiveResourceEstimate;
using cuda::coop::tile::detail::provider::record_nvrtc_compile_counter;
using cuda::coop::tile::detail::provider::ReduceSpec;
using cuda::coop::tile::detail::provider::replace_all;
using cuda::coop::tile::detail::provider::request_payload;
using cuda::coop::tile::detail::provider::required_header;
using cuda::coop::tile::detail::provider::resolve_provider_include_paths;
using cuda::coop::tile::detail::provider::resolve_request_target;
using cuda::coop::tile::detail::provider::RuntimeState;
using cuda::coop::tile::detail::provider::seen_artifact_cache_limit;
using cuda::coop::tile::detail::provider::semantic_identity;
using cuda::coop::tile::detail::provider::SourceEmissionRequest;
using cuda::coop::tile::detail::provider::SourceEmissionStatus;
using cuda::coop::tile::detail::provider::specialize_source_for_bundle;
using cuda::coop::tile::detail::provider::split_source_includes_and_body;
using cuda::coop::tile::detail::provider::symbol_is_concrete_specialization_of;
using cuda::coop::tile::detail::provider::TargetArchitecture;
using cuda::coop::tile::detail::provider::TargetArchitectureFamily;
using cuda::coop::tile::detail::provider::trace_artifact_event;
using cuda::coop::tile::detail::provider::trace_bundle_event;
using cuda::coop::tile::detail::provider::trace_const_params;
using cuda::coop::tile::detail::provider::trace_enabled;
using cuda::coop::tile::detail::provider::trace_resource_event;
using cuda::coop::tile::detail::provider::uses_warp_execution_scope;
using cuda::coop::tile::detail::provider::validate_provider_configuration;

#if TILEIR_LINKAGE_PROVIDER_API_VERSION_MAJOR != 0u || TILEIR_LINKAGE_PROVIDER_API_VERSION_MINOR != 0u \
  || TILEIR_LINKAGE_PROVIDER_API_VERSION_PATCH != 1u
#  error "cuda.coop.tile requires prerelease linkage-provider API 0.0.1"
#endif

constexpr char kProviderName[]                    = "cuda-coop-tile";
constexpr char kCodeBufferName[]                  = "cuda-coop-tile.ltoir";
constexpr int64_t kAutoTopologyCandidateThreads[] = {32, 64, 128, 256, 512, 1024};

std::string to_string(TileIRStringView sv)
{
  if (!sv.data || sv.len == 0)
  {
    return {};
  }
  return std::string(sv.data, sv.len);
}

std::string to_string(TileIRDataType dtype)
{
  switch (dtype.kind)
  {
    case TILEIR_DTYPE_I1:
      return "i1";
    case TILEIR_DTYPE_I8:
      return "i8";
    case TILEIR_DTYPE_I16:
      return "i16";
    case TILEIR_DTYPE_I32:
      return "i32";
    case TILEIR_DTYPE_I64:
      return "i64";
    case TILEIR_DTYPE_F16:
      return "f16";
    case TILEIR_DTYPE_BF16:
      return "bf16";
    case TILEIR_DTYPE_F32:
      return "f32";
    case TILEIR_DTYPE_TF32:
      return "tf32";
    case TILEIR_DTYPE_F64:
      return "f64";
    case TILEIR_DTYPE_F8E4M3FN:
      return "f8e4m3fn";
    case TILEIR_DTYPE_F8E5M2:
      return "f8e5m2";
    case TILEIR_DTYPE_F8E8M0FNU:
      return "f8e8m0fnu";
    case TILEIR_DTYPE_F4E2M1FN:
      return "f4e2m1fn";
    case TILEIR_DTYPE_POINTER:
      return "ptr";
    case TILEIR_DTYPE_INVALID:
    default:
      return "";
  }
}

// Own cache storage for the SDK-managed lifetime of one provider instance.
// Callback requests borrow compiler records. Catalogs and returned modules use
// SDK builders to own their strings and byte buffers across the C ABI.
struct CoopProviderState final : lp::TileLinkageProvider<CoopProviderState>
{
  RuntimeState runtime;

  lp::Result<lp::ImplCatalog> provideImpls(const lp::ProvideImplsRequest& request);
  lp::Result<void> filterAndRank(lp::FilterAndRankBatch& batch);
  lp::Result<lp::ForeignModule> instantiate(const lp::ResolvedCallBatch& calls);
};

bool build_source_for_primitive_detailed(
  const CallMetadata& meta, const std::string& callable_symbol, std::string* out_source, std::string* out_error)
{
  if (out_error)
  {
    out_error->clear();
  }
  if (!out_source)
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] source emission received a null output";
    }
    return false;
  }
  out_source->clear();
  if (!meta.request)
  {
    if (out_error)
    {
      *out_error = "[LP-PROVIDER-INTERNAL-001] source emission requires typed provider metadata";
    }
    return false;
  }
  if (!is_source_symbol_identifier(callable_symbol))
  {
    if (out_error)
    {
      *out_error = "[LP-SYMBOL-001] generated source requires a valid callable symbol";
    }
    return false;
  }

  std::string emission_error;
  const SourceEmissionStatus emission_status =
    emit_family_source(SourceEmissionRequest{*meta.request, callable_symbol}, out_source, &emission_error);
  if (emission_status == SourceEmissionStatus::emitted)
  {
    return true;
  }
  if (out_error)
  {
    *out_error = "[LP-PROVIDER-INTERNAL-001] source emission failed for " + semantic_identity(*meta.request) + ": "
               + (emission_error.empty() ? "emitter returned no detail" : emission_error);
  }
  return false;
}

bool build_source_for_primitive(const CallMetadata& meta, const std::string& callable_symbol, std::string* out_source)
{
  return build_source_for_primitive_detailed(meta, callable_symbol, out_source, nullptr);
}

std::string resource_probe_symbol(const CallMetadata& meta)
{
  if (!meta.request)
  {
    return {};
  }
  return "cuda_coop_tile_resource_probe_" + hex_u64(fnv1a_hash(artifact_identity(*meta.request)));
}

bool build_resource_source_for_primitive(const CallMetadata& meta, std::string* out_source, std::string* out_detail)
{
  if (!out_source || !out_detail)
  {
    return false;
  }
  out_source->clear();
  out_detail->clear();

  if (!meta.request)
  {
    emit_trace_line("[coop_linkage_provider][trace][LP-SCRATCH-001] resource query requires typed metadata");
    return false;
  }

  if (!build_source_for_primitive(meta, resource_probe_symbol(meta), out_source) || out_source->empty())
  {
    return false;
  }
  *out_source = replace_all(std::move(*out_source), "%ADVERTISED_SCRATCH_BYTES%", "sizeof(ScratchStorage)");
  *out_source = replace_all(std::move(*out_source), "%ADVERTISED_SCRATCH_ALIGNMENT%", "alignof(ScratchStorage)");
  *out_detail = request_payload(meta.request->primitive());
  return true;
}

uint64_t max_static_smem_bytes_for_provider_bundle(TargetArchitecture target_arch)
{
  if (const char* env = std::getenv("CUDA_COOP_TILE_PROVIDER_MAX_STATIC_SMEM_BYTES"); env && env[0] != '\0')
  {
    char* end                       = nullptr;
    const unsigned long long parsed = std::strtoull(env, &end, 10);
    if (end && *end == '\0' && parsed > 0)
    {
      return static_cast<uint64_t>(parsed);
    }
  }
  (void) target_arch;
  // The compiler owns the advertised scratch allocation. The provider
  // applies this discovery and bundle precheck only for an explicit debug limit.
  return 0;
}

const char* provider_family_name(FamilyId family)
{
  return family_name(family);
}

const char* provider_family_required_header(FamilyId family)
{
  return required_header(family);
}

bool provider_include_paths_have_required_headers(const std::vector<std::string>& include_paths, FamilyId family)
{
  const char* family_header = provider_family_required_header(family);
  return family_header && include_paths_contain_file(include_paths, family_header)
      && include_paths_contain_file(include_paths, "cuda_runtime.h");
}

constexpr char kTileIRFloat16Type[]       = "f16";
constexpr char kTileIRBFloat16Type[]      = "bf16";
constexpr char kFrontendHalfWrapperType[] = "!llvm.struct<(i16)>";

bool string_view_is_valid(TileIRStringView value)
{
  return value.len == 0 || value.data != nullptr;
}

bool impl_symbol_is_specialization_of(const std::string& value, const std::string& base)
{
  // Fixed wildcard catalogs keep the provider's target-bearing base symbol.
  // Parametric rows append either concrete bindings (`__...`) or a SymbolTable
  // decimal uniquifier (`_0`, `_1`, ...). The selected symbol is interpolated
  // into generated C++, so reject every non-identifier spelling here as well.
  return is_source_symbol_identifier(value)
      && (value == base || symbol_is_concrete_specialization_of(value.data(), value.size(), base.c_str()));
}

const char* metadata_status_name(MetadataStatus status)
{
  switch (status)
  {
    case MetadataStatus::supported:
      return "supported";
    case MetadataStatus::not_owned:
      return "not_owned";
    case MetadataStatus::unsupported:
      return "unsupported";
    case MetadataStatus::malformed:
      return "malformed";
    case MetadataStatus::internal_error:
      return "internal_error";
  }
  return "unknown";
}

void trace_metadata_result(const char* phase, const std::string& symbol, size_t index, const MetadataResult& result)
{
  if (!trace_enabled())
  {
    return;
  }
  std::ostringstream trace;
  trace << "[coop_linkage_provider][trace] " << phase << " index=" << index << " symbol=" << symbol
        << " status=" << metadata_status_name(result.status);
  if (!result.diagnostic.empty())
  {
    trace << " reason=" << result.diagnostic;
  }
  emit_trace_line(trace.str());
}

const FamilyDescriptor* find_family_for_impl_symbol(TileIRStringView symbol)
{
  return find_family_descriptor_for_implementation(symbol.data, symbol.len);
}

MetadataResult parse_impl_target_constraint(const TileIRResolvedCall& call, TargetArchitecture* out)
{
  if (!out)
  {
    return MetadataResult::internal_error("[LP-PROVIDER-INTERNAL-001] implementation target validation has no output");
  }
  *out = TargetArchitecture{};
  if (!call.impl)
  {
    return MetadataResult::internal_error(
      "[LP-PROVIDER-INTERNAL-001] implementation target validation has no "
      "implementation");
  }
  const TileIRStringView constraint = call.impl->execution_context_constraints.architecture;
  // TileIR consumes catalog architecture restrictions during hard
  // compatibility and does not currently preserve them on the concrete
  // implementation sent to late callbacks. An empty constraint is therefore
  // unspecified, not corrupted. The resolved target is still authenticated by
  // the provider's target-bearing implementation hash below.
  if (constraint.len == 0)
  {
    return MetadataResult::success();
  }
  if (!constraint.data)
  {
    return MetadataResult::malformed("[LP-TARGET-001] provider implementation has an invalid architecture view");
  }
  *out = parse_sm_arch(constraint);
  if (!out->valid())
  {
    return MetadataResult::malformed("[LP-TARGET-001] provider implementation has a malformed architecture constraint");
  }
  return MetadataResult::success();
}

MetadataResult compare_impl_target_constraint(TargetArchitecture impl_target, const CallMetadata& meta)
{
  if (!meta.request)
  {
    return MetadataResult::internal_error(
      "[LP-PROVIDER-INTERNAL-001] implementation target comparison is missing "
      "validated identity");
  }
  if (!impl_target.valid())
  {
    return MetadataResult::success();
  }
  const TargetArchitecture resolved_target = meta.request->target().sm_arch;
  if (impl_target != resolved_target)
  {
    return MetadataResult::unsupported("[LP-TARGET-001] implementation target " + impl_target.sm_name()
                                       + " does not match resolved target " + resolved_target.sm_name());
  }
  return MetadataResult::success();
}

MetadataResult
compare_impl_execution_context(const TileIRExecutionContext& implementation, const TileIRExecutionContext& resolved)
{
  if (implementation.scope != resolved.scope)
  {
    return MetadataResult::unsupported(
      "[LP-TOPOLOGY-001] implementation execution scope does not match resolved "
      "scope");
  }
  if (implementation.num_warps_per_cta > 0 && implementation.num_warps_per_cta != resolved.num_warps_per_cta)
  {
    return MetadataResult::unsupported(
      "[LP-TOPOLOGY-001] implementation warp count does not match resolved warp "
      "count");
  }
  return MetadataResult::success();
}

const FamilyDescriptor* find_family_for_symbol(TileIRStringView decl_symbol)
{
  return find_family_descriptor(decl_symbol.data, decl_symbol.len);
}

std::string make_impl_symbol(const FamilyDescriptor& entry, const CallMetadata& meta)
{
  if (!meta.request)
  {
    return {};
  }
  return std::string(entry.impl_symbol) + "__lpv4_" + hex_u64(fnv1a_hash(artifact_identity(*meta.request)));
}

bool impl_symbol_matches_admitted_metadata(
  const std::string& value, const FamilyDescriptor& entry, const CallMetadata& meta, std::string* expected)
{
  const std::string exact = make_impl_symbol(entry, meta);
  if (impl_symbol_is_specialization_of(value, exact))
  {
    if (expected)
    {
      *expected = exact;
    }
    return true;
  }

  // NVCC can discover an implementation for a generic target (for example,
  // sm_120) and refine the resolved execution context to the same-code family
  // target (sm_120a) before late callbacks. Authenticate that one-way
  // refinement against the exact discovery identity while retaining the
  // resolved family target for resource compilation and artifact caches.
  if (meta.request && meta.request->target().sm_arch.family != TargetArchitectureFamily::none)
  {
    const TargetArchitecture discovery_target{meta.request->target().sm_arch.code, TargetArchitectureFamily::none};
    const std::string discovery = std::string(entry.impl_symbol) + "__lpv4_"
                                + hex_u64(fnv1a_hash(artifact_identity(*meta.request, discovery_target)));
    if (impl_symbol_is_specialization_of(value, discovery))
    {
      if (expected)
      {
        *expected = discovery;
      }
      return true;
    }
    if (expected)
    {
      *expected = exact + " or refined-from " + discovery;
    }
    return false;
  }

  if (expected)
  {
    *expected = exact;
  }
  return false;
}

int64_t foreign_type_numel(const TileIRForeignType& type)
{
  if (type.kind != TILEIR_TYPE_TILE)
  {
    return 0;
  }
  if (type.num_shape == 0)
  {
    return 1;
  }
  if (!type.shape)
  {
    return 0;
  }
  int64_t numel = 1;
  for (size_t i = 0; i < type.num_shape; ++i)
  {
    const TileIRDim& dim = type.shape[i];
    if (dim.kind != TILEIR_DIM_CONCRETE || dim.value <= 0 || numel > INT64_MAX / dim.value)
    {
      return 0;
    }
    numel *= dim.value;
  }
  return numel;
}

std::optional<lp::ImplCatalogBuilder::ConstantParameterKind>
to_builder_const_param_kind(TileIRConstantParameterKind kind)
{
  using Kind = lp::ImplCatalogBuilder::ConstantParameterKind;
  switch (kind)
  {
    case TILEIR_CONSTANT_PARAMETER_INT:
      return Kind::Integer;
    case TILEIR_CONSTANT_PARAMETER_SHAPE:
      return Kind::Shape;
    case TILEIR_CONSTANT_PARAMETER_DTYPE:
      return Kind::DataType;
    case TILEIR_CONSTANT_PARAMETER_STRING:
      return Kind::String;
  }
  return std::nullopt;
}

std::optional<lp::ImplCatalogBuilder::Expression> make_concrete_catalog_argument(
  lp::ImplCatalogBuilder& builder, const TileIRConstantParameter& parameter, const TileIRSymbolicExpression& argument)
{
  switch (parameter.kind)
  {
    case TILEIR_CONSTANT_PARAMETER_INT:
      if (argument.kind == TILEIR_SYMBOLIC_EXPRESSION_INT)
      {
        return builder.integer(argument.int_val);
      }
      break;
    case TILEIR_CONSTANT_PARAMETER_DTYPE:
      if (argument.kind == TILEIR_SYMBOLIC_EXPRESSION_DTYPE)
      {
        const std::string dtype = to_string(argument.dtype_val);
        if (!dtype.empty())
        {
          return builder.type(dtype);
        }
      }
      break;
    case TILEIR_CONSTANT_PARAMETER_STRING:
      if (argument.kind == TILEIR_SYMBOLIC_EXPRESSION_STRING
          && (argument.string_val.data || argument.string_val.len == 0))
      {
        return builder.string(to_string(argument.string_val));
      }
      break;
    case TILEIR_CONSTANT_PARAMETER_SHAPE:
      break;
  }
  return std::nullopt;
}

std::optional<lp::ImplCatalogBuilder::Type> make_implementation_fragment(
  lp::ImplCatalogBuilder& builder, const TileIRForeignType& type, int64_t block_threads, bool scalar_by_value)
{
  const int64_t numel = foreign_type_numel(type);
  if (numel <= 0 || block_threads <= 0 || (!scalar_by_value && (numel % block_threads) != 0)
      || (scalar_by_value && numel != 1))
  {
    return std::nullopt;
  }
  const int64_t footprint = scalar_by_value ? 1 : numel / block_threads;
  const std::string dtype = to_string(type.dtype);
  if (dtype.empty())
  {
    return std::nullopt;
  }

  const int64_t stride = footprint == 1 ? 0 : 1;
  const auto mem_layout =
    scalar_by_value ? builder.layout(std::initializer_list<int64_t>{}, std::initializer_list<int64_t>{})
                    : builder.layout({footprint}, {stride});
  const auto fragmap =
    builder.layout({1, block_threads, footprint}, {0, scalar_by_value ? 0 : footprint, scalar_by_value ? 0 : stride});
  return builder.fragment(
    builder.tile({footprint}, builder.type(dtype)), lp::ImplCatalogBuilder::MemorySpace::Local, mem_layout, fragmap, 1);
}

// A native invocation reduces one row, or a pair of rows with WarpReduceBatched.
// Short rows repeat real inputs across participant groups; CUB's valid count or
// logical warp width excludes duplicates. Batched outputs are distributed over
// the first two lanes and replicated across the rest of the physical warp.
std::optional<lp::ImplCatalogBuilder::Type> make_reduction_fragment(
  lp::ImplCatalogBuilder& builder,
  const TileIRForeignType& type,
  const ImplementationSpec& request,
  size_t argument_index)
{
  const auto& spec            = std::get<ReduceSpec>(request.primitive());
  const int64_t threads       = request.topology().block_threads;
  const int64_t row_extent    = request.demand().extent() / spec.batches;
  const int64_t valid_threads = std::min(row_extent, threads);
  const bool input            = argument_index == 0;
  const int64_t batches       = request.topology().reduction_batches_per_invocation;
  const int64_t items         = request.topology().items_per_thread;
  const int64_t footprint     = input ? items * batches : 1;
  const std::string dtype     = to_string(type.dtype);
  if (dtype.empty() || argument_index > 1)
  {
    return std::nullopt;
  }
  const auto mem_layout = builder.layout({footprint}, {footprint == 1 ? 0 : 1});
  if (batches > 1)
  {
    const auto participant_shape =
      input ? builder.tuple({valid_threads, threads / valid_threads}) : builder.tuple({batches, threads / batches});
    const auto participant_stride = input ? builder.tuple({items, 0}) : builder.tuple({1, 0});
    const auto register_shape     = input ? builder.tuple({items, batches}) : builder.integer(1);
    const auto register_stride    = input ? builder.tuple({items > 1 ? 1 : 0, row_extent}) : builder.integer(0);
    const auto fragmap = builder.layout({spec.batches / batches, participant_shape, register_shape},
                                        {input ? row_extent * batches : batches, participant_stride, register_stride});
    return builder.fragment(
      builder.tile({footprint}, builder.type(dtype)),
      lp::ImplCatalogBuilder::MemorySpace::Local,
      mem_layout,
      fragmap,
      1);
  }
  const auto participant_shape  = builder.tuple({valid_threads, threads / valid_threads});
  const auto participant_stride = builder.tuple({input ? footprint : 0, 0});
  const auto fragmap = builder.layout({spec.batches, participant_shape, footprint},
                                      {input ? row_extent : 1, participant_stride, input && footprint > 1 ? 1 : 0});
  return builder.fragment(
    builder.tile({footprint}, builder.type(dtype)), lp::ImplCatalogBuilder::MemorySpace::Local, mem_layout, fragmap, 1);
}

lp::ImplCatalogBuilder::Access implementation_access(TileIRLPAccessKind access)
{
  using Access = lp::ImplCatalogBuilder::Access;
  switch (access)
  {
    case TILEIR_LP_ACCESS_IN:
      return Access::In;
    case TILEIR_LP_ACCESS_OUT:
      return Access::Out;
    case TILEIR_LP_ACCESS_INOUT:
      return Access::InOut;
  }
  return Access::InOut;
}

// Advertise concrete implementations for one logical callsite.
// Bind the declaration before enumerating workers, then prove each candidate's
// scratch requirement and add its preference score. Unsupported candidates can
// be omitted; malformed owned calls return a diagnostic. CUDA 13.5 selects from
// this catalog without invoking filterAndRank, so discovery must be complete.
lp::Result<lp::ImplCatalog> CoopProviderState::provideImpls(const lp::ProvideImplsRequest& request)
{
  // Request format 1 discovers one callsite at a time. Accepting a batch here
  // would make catalog identity depend on an unsupported adapter shape and can
  // hide compiler/provider lifecycle drift.
  if (request.callsites.size != 1)
  {
    return lp::InvalidArgumentError("provide_impls request format 1 requires exactly one callsite");
  }

  lp::ImplCatalogBuilder builder;
  bool added = false;
  std::unordered_set<std::string> seen_specializations;
  for (size_t callsite_index = 0; callsite_index < request.callsites.size; ++callsite_index)
  {
    const TileIRLPCallsite& callsite = request.callsites[callsite_index];
    if (!callsite.decl)
    {
      return lp::InvalidArgumentError("provide_impls callsite has no declaration");
    }
    const FamilyDescriptor* entry = find_family_for_symbol(callsite.decl->symbol);
    if (!entry)
    {
      continue;
    }
    const std::string decl_symbol = to_string(callsite.decl->symbol);
    TargetArchitecture discovery_target;
    const MetadataResult discovery_target_result =
      resolve_request_target(request.compilationContext.target_architecture, *entry, &discovery_target);
    if (!discovery_target_result.is_supported())
    {
      trace_metadata_result("discovery", decl_symbol, callsite_index, discovery_target_result);
      return lp::ProviderError("malformed cuda.coop.tile call to " + decl_symbol, discovery_target_result.diagnostic);
    }

    // Bind the logical call before trying candidates. An invalid declaration
    // must be diagnosed even when no candidate divides its extent.
    std::optional<LogicalRequest> demand;
    const auto bound = bind_call_request(callsite, request.compilationContext, *entry, discovery_target, &demand);
    if (!bound.is_supported())
    {
      trace_metadata_result("discovery", decl_symbol, callsite_index, bound);
      if (bound.status == MetadataStatus::unsupported)
      {
        continue;
      }
      return lp::ProviderError("malformed cuda.coop.tile call to " + decl_symbol, bound.diagnostic);
    }
    std::string toolkit_error;
    if (!validate_provider_configuration(&toolkit_error))
    {
      return lp::ProviderError("cuda.coop.tile compiler configuration is invalid", std::move(toolkit_error));
    }
    const auto include_paths = resolve_provider_include_paths();
    if (!provider_include_paths_have_required_headers(include_paths, entry->family))
    {
      std::ostringstream detail;
      detail << "[LP-HEADERS-001] declaration " << decl_symbol << " requires "
             << provider_family_required_header(entry->family) << " and cuda_runtime.h; searched include paths:";
      for (const auto& path : include_paths)
      {
        detail << "\n  " << path;
      }
      return lp::ProviderError("cuda.coop.tile header configuration is incomplete", detail.str());
    }
    std::vector<CallMetadata> metas;
    if (request.compilationContext.num_warps_per_cta <= 0 && demand->automatic_topology())
    {
      for (const int64_t threads : kAutoTopologyCandidateThreads)
      {
        const auto* const reduce = std::get_if<ReduceSpec>(&demand->primitive());
        if (reduce && reduce->batches > 0 && reduce->algorithm == "warp_reductions" && threads != 32)
        {
          continue;
        }
        if ((!reduce || reduce->batches == 0) && (demand->extent() % threads) != 0)
        {
          continue;
        }
        CallMetadata meta;
        const MetadataResult result =
          materialize_call_request(callsite, request.compilationContext, *entry, *demand, threads, &meta);
        if (result.is_supported())
        {
          metas.push_back(std::move(meta));
          continue;
        }
        trace_metadata_result("discovery", decl_symbol, callsite_index, result);
        if (result.status == MetadataStatus::malformed)
        {
          return lp::ProviderError("malformed cuda.coop.tile call to " + decl_symbol, result.diagnostic);
        }
        if (result.status == MetadataStatus::internal_error)
        {
          return lp::ProviderError("cuda.coop.tile provider validation failed for " + decl_symbol, result.diagnostic);
        }
        if (result.status != MetadataStatus::unsupported)
        {
          return lp::ProviderError(
            "cuda.coop.tile provider validation failed for " + decl_symbol,
            "[LP-PROVIDER-INTERNAL-001] family parser returned an invalid discovery ownership status");
        }
      }
    }
    else
    {
      CallMetadata meta;
      const MetadataResult result =
        materialize_call_request(callsite, request.compilationContext, *entry, *demand, 0, &meta);
      if (result.is_supported())
      {
        metas.push_back(std::move(meta));
      }
      else
      {
        trace_metadata_result("discovery", decl_symbol, callsite_index, result);
        if (result.status == MetadataStatus::malformed)
        {
          return lp::ProviderError("malformed cuda.coop.tile call to " + decl_symbol, result.diagnostic);
        }
        if (result.status == MetadataStatus::internal_error)
        {
          return lp::ProviderError("cuda.coop.tile provider validation failed for " + decl_symbol, result.diagnostic);
        }
        if (result.status != MetadataStatus::unsupported)
        {
          return lp::ProviderError(
            "cuda.coop.tile provider validation failed for " + decl_symbol,
            "[LP-PROVIDER-INTERNAL-001] family parser returned an invalid discovery ownership status");
        }
      }
    }

    std::vector<CallMetadata> admitted_metas;
    std::vector<std::string> admitted_impl_symbols;
    for (const CallMetadata& meta : metas)
    {
      const std::string impl_symbol = make_impl_symbol(*entry, meta);
      if (!seen_specializations.insert(impl_symbol).second)
      {
        continue;
      }
      admitted_metas.push_back(meta);
      admitted_impl_symbols.push_back(impl_symbol);
    }

    std::vector<PrimitiveResourceEstimate> resource_estimates;
    std::string resource_error;
    const bool resources_proved =
      admitted_metas.empty()
      || (admitted_metas.front().request
          && estimate_primitive_resources_batch(
            &runtime,
            admitted_metas,
            admitted_metas.front().request->target().sm_arch,
            build_resource_source_for_primitive,
            &resource_estimates,
            &resource_error));
    if (!resources_proved)
    {
      return lp::ProviderError(
        "cuda.coop.tile could not query primitive resources for " + decl_symbol,
        resource_error.empty() ? "[LP-PROVIDER-INTERNAL-001] exact scratch storage could not be queried"
                               : std::move(resource_error));
    }
    for (size_t meta_index = 0; meta_index < admitted_metas.size(); ++meta_index)
    {
      const CallMetadata& meta = admitted_metas[meta_index];
      const PrimitiveResourceEstimate estimate =
        resources_proved && meta_index < resource_estimates.size()
          ? resource_estimates[meta_index]
          : PrimitiveResourceEstimate{};
      if (!resources_proved || !meta.request || !estimate.modeled || estimate.static_smem_bytes == 0
          || estimate.scratch_alignment_bytes == 0)
      {
        std::ostringstream trace;
        trace << "[coop_linkage_provider][trace] discovery status=unsupported symbol=" << decl_symbol
              << " reason=[LP-SCRATCH-001] [SCRATCH-BOUND-PROOF:" << provider_family_name(entry->family)
              << "] exact ScratchStorage size/alignment could not be proved";
        emit_trace_line(trace.str());
        continue;
      }
      const uint64_t static_smem_limit = max_static_smem_bytes_for_provider_bundle(meta.request->target().sm_arch);
      if (static_smem_limit > 0 && estimate.static_smem_bytes > static_smem_limit)
      {
        trace_resource_event("discovery_resource_over_budget", meta, estimate, static_smem_limit);
        continue;
      }

      const auto synchronization =
        uses_warp_execution_scope(*meta.request)
          ? lp::ImplCatalogBuilder::Synchronization::Warp
          : lp::ImplCatalogBuilder::Synchronization::CTA;
      auto impl = builder.implementation(admitted_impl_symbols[meta_index], decl_symbol, synchronization)
                    .frontendABI(kTileIRFloat16Type, kFrontendHalfWrapperType)
                    .frontendABI(kTileIRBFloat16Type, kFrontendHalfWrapperType);
      for (size_t j = 0; j < callsite.decl->num_const_params; ++j)
      {
        const TileIRConstantParameter& parameter = callsite.decl->const_params[j];
        const std::string name                   = to_string(parameter.name);
        const auto kind                          = to_builder_const_param_kind(parameter.kind);
        const auto argument =
          callsite.const_arguments && j < callsite.num_const_arguments
            ? make_concrete_catalog_argument(builder, parameter, callsite.const_arguments[j])
            : std::nullopt;
        if (name.empty() || !kind || !argument)
        {
          return lp::ProviderError("cuda.coop.tile provider catalog construction failed for " + decl_symbol,
                                   "[LP-PROVIDER-INTERNAL-001] admitted constant parameter cannot be represented");
        }
        // Each admitted row is tied to a concrete artifact identity and a
        // concrete fragment layout. Wildcard parameters plus concrete
        // arguments express that fixed specialization to TileIR so a row
        // discovered for one call cannot be monomorphized for another.
        impl.constantParameter("?", *kind).constantArgument(*argument);
      }
      for (size_t j = 0; j < callsite.num_arguments; ++j)
      {
        const bool scalar_by_value =
          argument_is_scalar_by_value(*entry, j, callsite.arguments[j], callsite.decl->param_access[j]);
        const auto* const reduce = std::get_if<ReduceSpec>(&meta.request->primitive());
        const auto fragment =
          reduce && reduce->batches > 0
            ? make_reduction_fragment(builder, callsite.arguments[j], *meta.request, j)
            : make_implementation_fragment(
                builder, callsite.arguments[j], meta.request->topology().block_threads, scalar_by_value);
        if (!fragment)
        {
          return lp::ProviderError("cuda.coop.tile provider catalog construction failed for " + decl_symbol,
                                   "[LP-PROVIDER-INTERNAL-001] admitted fragment cannot be represented");
        }
        const std::string argument_name = "arg" + std::to_string(j);
        impl.argument(argument_name, *fragment, implementation_access(callsite.decl->param_access[j]));
      }
      // CUDA 13.5 selects from the discovery catalog without filter_and_rank.
      // Rank each concrete topology while its metadata is available, including
      // when the compiler has not selected a kernel thread count yet.
      const int64_t provider_score = 100 + family_candidate_preference_score(*entry, *meta.request);
      impl
        .scratch(static_cast<int64_t>(estimate.static_smem_bytes),
                 static_cast<int64_t>(estimate.scratch_alignment_bytes))
        .providerScore(provider_score)
        .architecture(meta.request->target().sm_arch.sm_name());
      if (trace_enabled())
      {
        std::ostringstream trace;
        trace
          << "[coop_linkage_provider][trace] discovery candidate=" << meta_index << " callsite=" << callsite.id
          << " decl=" << decl_symbol << " impl=" << admitted_impl_symbols[meta_index] << " semantic_key="
          << semantic_identity(*meta.request) << " payload=" << request_payload(meta.request->primitive())
          << " threads=" << meta.request->topology().block_threads
          << " ipt=" << meta.request->topology().items_per_thread << " provider_score=" << provider_score
          << " compilation_warps=" << request.compilationContext.num_warps_per_cta;
        emit_trace_line(trace.str());
      }
      added = true;
    }
  }

  if (!added)
  {
    return lp::ImplCatalog{};
  }
  auto catalog = builder.build();
  if (!catalog)
  {
    return catalog.takeError();
  }
  return catalog.takeValue();
}

// Revalidate resolved candidates for compilers that invoke this optional hook.
// Write disposition and score into the compiler-owned result array. Preserve
// the baseline score because discovery already added topology preference.
// Neither catalog admission nor instantiation depends on this callback running.
lp::Result<void> CoopProviderState::filterAndRank(lp::FilterAndRankBatch& batch)
{
  for (size_t i = 0; i < batch.candidates.size; ++i)
  {
    const TileIRLPFilterCandidate& candidate = batch.candidates[i];
    TileIRLPRankResult& result               = batch.results[i];
    result.disposition                       = TILEIR_LP_RANK_SCORED;
    // The catalog already contains the topology preference. Preserve the
    // compiler's baseline if a compiler invokes this optional validation hook.
    result.score = candidate.baseline_score;
    if (candidate.provenance != TILEIR_LP_CANDIDATE_PROVIDER)
    {
      continue;
    }

    const FamilyDescriptor* entry =
      candidate.resolved_call.impl ? find_family_for_impl_symbol(candidate.resolved_call.impl->symbol) : nullptr;
    const std::string impl_symbol = candidate.resolved_call.impl ? to_string(candidate.resolved_call.impl->symbol) : "";
    if (!entry)
    {
      result.disposition = TILEIR_LP_RANK_FILTERED;
      result.score       = 0;
      trace_metadata_result("filter", impl_symbol.empty() ? "<null>" : impl_symbol, i, MetadataResult::not_owned());
      continue;
    }

    const std::string decl_symbol =
      candidate.resolved_call.decl ? to_string(candidate.resolved_call.decl->symbol) : "<null>";
    if (!candidate.resolved_call.decl || !candidate.resolved_call.impl
        || to_string(candidate.resolved_call.impl->decl_symbol) != decl_symbol
        || !declaration_symbol_matches_family(
          *entry, candidate.resolved_call.decl->symbol.data, candidate.resolved_call.decl->symbol.len, true))
    {
      return lp::ProviderError("cuda.coop.tile provider validation failed for candidate " + std::to_string(i),
                               "[LP-SYMBOL-001] Coop-owned implementation does not match a registered declaration");
    }

    TargetArchitecture impl_target;
    const MetadataResult impl_target_result = parse_impl_target_constraint(candidate.resolved_call, &impl_target);
    if (!impl_target_result.is_supported())
    {
      return lp::ProviderError("cuda.coop.tile provider target validation failed for " + decl_symbol,
                               impl_target_result.diagnostic);
    }

    CallMetadata meta;
    const MetadataResult metadata_result = parse_call_metadata(candidate.resolved_call, *entry, i, &meta);
    if (!metadata_result.is_supported())
    {
      trace_metadata_result("filter", decl_symbol, i, metadata_result);
      if (metadata_result.status == MetadataStatus::unsupported)
      {
        result.disposition = TILEIR_LP_RANK_FILTERED;
        result.score       = 0;
        continue;
      }
      const std::string message =
        metadata_result.status == MetadataStatus::malformed
          ? "malformed cuda.coop.tile ranking candidate for " + decl_symbol
          : "cuda.coop.tile provider validation failed for " + decl_symbol;
      return lp::ProviderError(message, metadata_result.diagnostic);
    }

    const MetadataResult target_result = compare_impl_target_constraint(impl_target, meta);
    if (!target_result.is_supported())
    {
      trace_metadata_result("filter", decl_symbol, i, target_result);
      if (target_result.status == MetadataStatus::unsupported)
      {
        result.disposition = TILEIR_LP_RANK_FILTERED;
        result.score       = 0;
        continue;
      }
      return lp::ProviderError("cuda.coop.tile provider target validation failed for " + decl_symbol,
                               target_result.diagnostic);
    }

    // A provider candidate must be the concrete specialization derived from
    // this resolved call. Request-format-1 discovery no longer admits a
    // process-wide history of otherwise-shaped implementation symbols.
    std::string expected_impl;
    if (!impl_symbol_matches_admitted_metadata(impl_symbol, *entry, meta, &expected_impl))
    {
      return lp::ProviderError(
        "cuda.coop.tile provider identity validation failed for " + decl_symbol,
        "[LP-PROVIDER-INTERNAL-001] Coop-owned implementation identity does not match admitted metadata: selected="
          + impl_symbol + " expected=" + expected_impl);
    }

    const TileIRExecutionContext& impl_context     = candidate.resolved_call.impl->execution_context_constraints;
    const TileIRExecutionContext& resolved_context = candidate.resolved_call.resolved_execution_context;
    if (!compare_impl_execution_context(impl_context, resolved_context).is_supported())
    {
      result.disposition = TILEIR_LP_RANK_FILTERED;
      result.score       = 0;
      if (trace_enabled())
      {
        std::ostringstream trace;
        trace << "[coop_linkage_provider][trace] filter candidate=" << i << " decl=" << decl_symbol
              << " status=incompatible-implementation-context impl_scope=" << static_cast<int>(impl_context.scope)
              << " resolved_scope=" << static_cast<int>(resolved_context.scope) << " impl_warps="
              << impl_context.num_warps_per_cta << " resolved_warps=" << resolved_context.num_warps_per_cta;
        emit_trace_line(trace.str());
      }
      continue;
    }

    const bool warp_scope     = meta.request && uses_warp_execution_scope(*meta.request);
    const auto expected_scope = warp_scope ? TILEIR_EXEC_SCOPE_WARP : TILEIR_EXEC_SCOPE_CTA;
    const bool scope_valid    = resolved_context.scope == expected_scope;
    const bool topology_valid =
      warp_scope || (meta.request && resolved_context.num_warps_per_cta * 32 == meta.request->topology().block_threads);
    if (trace_enabled())
    {
      trace_const_params(candidate.resolved_call, i);
      std::ostringstream trace;
      trace
        << "[coop_linkage_provider][trace] filter candidate=" << i << " callsite=" << candidate.callsite_id
        << " decl=" << decl_symbol << " impl=" << impl_symbol << " expected_impl=" << expected_impl
        << " metadata_status=supported target_status=supported scope_valid=" << scope_valid
        << " topology_valid=" << topology_valid << " scope=" << static_cast<int>(resolved_context.scope)
        << " warps=" << resolved_context.num_warps_per_cta
        << " threads=" << (meta.request ? meta.request->topology().block_threads : 0)
        << " ipt=" << (meta.request ? meta.request->topology().items_per_thread : 0)
        << " decl_const_params=" << (candidate.resolved_call.decl ? candidate.resolved_call.decl->num_const_params : 0)
        << " decl_access=" << (candidate.resolved_call.decl ? candidate.resolved_call.decl->num_param_access : 0);
      emit_trace_line(trace.str());
    }
    if (!scope_valid || !topology_valid)
    {
      result.disposition = TILEIR_LP_RANK_FILTERED;
      result.score       = 0;
      continue;
    }
  }
  return {};
}

// Compile the validated selected calls into one owned LTO-IR module.
// Rebind every call and verify selected symbols, targets, and execution scopes
// before using caches or NVRTC. Deduplicate identical artifacts for compilation,
// while mapping every original call index to its selected symbol. Successful
// cache lookups copy bytes; the returned module never borrows cache storage.
lp::Result<lp::ForeignModule> CoopProviderState::instantiate(const lp::ResolvedCallBatch& calls)
{
  if (calls.size == 0)
  {
    return lp::ForeignModuleBuilder(0).build();
  }

  const size_t num_calls = calls.size;
  std::vector<CallMetadata> call_metas(num_calls);
  std::vector<std::string> resolved_symbol_names;
  std::vector<std::string> unresolved_reasons(num_calls);
  std::vector<std::string> internal_failures;
  std::optional<TargetArchitecture> target_arch;
  std::optional<int64_t> target_epoch;
  bool mixed_target = false;
  bool mixed_epoch  = false;
  resolved_symbol_names.reserve(num_calls);
  const auto add_unresolved = [&](size_t index, const std::string& reason) {
    if (!unresolved_reasons[index].empty())
    {
      unresolved_reasons[index] += "; ";
    }
    unresolved_reasons[index] += reason;
  };

  // Validate the complete selected batch before touching provider caches,
  // probing resources, or invoking NVRTC. User-correctable failures are
  // accumulated so the compiler can report every unresolved call at once.
  for (size_t i = 0; i < num_calls; ++i)
  {
    const TileIRResolvedCall& call  = calls[i];
    const std::string decl_symbol   = call.decl ? to_string(call.decl->symbol) : "<null>";
    const std::string impl_identity = call.impl ? to_string(call.impl->symbol) : "";
    const FamilyDescriptor* entry   = call.impl ? find_family_for_impl_symbol(call.impl->symbol) : nullptr;
    if (!entry)
    {
      const MetadataResult result = MetadataResult::not_owned();
      trace_metadata_result("instantiate", decl_symbol, i, result);
      const std::string unresolved_impl = call.impl ? (impl_identity.empty() ? "<empty>" : impl_identity) : "<null>";
      add_unresolved(i,
                     "[LP-SYMBOL-001] selected implementation is not registered by cuda.coop.tile: " + unresolved_impl
                       + " (declaration " + decl_symbol + ")");
      continue;
    }
    if (!call.decl || !declaration_symbol_matches_family(*entry, call.decl->symbol.data, call.decl->symbol.len, true))
    {
      add_unresolved(i, "[LP-SYMBOL-001] selected implementation family does not match declaration " + decl_symbol);
      continue;
    }

    bool implementation_valid = true;
    TargetArchitecture impl_target;
    bool impl_target_valid = false;
    if (!call.impl || to_string(call.impl->decl_symbol) != decl_symbol)
    {
      add_unresolved(i, "[LP-SYMBOL-001] selected implementation does not name declaration " + decl_symbol);
      implementation_valid = false;
    }
    else
    {
      const MetadataResult impl_target_result = parse_impl_target_constraint(call, &impl_target);
      if (!impl_target_result.is_supported())
      {
        trace_metadata_result("instantiate-target", decl_symbol, i, impl_target_result);
        if (impl_target_result.status == MetadataStatus::internal_error)
        {
          internal_failures.push_back(
            "call " + std::to_string(i) + " (" + decl_symbol + "): " + impl_target_result.diagnostic);
        }
        else
        {
          add_unresolved(i, impl_target_result.diagnostic);
        }
      }
      else
      {
        impl_target_valid = true;
      }
    }

    CallMetadata meta;
    const MetadataResult metadata_result = parse_call_metadata(call, *entry, i, &meta);
    if (!metadata_result.is_supported())
    {
      trace_metadata_result("instantiate", decl_symbol, i, metadata_result);
      if (metadata_result.status == MetadataStatus::internal_error)
      {
        internal_failures.push_back(
          "call " + std::to_string(i) + " (" + decl_symbol + "): " + metadata_result.diagnostic);
      }
      else
      {
        add_unresolved(i, metadata_result.diagnostic);
      }
      continue;
    }
    if (!meta.request)
    {
      internal_failures.push_back("call " + std::to_string(i) + " (" + decl_symbol
                                  + "): [LP-PROVIDER-INTERNAL-001] supported metadata has no typed request");
      continue;
    }

    if (implementation_valid)
    {
      const MetadataResult context_result =
        compare_impl_execution_context(call.impl->execution_context_constraints, call.resolved_execution_context);
      if (!context_result.is_supported())
      {
        add_unresolved(i, context_result.diagnostic);
        implementation_valid = false;
      }
    }

    bool target_compatible = false;
    if (implementation_valid && impl_target_valid)
    {
      const MetadataResult target_result = compare_impl_target_constraint(impl_target, meta);
      if (!target_result.is_supported())
      {
        trace_metadata_result("instantiate-target", decl_symbol, i, target_result);
        if (target_result.status == MetadataStatus::internal_error)
        {
          internal_failures.push_back(
            "call " + std::to_string(i) + " (" + decl_symbol + "): " + target_result.diagnostic);
        }
        else
        {
          add_unresolved(i, target_result.diagnostic);
        }
      }
      else
      {
        target_compatible = true;
      }
    }

    if (implementation_valid && target_compatible)
    {
      std::string expected_impl;
      if (!impl_symbol_matches_admitted_metadata(impl_identity, *entry, meta, &expected_impl))
      {
        add_unresolved(
          i,
          "[LP-PROVIDER-INTERNAL-001] selected implementation identity does not match admitted metadata for "
            + decl_symbol + ": selected=" + impl_identity + " expected=" + expected_impl);
        implementation_valid = false;
      }
    }

    const ImplementationSpec& request = *meta.request;
    if (!target_arch)
    {
      target_arch = request.target().sm_arch;
    }
    else if (*target_arch != request.target().sm_arch)
    {
      mixed_target = true;
    }

    if (!target_epoch)
    {
      target_epoch = request.target().cache_epoch;
    }
    else if (*target_epoch != request.target().cache_epoch)
    {
      mixed_epoch = true;
    }

    call_metas[i] = std::move(meta);
    resolved_symbol_names.push_back(impl_identity);
  }

  if (!internal_failures.empty())
  {
    std::ostringstream details;
    for (size_t i = 0; i < internal_failures.size(); ++i)
    {
      if (i != 0)
      {
        details << '\n';
      }
      details << internal_failures[i];
    }
    return lp::ProviderError("cuda.coop.tile provider validation failed during instantiation", details.str());
  }
  std::vector<std::pair<size_t, std::string>> unresolved_calls;
  unresolved_calls.reserve(num_calls);
  for (size_t i = 0; i < unresolved_reasons.size(); ++i)
  {
    if (!unresolved_reasons[i].empty())
    {
      unresolved_calls.emplace_back(i, std::move(unresolved_reasons[i]));
    }
  }
  if (!unresolved_calls.empty())
  {
    return lp::UnresolvedCallsError(
      "cuda.coop.tile could not instantiate one or more selected calls", std::move(unresolved_calls));
  }
  if (!target_arch || !target_arch->valid() || !target_epoch || resolved_symbol_names.size() != num_calls)
  {
    return lp::ProviderError("cuda.coop.tile provider validation failed during instantiation",
                             "[LP-PROVIDER-INTERNAL-001] validated batch has incomplete target or symbol identity");
  }
  if (mixed_target)
  {
    return lp::ProviderError(
      "instantiate batch contains mixed target architectures",
      "[LP-TARGET-001] every otherwise-valid call in one batch must use the same exact target family");
  }
  if (mixed_epoch)
  {
    return lp::ProviderError("instantiate batch contains mixed cache epochs",
                             "[LP-PROVIDER-INTERNAL-001] every call in one batch must use the same cache epoch");
  }
  std::string toolkit_error;
  if (!validate_provider_configuration(&toolkit_error))
  {
    return lp::ProviderError("cuda.coop.tile compiler configuration is invalid", std::move(toolkit_error));
  }

  std::vector<CallMetadata> unique_metas;
  std::vector<std::string> unique_symbols;
  std::unordered_map<std::string, size_t> artifact_key_to_meta_index;
  std::vector<std::string> artifact_keys;
  unique_metas.reserve(num_calls);
  unique_symbols.reserve(num_calls);
  artifact_keys.reserve(num_calls);
  for (size_t i = 0; i < num_calls; ++i)
  {
    const CallMetadata& meta       = call_metas[i];
    const std::string& symbol      = resolved_symbol_names[i];
    const std::string artifact_key = make_codegen_artifact_cache_key(meta, symbol);
    if (artifact_key.empty())
    {
      return lp::ProviderError("provider artifact identity is unresolved",
                               "[LP-PROVIDER-INTERNAL-001] admitted metadata did not produce an artifact cache key");
    }
    const bool inserted = artifact_key_to_meta_index.emplace(artifact_key, unique_metas.size()).second;
    if (!inserted)
    {
      continue;
    }
    artifact_keys.push_back(artifact_key);
    unique_metas.push_back(meta);
    unique_symbols.push_back(symbol);
  }

  std::unordered_set<std::string> bundle_symbols;
  for (const std::string& symbol : unique_symbols)
  {
    if (!bundle_symbols.insert(symbol).second)
    {
      return lp::ProviderError(
        "instantiate batch contains an ambiguous selected implementation symbol",
        "[LP-PROVIDER-INTERNAL-001] distinct artifacts resolved to the same exported symbol " + symbol);
    }
  }

  const std::vector<std::string> include_paths = collect_bundle_include_paths(unique_metas);
  if (include_paths.empty())
  {
    return lp::ProviderError("provider bundle has no include paths",
                             "[LP-PROVIDER-INTERNAL-001] admitted metadata lost its compiler include paths");
  }

  for (size_t i = 0; i < unique_metas.size(); ++i)
  {
    bool seen_artifact_inserted = false;
    {
      std::lock_guard<std::mutex> lock(runtime.cache_mutex);
      clear_cache_if_at_limit(&runtime.seen_artifact_keys, seen_artifact_cache_limit());
      seen_artifact_inserted = runtime.seen_artifact_keys.insert(artifact_keys[i]).second;
    }
    trace_artifact_event(seen_artifact_inserted ? "cache_miss" : "cache_hit", unique_metas[i], artifact_keys[i]);
  }

  const uint64_t static_smem_limit = max_static_smem_bytes_for_provider_bundle(*target_arch);
  uint64_t max_modeled_static_smem = 0;
  std::vector<PrimitiveResourceEstimate> resource_estimates;
  std::string resource_error;
  if (!estimate_primitive_resources_batch(
        &runtime, unique_metas, *target_arch, build_resource_source_for_primitive, &resource_estimates, &resource_error))
  {
    if (resource_error.empty())
    {
      resource_error = "[LP-PROVIDER-INTERNAL-001] selected provider specialization failed its resource probe for "
                     + target_arch->sm_name();
    }
    return lp::ProviderError("failed to estimate primitive resources", std::move(resource_error));
  }
  for (size_t meta_index = 0; meta_index < unique_metas.size(); ++meta_index)
  {
    const CallMetadata& meta                  = unique_metas[meta_index];
    const PrimitiveResourceEstimate& estimate = resource_estimates[meta_index];
    if (!estimate.modeled)
    {
      continue;
    }
    trace_resource_event("resource_estimate", meta, estimate, static_smem_limit);
    max_modeled_static_smem = std::max(max_modeled_static_smem, estimate.static_smem_bytes);
  }
  if (static_smem_limit > 0 && max_modeled_static_smem > static_smem_limit)
  {
    std::ostringstream oss;
    oss << "[coop_linkage_provider][trace] resource_over_budget"
        << " static_smem=" << max_modeled_static_smem << " limit=" << static_smem_limit
        << " primitives=" << unique_metas.size();
    emit_trace_line(oss.str());

    std::ostringstream details;
    details << "[LP-UNSUPPORTED-001] modeled static shared memory " << max_modeled_static_smem
            << " exceeds configured limit " << static_smem_limit << " for " << target_arch->sm_name();
    return lp::ProviderError("provider bundle exceeds the configured static shared-memory limit", details.str());
  }

  const std::string bundle_key = make_bundle_cache_key(*target_arch, *target_epoch, artifact_keys, include_paths);

  std::vector<char> module_bytes;
  bool bundle_cache_hit = false;
  {
    std::lock_guard<std::mutex> lock(runtime.cache_mutex);
    if (const auto bundle_it = runtime.bundle_ltoir_cache.find(bundle_key);
        bundle_it != runtime.bundle_ltoir_cache.end())
    {
      module_bytes     = bundle_it->second;
      bundle_cache_hit = true;
    }
  }
  if (bundle_cache_hit)
  {
    trace_bundle_event("bundle_cache_hit", bundle_key, unique_metas.size());
  }
  else
  {
    trace_bundle_event("bundle_cache_miss", bundle_key, unique_metas.size());
    std::unordered_set<std::string> include_lines_seen;
    std::string bundle_includes;
    std::string bundle_fragments;
    for (size_t i = 0; i < unique_metas.size(); ++i)
    {
      const CallMetadata& meta = unique_metas[i];
      if (!meta.request)
      {
        return lp::ProviderError(
          "instantiate batch contains metadata without a typed provider request",
          "[LP-PROVIDER-INTERNAL-001] validated bundle metadata lost its typed provider request");
      }

      std::string source;
      std::string emission_error;
      if (!build_source_for_primitive_detailed(meta, unique_symbols[i], &source, &emission_error))
      {
        return lp::ProviderError(
          "provider source emission failed after validation",
          emission_error.empty() ? "[LP-PROVIDER-INTERNAL-001] source emitter returned no diagnostic"
                                 : std::move(emission_error));
      }
      const PrimitiveResourceEstimate& estimate = resource_estimates[i];
      source = replace_all(source, "%ADVERTISED_SCRATCH_BYTES%", std::to_string(estimate.static_smem_bytes));
      source = replace_all(source, "%ADVERTISED_SCRATCH_ALIGNMENT%", std::to_string(estimate.scratch_alignment_bytes));
      if (!specialize_source_for_bundle(&source, meta, i))
      {
        return lp::ProviderError("failed to specialize generated provider source",
                                 "[LP-PROVIDER-INTERNAL-001] bundle specialization rejected admitted metadata");
      }
      std::vector<std::string> source_includes;
      std::string source_body;
      split_source_includes_and_body(source, &source_includes, &source_body);
      if (source_body.empty())
      {
        return lp::ProviderError(
          "provider generated an empty source body",
          "[LP-PROVIDER-INTERNAL-001] emitted source for " + semantic_identity(*meta.request) + " has no body");
      }

      for (const std::string& include_line : source_includes)
      {
        if (include_line.empty())
        {
          continue;
        }
        if (include_lines_seen.insert(include_line).second)
        {
          bundle_includes += include_line;
          bundle_includes.push_back('\n');
        }
      }

      const std::string scope_name = "cuda_coop_tile_bundle_scope_" + std::to_string(i);
      bundle_fragments += "\n// ---- coop bundle fragment " + std::to_string(i) + " ----\n";
      // The body keeps its generated extern "C" entry. C language linkage
      // preserves the exact exported name inside a namespace, while the
      // namespace isolates each fragment's aliases and helper declarations.
      bundle_fragments += "namespace " + scope_name + " {\n";
      bundle_fragments += source_body;
      if (!bundle_fragments.empty() && bundle_fragments.back() != '\n')
      {
        bundle_fragments.push_back('\n');
      }
      bundle_fragments += "} // namespace " + scope_name + "\n";
    }

    std::string bundle_source = bundle_includes;
    bundle_source.push_back('\n');
    bundle_source += bundle_fragments;

    record_nvrtc_compile_counter();
    trace_bundle_event("nvrtc_compile", bundle_key, unique_metas.size());
    std::string nvrtc_error;
    if (!compile_source_to_ltoir(
          bundle_source, "coop_bundle", *target_arch, 0, 0, include_paths, &module_bytes, &nvrtc_error))
    {
      return lp::ProviderError(
        "failed to compile bundled provider source for " + target_arch->sm_name(),
        nvrtc_error.empty() ? "[LP-PROVIDER-INTERNAL-001] NVRTC failed without a diagnostic" : std::move(nvrtc_error));
    }
    if (module_bytes.empty())
    {
      return lp::ProviderError("bundled provider compilation produced empty LTO IR",
                               "[LP-PROVIDER-INTERNAL-001] NVRTC reported success with an empty output buffer");
    }
    {
      std::lock_guard<std::mutex> lock(runtime.cache_mutex);
      clear_cache_if_at_limit(&runtime.bundle_ltoir_cache, bundle_cache_limit());
      runtime.bundle_ltoir_cache.emplace(bundle_key, module_bytes);
    }
  }

  std::vector<std::string> exported_symbol_names;
  std::unordered_set<std::string> exported_symbols;
  exported_symbol_names.reserve(resolved_symbol_names.size());
  for (const std::string& name : resolved_symbol_names)
  {
    if (exported_symbols.insert(name).second)
    {
      exported_symbol_names.push_back(name);
    }
  }

  lp::ForeignModuleBuilder module(num_calls);
  for (size_t i = 0; i < num_calls; ++i)
  {
    module.mapCall(i, resolved_symbol_names[i]);
  }
  // Transfer the independent byte vector and symbol strings to the SDK. Cache
  // eviction or destruction of this callback's locals cannot invalidate them.
  module.addCodeBuffer(kCodeBufferName, TILEIR_OUTPUT_LTO_IR, std::move(module_bytes), std::move(exported_symbol_names));
  return module.build();
}

TileIRLPResult createProvider(TileIRLinkageProviderAPIVersion compiler_api_version, TileIRLinkageProviderInstance* out)
{
  return CoopProviderState::create(compiler_api_version, out, kProviderName);
}
} // namespace

TILEIR_DEFINE_LINKAGE_PROVIDER_ENTRY(createProvider)
