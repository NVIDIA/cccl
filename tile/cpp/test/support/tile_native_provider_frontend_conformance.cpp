// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <array>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "provider/families/emitter.h"
#include "provider/metadata_parser.h"
#include "provider/provider_request.h"
#include "provider/registry.h"
#include "provider/runtime/runtime.h"
#include "provider/source_common.h"
#include "provider/spec_identity.h"

// Source and implementation-symbol construction are intentionally private to
// the provider translation unit. Include it here instead of adding test-only
// production hooks: this host test exercises the exact code shipped in the
// provider without invoking the TileIR compiler.
#include "coop_linkage_provider.cpp"

namespace provider = cuda::coop::tile::detail::provider;

static_assert(!std::is_default_constructible_v<provider::LogicalRequest>);
static_assert(!std::is_default_constructible_v<provider::ImplementationSpec>);
static_assert(!std::is_constructible_v<provider::LogicalRequest, provider::PrimitiveSpec, int64_t, provider::Target>);
static_assert(!std::is_constructible_v<provider::ImplementationSpec, provider::LogicalRequest, provider::Topology>);
static_assert(
  std::is_same_v<decltype(std::declval<provider::ImplementationSpec&>().demand()), const provider::LogicalRequest&>);
static_assert(
  std::is_same_v<decltype(std::declval<provider::ImplementationSpec&>().topology()), const provider::Topology&>);
static_assert(
  std::is_same_v<decltype(std::declval<provider::ImplementationSpec&>().primitive()), const provider::PrimitiveSpec&>);

namespace cuda::coop::tile::detail::provider
{
// This host-only test does not resolve or compile against CUB include paths.
// Keep metadata construction linkable without pulling in the filesystem-facing
// include-path discovery translation unit.
std::vector<std::string> resolve_provider_include_paths()
{
  return {"canonical-headers"};
}

bool validate_provider_toolkit(std::string* out_error)
{
  out_error->clear();
  return true;
}
} // namespace cuda::coop::tile::detail::provider

namespace
{
inline constexpr char canonical_declaration[] = "cuda_coop_tile_sum";
inline constexpr char resolved_declaration[]  = "cuda_coop_tile_sum__canonical_frontend_fixture";
inline constexpr char algorithm[]             = "raking_commutative_only";
inline constexpr char dtype_token[]           = "i32";
inline constexpr char target[]                = "sm_120";

TileIRStringView tile_string_view(const char* value)
{
  return TileIRStringView{value, std::strlen(value)};
}

bool check(bool condition, const char* message)
{
  if (!condition)
  {
    std::cerr << message << '\n';
  }
  return condition;
}

struct CanonicalSumCall
{
  std::array<TileIRDim, 1> shape{};
  std::array<TileIRForeignType, 2> arguments{};
  std::array<TileIRForeignType, 1> results{};
  std::array<TileIRLPAccessKind, 2> access{};
  TileIRForeignDecl declaration{};
  std::array<TileIRConstantArgument, 4> constants{};
  TileIRResolvedCall call{};

  CanonicalSumCall()
  {
    shape[0].kind  = TILEIR_DIM_CONCRETE;
    shape[0].value = 256;

    const TileIRForeignType value_type{
      TILEIR_TYPE_TILE,
      shape.data(),
      shape.size(),
      TileIRDataType{TILEIR_DTYPE_I32},
    };
    arguments = {value_type, value_type};
    results   = {value_type};

    access = {TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_OUT};

    // Ranking and instantiation receive a compiler-concretized declaration:
    // its symbol has an opaque compiler suffix and its symbolic parameter
    // schema has been erased. The ordered concrete arguments below are the
    // surviving source of constant metadata at this lifecycle boundary.
    declaration = TileIRForeignDecl{
      tile_string_view(resolved_declaration),
      nullptr,
      0,
      arguments.data(),
      arguments.size(),
      results.data(),
      results.size(),
      access.data(),
      access.size(),
    };

    constants[0].kind       = TILEIR_CONSTANT_ARGUMENT_INT;
    constants[0].int_val    = 256;
    constants[1].kind       = TILEIR_CONSTANT_ARGUMENT_DTYPE;
    constants[1].dtype_val  = TileIRDataType{TILEIR_DTYPE_I32};
    constants[2].kind       = TILEIR_CONSTANT_ARGUMENT_STRING;
    constants[2].string_val = tile_string_view(algorithm);
    constants[3].kind       = TILEIR_CONSTANT_ARGUMENT_STRING;
    constants[3].string_val = tile_string_view(dtype_token);

    call.decl                       = &declaration;
    call.impl                       = nullptr;
    call.resolved_execution_context = TileIRExecutionContext{
      TILEIR_EXEC_SCOPE_CTA,
      4,
      -1,
      tile_string_view(target),
    };
    call.const_arguments     = constants.data();
    call.num_const_arguments = constants.size();
    call.arguments           = arguments.data();
    call.num_arguments       = arguments.size();
  }
};

struct CanonicalSumDiscovery : CanonicalSumCall
{
  std::array<TileIRConstantParameter, 4> parameters{};
  std::array<TileIRSymbolicExpression, 4> symbolic_constants{};
  TileIRLPCallsite callsite{};
  TileIRLPCompilationContext context{};

  CanonicalSumDiscovery()
  {
    const char* names[] = {"N", "T", "Algorithm", "DTypeToken"};
    for (size_t i = 0; i < constants.size(); ++i)
    {
      parameters[i].name = tile_string_view(names[i]);
      if (constants[i].kind == TILEIR_CONSTANT_ARGUMENT_INT)
      {
        parameters[i].kind            = TILEIR_CONSTANT_PARAMETER_INT;
        symbolic_constants[i].kind    = TILEIR_SYMBOLIC_EXPRESSION_INT;
        symbolic_constants[i].int_val = constants[i].int_val;
      }
      else if (constants[i].kind == TILEIR_CONSTANT_ARGUMENT_DTYPE)
      {
        parameters[i].kind              = TILEIR_CONSTANT_PARAMETER_DTYPE;
        symbolic_constants[i].kind      = TILEIR_SYMBOLIC_EXPRESSION_DTYPE;
        symbolic_constants[i].dtype_val = constants[i].dtype_val;
      }
      else
      {
        parameters[i].kind               = TILEIR_CONSTANT_PARAMETER_STRING;
        symbolic_constants[i].kind       = TILEIR_SYMBOLIC_EXPRESSION_STRING;
        symbolic_constants[i].string_val = constants[i].string_val;
      }
    }
    declaration.symbol           = tile_string_view(canonical_declaration);
    declaration.const_params     = parameters.data();
    declaration.num_const_params = parameters.size();
    callsite.decl                = &declaration;
    callsite.const_arguments     = symbolic_constants.data();
    callsite.num_const_arguments = symbolic_constants.size();
    callsite.arguments           = arguments.data();
    callsite.num_arguments       = arguments.size();
    callsite.results             = results.data();
    callsite.num_results         = results.size();
    context.target_architecture  = tile_string_view(target);
    context.num_warps_per_cta    = -1;
    context.num_ctas_per_cga     = -1;
  }
};

provider::CallMetadata parse_metadata(
  const provider::FamilyDescriptor& descriptor, const CanonicalSumCall& frontend, const char* frontend_name)
{
  provider::CallMetadata metadata;
  const provider::MetadataResult result = provider::parse_call_metadata(frontend.call, descriptor, 0, &metadata);
  if (!result.is_supported())
  {
    std::cerr << frontend_name << " canonical metadata was rejected: " << result.diagnostic << '\n';
    std::exit(2);
  }
  return metadata;
}

bool typed_requests_match(const provider::ImplementationSpec& lhs, const provider::ImplementationSpec& rhs)
{
  const auto* lhs_reduce = std::get_if<provider::ReduceSpec>(&lhs.primitive());
  const auto* rhs_reduce = std::get_if<provider::ReduceSpec>(&rhs.primitive());
  return lhs.topology().block_threads == rhs.topology().block_threads
      && lhs.topology().items_per_thread == rhs.topology().items_per_thread
      && lhs.target().sm_arch == rhs.target().sm_arch && lhs.target().cache_epoch == rhs.target().cache_epoch
      && lhs_reduce && rhs_reduce && lhs_reduce->generic == rhs_reduce->generic
      && lhs_reduce->dtype == rhs_reduce->dtype && lhs_reduce->algorithm == rhs_reduce->algorithm
      && lhs_reduce->binary_op == rhs_reduce->binary_op;
}

bool check_logical_selection(const provider::FamilyDescriptor& descriptor)
{
  CanonicalSumDiscovery frontend;
  const provider::TargetArchitecture architecture{120};
  std::optional<provider::LogicalRequest> demand;
  if (!check(provider::bind_call_request(frontend.callsite, frontend.context, descriptor, architecture, &demand)
               .is_supported(),
             "automatic discovery demand did not bind")
      || !check(demand && demand->automatic_topology() && demand->extent() == 256,
                "bound demand lost its logical extent or automatic constraint"))
  {
    return false;
  }

  // A bound demand owns its values; a changed compiler callsite must not alter
  // candidate selection or the original operation options.
  frontend.symbolic_constants[0].int_val = 512;
  std::optional<provider::ImplementationSpec> selected_128;
  std::optional<provider::ImplementationSpec> selected_256;
  if (!check(provider::select_implementation(*demand, 128, &selected_128).is_supported()
               && provider::select_implementation(*demand, 256, &selected_256).is_supported(),
             "one automatic demand could not select both worker counts")
      || !check(selected_128 && selected_256 && selected_128->topology().block_threads == 128
                  && selected_128->topology().items_per_thread == 2 && selected_256->topology().block_threads == 256
                  && selected_256->topology().items_per_thread == 1,
                "automatic selection did not derive each concrete topology")
      || !check(selected_128->demand().extent() == 256 && selected_256->demand().extent() == 256
                  && demand->automatic_topology(),
                "selection mutated the logical demand or borrowed a changed callsite")
      || !check(std::get<provider::ReduceSpec>(selected_128->primitive()).algorithm == algorithm
                  && std::get<provider::ReduceSpec>(selected_256->primitive()).algorithm == algorithm,
                "candidate selection changed operation options")
      || !check(provider::artifact_identity(*selected_128) != provider::artifact_identity(*selected_256),
                "distinct concrete topologies shared an artifact identity"))
  {
    return false;
  }
  const auto unsupported = provider::select_implementation(*demand, 96, &selected_256);
  if (!check(unsupported.status == provider::MetadataStatus::unsupported && !selected_256,
             "failed candidate selection retained a stale concrete implementation"))
  {
    return false;
  }

  provider::RunLengthDecodeSpec rld;
  rld.with_offsets   = true;
  rld.value_dtype    = "i32";
  rld.length_dtype   = "u32";
  rld.decoded_extent = 1024;
  if (!check(provider::make_logical_request(
               provider::FamilyId::run_length_decode_with_offsets, rld, 256, {architecture, 0}, &demand)
               .is_supported(),
             "run-length decode demand did not bind")
      || !check(provider::select_implementation(*demand, 128, &selected_128).is_supported()
                  && provider::select_implementation(*demand, 256, &selected_256).is_supported(),
                "run-length decode demand could not select both worker counts")
      || !check(
        selected_128 && selected_256 && selected_128->topology().items_per_thread == 2
          && selected_128->topology().decoded_items_per_thread == 8 && selected_256->topology().items_per_thread == 1
          && selected_256->topology().decoded_items_per_thread == 4,
        "run-length decode did not derive both per-thread extents")
      || !check(demand->extent() == 256
                  && std::get<provider::RunLengthDecodeSpec>(demand->primitive()).decoded_extent == 1024,
                "run-length decode selection changed a logical extent"))
  {
    return false;
  }
  rld.decoded_extent     = 0;
  const auto invalid_rld = provider::make_logical_request(
    provider::FamilyId::run_length_decode_with_offsets, rld, 256, {architecture, 0}, &demand);
  return check(invalid_rld.status == provider::MetadataStatus::malformed && !demand,
               "invalid decoded extent retained a bound demand");
}

std::string resource_source_oracle(
  const std::string& emitted_source, const std::string& selected_symbol, const std::string& probe_symbol)
{
  std::string source = emitted_source;
  source             = provider::replace_all(std::move(source), selected_symbol, probe_symbol);
  source             = provider::replace_all(std::move(source), "%ADVERTISED_SCRATCH_BYTES%", "sizeof(ScratchStorage)");
  source = provider::replace_all(std::move(source), "%ADVERTISED_SCRATCH_ALIGNMENT%", "alignof(ScratchStorage)");
  return source;
}
} // namespace

int main()
{
  const provider::TargetArchitecture corrupted_target{
    90,
    static_cast<provider::TargetArchitectureFamily>(77),
  };
  if (!check(!corrupted_target.valid(), "out-of-range target family was treated as valid")
      || !check(corrupted_target.sm_name().empty(), "corrupted target produced an sm cache identity")
      || !check(corrupted_target.compute_name().empty(), "corrupted target produced an NVRTC architecture"))
  {
    return 1;
  }

  const provider::FamilyDescriptor* const descriptor =
    provider::find_family_descriptor(canonical_declaration, std::strlen(canonical_declaration));
  if (!check(descriptor != nullptr, "registry rejected the canonical sum declaration")
      || !check(descriptor->family == provider::FamilyId::sum, "registry selected the wrong family")
      || !check(std::strcmp(descriptor->decl_symbol, canonical_declaration) == 0,
                "registry changed the canonical declaration")
      || !check(std::strcmp(descriptor->impl_symbol, "cuda_coop_tile_sum_impl") == 0,
                "registry changed the canonical implementation base"))
  {
    return 1;
  }

  if (!check_logical_selection(*descriptor))
  {
    return 1;
  }

  // Source-contract tests independently prove that Python descriptors and C++
  // foreign declarations converge on this call shape. From that shared LP
  // boundary, prove that frontend labels cannot enter provider identities or
  // generated artifacts. Executing both real frontends remains an exact-stack
  // qualification gate rather than a host-only provider test.
  const CanonicalSumCall python_frontend;
  const CanonicalSumCall cpp_frontend;
  const provider::CallMetadata python_metadata = parse_metadata(*descriptor, python_frontend, "Python");
  const provider::CallMetadata cpp_metadata    = parse_metadata(*descriptor, cpp_frontend, "C++");
  if (!check(python_metadata.request.has_value() && cpp_metadata.request.has_value(),
             "canonical parsing did not produce typed requests")
      || !check(typed_requests_match(*python_metadata.request, *cpp_metadata.request),
                "equivalent frontends produced different typed requests"))
  {
    return 1;
  }

  provider::CallMetadata reset_probe = python_metadata;
  CanonicalSumCall invalid_target_frontend;
  invalid_target_frontend.call.resolved_execution_context.architecture = tile_string_view("SM_90");
  const provider::MetadataResult reset_result =
    provider::parse_call_metadata(invalid_target_frontend.call, *descriptor, 0, &reset_probe);
  if (!check(reset_result.status == provider::MetadataStatus::malformed,
             "invalid target did not produce malformed metadata")
      || !check(!reset_probe.request, "target failure left typed metadata in the output")
      || !check(reset_probe.include_paths.empty(), "target failure left artifact metadata in the output"))
  {
    return 1;
  }

  const provider::ImplementationSpec& python_request = *python_metadata.request;
  const provider::ImplementationSpec& cpp_request    = *cpp_metadata.request;
  const std::string expected_semantic_identity =
    "coop::cuda_coop_tile_sum|algorithm=raking_commutative_only;"
    "block_threads=128;dtype=i32;items_per_thread=2";
  const std::string expected_artifact_identity = expected_semantic_identity + "|tile_interface_abi=2|target=sm_120";
  if (!check(provider::semantic_identity(python_request) == provider::semantic_identity(cpp_request),
             "semantic identity differs across equivalent frontends")
      || !check(provider::semantic_identity(python_request) == expected_semantic_identity,
                "semantic identity does not match the canonical oracle")
      || !check(provider::codegen_metadata(python_request) == provider::codegen_metadata(cpp_request),
                "generated-source metadata differs across equivalent frontends")
      || !check(provider::artifact_identity(python_request) == provider::artifact_identity(cpp_request),
                "artifact identity differs across equivalent frontends")
      || !check(provider::artifact_identity(python_request) == expected_artifact_identity,
                "artifact identity omitted or changed the exact target"))
  {
    return 1;
  }

  CanonicalSumCall unsigned_frontend;
  unsigned_frontend.constants[3].string_val      = tile_string_view("u32");
  const provider::CallMetadata unsigned_metadata = parse_metadata(*descriptor, unsigned_frontend, "unsigned");
  const std::string expected_unsigned_identity =
    "coop::cuda_coop_tile_sum|algorithm=raking_commutative_only;"
    "block_threads=128;dtype=u32;items_per_thread=2|tile_interface_abi=2|target=sm_120";
  if (!check(unsigned_metadata.request
               && std::get<provider::ReduceSpec>(unsigned_metadata.request->primitive()).dtype == "u32",
             "collapsed TileIR i32 lost the explicit unsigned token")
      || !check(provider::artifact_identity(*unsigned_metadata.request) == expected_unsigned_identity,
                "unsigned artifact identity changed during implementation selection")
      || !check(provider::artifact_identity(*unsigned_metadata.request) != expected_artifact_identity,
                "signed and unsigned calls shared an artifact identity"))
  {
    return 1;
  }

  const std::string python_impl_symbol = make_impl_symbol(*descriptor, python_metadata);
  const std::string cpp_impl_symbol    = make_impl_symbol(*descriptor, cpp_metadata);
  const std::string selected_symbol    = python_impl_symbol + "__canonical_frontend_fixture";
  if (!check(python_impl_symbol == cpp_impl_symbol, "implementation symbol differs across equivalent frontends")
      || !check(python_impl_symbol
                  == std::string(descriptor->impl_symbol) + "__lpv4_"
                       + provider::hex_u64(provider::fnv1a_hash(expected_artifact_identity)),
                "implementation symbol does not match the canonical artifact identity")
      || !check(impl_symbol_is_specialization_of(selected_symbol, python_impl_symbol),
                "selected symbol is not an authenticated concrete implementation"))
  {
    return 1;
  }

  const std::string python_cache_key = provider::make_codegen_artifact_cache_key(python_metadata, selected_symbol);
  const std::string cpp_cache_key    = provider::make_codegen_artifact_cache_key(cpp_metadata, selected_symbol);
  if (!check(python_cache_key == cpp_cache_key, "cache key differs across equivalent frontends")
      || !check(python_cache_key.find("|callable=" + selected_symbol + "|") != std::string::npos,
                "cache key omitted the exact selected callable")
      || !check(python_cache_key.find("|sm=sm_120|") != std::string::npos,
                "cache key omitted the exact target identity"))
  {
    return 1;
  }

  std::string python_source;
  std::string cpp_source;
  const std::string expected_entry_definition = "extern \"C\" __device__ void " + selected_symbol + "(";
  if (!check(build_source_for_primitive(python_metadata, selected_symbol, &python_source),
             "Python-lowered canonical source construction failed")
      || !check(build_source_for_primitive(cpp_metadata, selected_symbol, &cpp_source),
                "C++-lowered canonical source construction failed")
      || !check(python_source == cpp_source, "emitted wrapper source differs across equivalent frontends")
      || !check(python_source.find(expected_entry_definition) != std::string::npos,
                "emitted wrapper does not define the canonical entry"))
  {
    return 1;
  }

  std::string python_resource_source;
  std::string cpp_resource_source;
  std::string python_detail;
  std::string cpp_detail;
  const std::string probe_symbol = resource_probe_symbol(python_metadata);
  if (!check(build_resource_source_for_primitive(python_metadata, &python_resource_source, &python_detail),
             "Python-lowered resource source construction failed")
      || !check(build_resource_source_for_primitive(cpp_metadata, &cpp_resource_source, &cpp_detail),
                "C++-lowered resource source construction failed")
      || !check(python_resource_source == cpp_resource_source, "resource source differs across equivalent frontends")
      || !check(python_resource_source == resource_source_oracle(python_source, selected_symbol, probe_symbol),
                "resource source does not match the emitted-wrapper oracle")
      || !check(python_resource_source.find("extern \"C\" __device__ void " + probe_symbol + "(") != std::string::npos,
                "resource source does not use its internal probe symbol")
      || !check(python_resource_source.find(selected_symbol) == std::string::npos,
                "resource source leaked the selected callable symbol")
      || !check(python_detail == "reduce_sum" && cpp_detail == "reduce_sum",
                "resource-source detail differs from the canonical family payload"))
  {
    return 1;
  }

  std::vector<char> invalid_ltoir{'s', 't', 'a', 'l', 'e'};
  std::string nvrtc_error;
  const bool invalid_source_compiled = provider::compile_source_to_ltoir(
    "this is not valid CUDA C++ source",
    "invalid_provider_source",
    provider::TargetArchitecture{80, provider::TargetArchitectureFamily::none},
    0,
    0,
    {},
    &invalid_ltoir,
    &nvrtc_error);
  if (!check(!invalid_source_compiled, "invalid provider source unexpectedly compiled")
      || !check(invalid_ltoir.empty(), "failed provider compile produced LTO-IR")
      || !check(nvrtc_error.find("[LP-PROVIDER-INTERNAL-001]") != std::string::npos,
                "NVRTC failure omitted the stable provider category")
      || !check(nvrtc_error.find("nvrtcCompileProgram") != std::string::npos,
                "NVRTC failure omitted the failing operation"))
  {
    return 1;
  }

  std::vector<uint64_t> invalid_resource_sizes{42};
  std::string resource_error;
  const bool invalid_resource_query = provider::compile_resource_query_to_sizes(
    "unused", corrupted_target, {}, 1, &invalid_resource_sizes, &resource_error);
  if (!check(!invalid_resource_query, "invalid resource target unexpectedly compiled")
      || !check(invalid_resource_sizes.empty(), "failed resource query preserved stale size data")
      || !check(resource_error.find("[LP-PROVIDER-INTERNAL-001]") != std::string::npos,
                "resource-query target failure omitted the stable provider category"))
  {
    return 1;
  }

  return 0;
}
