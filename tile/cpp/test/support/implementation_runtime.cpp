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
  rld.decoded_extent = 128;
  if (!check(provider::make_logical_request(
               provider::FamilyId::run_length_decode_with_offsets, rld, 256, {architecture, 0}, &demand)
               .is_supported(),
             "run-length decode with a smaller output extent did not bind")
      || !check(provider::select_implementation(*demand, 128, &selected_128).is_supported(),
                "compatible correlated run-length decode extents were rejected")
      || !check(selected_128 && selected_128->topology().items_per_thread == 2
                  && selected_128->topology().decoded_items_per_thread == 1,
                "run-length decode did not derive the smaller output extent"))
  {
    return false;
  }
  selected_256                           = selected_128;
  const auto incompatible_decoded_extent = provider::select_implementation(*demand, 256, &selected_256);
  if (!check(incompatible_decoded_extent.status == provider::MetadataStatus::unsupported && !selected_256,
             "incompatible decoded extent was admitted or retained a stale implementation")
      || !check(demand->extent() == 256
                  && std::get<provider::RunLengthDecodeSpec>(demand->primitive()).decoded_extent == 128,
                "failed run-length decode candidate selection changed logical extents"))
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

  auto resource_source = [](const provider::CallMetadata& meta, std::string* source, std::string* error) {
    if (!meta.request)
    {
      return false;
    }
    if (provider::emit_family_source({*meta.request, "probe_sum"}, source, error)
        != provider::SourceEmissionStatus::emitted)
    {
      return false;
    }
    *source = provider::replace_all(std::move(*source), "%ADVERTISED_SCRATCH_BYTES%", "sizeof(ScratchStorage)");
    *source = provider::replace_all(std::move(*source), "%ADVERTISED_SCRATCH_ALIGNMENT%", "alignof(ScratchStorage)");
    return true;
  };
  provider::RuntimeState state;
  provider::PrimitiveResourceEstimate estimate;
  if (!check(provider::estimate_primitive_resources(
               &state, python_metadata, python_request.target().sm_arch, resource_source, &estimate),
             "CUB scratch probe failed")
      || !check(estimate.modeled && estimate.temp_storage_bytes > 0
                  && estimate.static_smem_bytes >= estimate.temp_storage_bytes && estimate.scratch_alignment_bytes > 0,
                "CUB scratch probe returned an invalid packed layout"))
  {
    return 1;
  }
  std::string source;
  std::string error;
  const std::string oracle = R"CUDA(
#include <cub/block/block_reduce.cuh>
using BlockReduce = cub::BlockReduce<int, 128, cub::BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY>;
struct ExpectedStorage {
  typename BlockReduce::TempStorage temp_storage;
  int broadcast_result;
};
extern "C" __device__ __constant__ unsigned long long cuda_coop_tile_resource_sizes[] = {
  0x43554f5052455355ULL, 2ULL,
  sizeof(ExpectedStorage), alignof(ExpectedStorage), sizeof(BlockReduce::TempStorage)
};
)CUDA";
  std::vector<uint64_t> expected;
  if (!check(provider::compile_resource_query_to_sizes(
               oracle, python_request.target().sm_arch, python_metadata.include_paths, 3, &expected, &error),
             error.c_str())
      || !check(expected.size() == 3 && estimate.static_smem_bytes == expected[0]
                  && estimate.scratch_alignment_bytes == expected[1] && estimate.temp_storage_bytes == expected[2],
                "provider scratch disagrees with independent CUB layout"))
  {
    return 1;
  }
  if (!check(provider::emit_family_source({python_request, "probe_sum"}, &source, &error)
               == provider::SourceEmissionStatus::emitted,
             "CUB source emission failed"))
  {
    return 1;
  }
  const std::string original_source = source;
  source =
    provider::replace_all(std::move(source), "%ADVERTISED_SCRATCH_BYTES%", std::to_string(estimate.static_smem_bytes));
  source = provider::replace_all(
    std::move(source), "%ADVERTISED_SCRATCH_ALIGNMENT%", std::to_string(estimate.scratch_alignment_bytes));
  std::vector<char> ltoir;
  if (!check(
        provider::compile_source_to_ltoir(
          source, "probe_sum", python_request.target().sm_arch, 128, 2, python_metadata.include_paths, &ltoir, &error),
        error.c_str())
      || !check(!ltoir.empty(), "valid CUB source produced no LTO IR"))
  {
    return 1;
  }
  std::string insufficient = provider::replace_all(
    original_source, "%ADVERTISED_SCRATCH_BYTES%", std::to_string(estimate.static_smem_bytes - 1));
  insufficient = provider::replace_all(
    std::move(insufficient), "%ADVERTISED_SCRATCH_ALIGNMENT%", std::to_string(estimate.scratch_alignment_bytes));
  if (!check(!provider::compile_source_to_ltoir(
               insufficient,
               "probe_sum",
               python_request.target().sm_arch,
               128,
               2,
               python_metadata.include_paths,
               &ltoir,
               &error),
             "under-advertised scratch compiled")
      || !check(ltoir.empty() && error.find("LP-SCRATCH-001") != std::string::npos,
                "under-advertised scratch lost its diagnostic or retained an artifact"))
  {
    return 1;
  }
  std::vector<char> invalid_ltoir{'s', 't', 'a', 'l', 'e'};
  if (!check(!provider::compile_source_to_ltoir(
               "invalid CUDA source", "invalid", python_request.target().sm_arch, 128, 2, {}, &invalid_ltoir, &error),
             "invalid source compiled")
      || !check(invalid_ltoir.empty() && error.find("nvrtcCompileProgram") != std::string::npos,
                "compile failure retained output or lost its diagnostic"))
  {
    return 1;
  }
  return 0;
}
