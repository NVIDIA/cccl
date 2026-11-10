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

#include "provider/metadata_parser.h"
#include "provider/provider_request.h"
#include "provider/registry.h"
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
} // namespace

int main()
{
  const auto* descriptor = provider::find_family_descriptor(canonical_declaration, std::strlen(canonical_declaration));
  if (!check(descriptor && descriptor->family == provider::FamilyId::sum, "sum registry lookup failed"))
  {
    return 1;
  }

  CanonicalSumDiscovery frontend;
  const provider::TargetArchitecture architecture{120};
  std::optional<provider::LogicalRequest> demand;
  auto bind = [&] {
    return provider::bind_call_request(frontend.callsite, frontend.context, *descriptor, architecture, &demand);
  };
  if (!check(bind().is_supported(), "automatic sum call did not bind")
      || !check(demand && demand->automatic_topology() && demand->extent() == 256,
                "logical shape or automatic selection changed"))
  {
    return 1;
  }

  frontend.symbolic_constants[0].int_val = 512;
  if (!check(demand->extent() == 256, "logical request borrowed mutable compiler storage"))
  {
    return 1;
  }
  auto mismatch = bind();
  if (!check(mismatch.status == provider::MetadataStatus::malformed && !demand,
             "shape mismatch retained a stale logical request"))
  {
    return 1;
  }

  frontend.symbolic_constants[0].int_val    = 256;
  frontend.symbolic_constants[3].string_val = tile_string_view("u32");
  if (!check(bind().is_supported() && demand, "unsigned dtype token did not bind")
      || !check(std::get<provider::ReduceSpec>(demand->primitive()).dtype == "u32",
                "signless TileIR type lost the unsigned dtype token"))
  {
    return 1;
  }

  // An ABI-v1 tuple must not silently bind after physical constants move into
  // implementation selection.
  CanonicalSumDiscovery legacy;
  std::array<TileIRSymbolicExpression, 6> legacy_constants{
    legacy.symbolic_constants[0],
    legacy.symbolic_constants[1],
    {},
    {},
    legacy.symbolic_constants[2],
    legacy.symbolic_constants[3]};
  legacy_constants[2].kind            = TILEIR_SYMBOLIC_EXPRESSION_INT;
  legacy_constants[2].int_val         = 128;
  legacy_constants[3].kind            = TILEIR_SYMBOLIC_EXPRESSION_INT;
  legacy_constants[3].int_val         = 2;
  legacy.callsite.const_arguments     = legacy_constants.data();
  legacy.callsite.num_const_arguments = legacy_constants.size();
  const auto legacy_error =
    provider::bind_call_request(legacy.callsite, legacy.context, *descriptor, architecture, &demand);
  if (!check(legacy_error.status == provider::MetadataStatus::malformed && !demand,
             "obsolete physical declaration constants were admitted"))
  {
    return 1;
  }

  frontend.access[0] = TILEIR_LP_ACCESS_OUT;
  auto access_error  = bind();
  if (!check(access_error.status == provider::MetadataStatus::malformed && !demand,
             "invalid access was admitted or retained stale output"))
  {
    return 1;
  }
  frontend.access[0]          = TILEIR_LP_ACCESS_IN;
  frontend.parameters[0].kind = TILEIR_CONSTANT_PARAMETER_STRING;
  auto kind_error             = bind();
  if (!check(kind_error.status == provider::MetadataStatus::malformed && !demand,
             "invalid constant kind was admitted or retained stale output"))
  {
    return 1;
  }
  return 0;
}
