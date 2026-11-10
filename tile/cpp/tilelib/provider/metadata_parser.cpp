// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "provider/metadata_parser.h"

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

#include "provider/call_view.h"
#include "provider/declaration_schemas.h"
#include "provider/families/emitter.h"
#include "provider/provider_request.h"
#include "provider/registry.h"
#include "provider/runtime/runtime.h"
#include "provider/source_common.h"
#include "provider/spec_identity.h"

namespace cuda::coop::tile::detail::provider
{
MetadataResult MetadataResult::success()
{
  return {MetadataStatus::supported, {}};
}

MetadataResult MetadataResult::not_owned()
{
  return {MetadataStatus::not_owned, {}};
}

MetadataResult MetadataResult::unsupported(std::string diagnostic)
{
  if (diagnostic.empty())
  {
    diagnostic = "[LP-UNSUPPORTED-001] unsupported metadata";
  }
  return {MetadataStatus::unsupported, std::move(diagnostic)};
}

MetadataResult MetadataResult::malformed(std::string diagnostic)
{
  if (diagnostic.empty())
  {
    diagnostic = "[LP-SCHEMA-001] malformed metadata";
  }
  return {MetadataStatus::malformed, std::move(diagnostic)};
}

MetadataResult MetadataResult::internal_error(std::string diagnostic)
{
  if (diagnostic.empty())
  {
    diagnostic = "[LP-PROVIDER-INTERNAL-001] metadata validation failed without a diagnostic";
  }
  return {MetadataStatus::internal_error, std::move(diagnostic)};
}

bool merge_sort_block_threads_are_supported(int64_t block_threads)
{
  return block_threads >= 32 && block_threads <= 1024 && (block_threads & (block_threads - 1)) == 0;
}

namespace
{
constexpr int64_t kMaxRadixSortBits = 8;

MetadataResult
metadata_error(MetadataStatus status, const FamilyDescriptor& entry, const char* code, const std::string& detail)
{
  const std::string diagnostic = std::string(code) + " " + family_name(entry.family) + ": " + detail;
  switch (status)
  {
    case MetadataStatus::unsupported:
      return MetadataResult::unsupported(diagnostic);
    case MetadataStatus::malformed:
      return MetadataResult::malformed(diagnostic);
    case MetadataStatus::internal_error:
      return MetadataResult::internal_error(diagnostic);
    case MetadataStatus::supported:
      return MetadataResult::success();
    case MetadataStatus::not_owned:
      return MetadataResult::not_owned();
  }
  return MetadataResult::internal_error("[LP-PROVIDER-INTERNAL-001] unknown metadata status");
}

MetadataResult schema_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-SCHEMA-001]", detail);
}

MetadataResult constant_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-CONSTANT-001]", detail);
}

MetadataResult type_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-TYPE-001]", detail);
}

MetadataResult dtype_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-DTYPE-001]", detail);
}

MetadataResult topology_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-TOPOLOGY-001]", detail);
}

MetadataResult symbol_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-SYMBOL-001]", detail);
}

MetadataResult target_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::malformed, entry, "[LP-TARGET-001]", detail);
}

MetadataResult unsupported_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::unsupported, entry, "[LP-UNSUPPORTED-001]", detail);
}

MetadataResult internal_error(const FamilyDescriptor& entry, const std::string& detail)
{
  return metadata_error(MetadataStatus::internal_error, entry, "[LP-PROVIDER-INTERNAL-001]", detail);
}

MetadataResult resolve_request_target_impl(
  TileIRStringView architecture, const FamilyDescriptor& entry, TargetArchitecture* target_architecture)
{
  if (!target_architecture)
  {
    return internal_error(entry, "null output for target validation");
  }
  if (architecture.len != 0)
  {
    if (!architecture.data)
    {
      return target_error(entry, "compilation context contains an invalid architecture string view");
    }
    *target_architecture = parse_sm_arch(architecture);
    if (!target_architecture->valid())
    {
      return target_error(entry, "compilation context architecture has an invalid spelling");
    }
    return MetadataResult::success();
  }

  constexpr const char* environment_names[] = {
    "CUDA_COOP_TILE_PROVIDER_SM_ARCH",
    "CUDA_ARCH",
  };
  for (const char* const name : environment_names)
  {
    const char* const value = std::getenv(name);
    if (!value || value[0] == '\0')
    {
      continue;
    }
    *target_architecture = parse_sm_arch(TileIRStringView{value, std::strlen(value)});
    if (!target_architecture->valid())
    {
      return target_error(entry, std::string(name) + " has an invalid architecture spelling");
    }
    return MetadataResult::success();
  }
  return target_error(entry, "no target architecture was provided by the compilation context or environment");
}

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

bool is_integral_rank_dtype_suffix(const std::string& dtype_suffix)
{
  return is_integer_dtype_suffix(dtype_suffix);
}

bool string_view_equals(TileIRStringView value, const char* expected)
{
  const size_t expected_size = std::strlen(expected);
  return value.len == expected_size
      && (expected_size == 0 || (value.data && std::memcmp(value.data, expected, expected_size) == 0));
}

std::string collapsed_tileir_dtype_token(std::string token)
{
  if (token.size() >= 2 && token[0] == 'u' && std::isdigit(static_cast<unsigned char>(token[1])))
  {
    token[0] = 'i';
  }
  return token;
}

bool scan_tile_matches(const TileIRForeignType& type, int64_t extent, const std::string& dtype_token)
{
  return type.kind == TILEIR_TYPE_TILE && type.shape && type.num_shape == 1 && type.shape[0].kind == TILEIR_DIM_CONCRETE
      && type.shape[0].value == extent
      && collapsed_tileir_dtype_token(to_string(type.dtype)) == collapsed_tileir_dtype_token(dtype_token);
}

bool rank_zero_tile_matches(const TileIRForeignType& type, const std::string& dtype_token)
{
  return type.kind == TILEIR_TYPE_TILE && type.num_shape == 0
      && collapsed_tileir_dtype_token(to_string(type.dtype)) == collapsed_tileir_dtype_token(dtype_token);
}

inline constexpr size_t scan_expected_constants    = 8;
inline constexpr size_t scan_expected_arguments    = 4;
inline constexpr size_t scan_output_argument_index = 3;

bool scan_decl_matches(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  if (!decl.params || decl.num_params != scan_expected_arguments || !decl.results || decl.num_results != 1)
  {
    return false;
  }
  for (size_t index = 1; index < scan_output_argument_index; ++index)
  {
    if (!rank_zero_tile_matches(decl.params[index], dtype_token))
    {
      return false;
    }
  }
  return scan_tile_matches(decl.params[0], extent, dtype_token)
      && scan_tile_matches(decl.params[scan_output_argument_index], extent, dtype_token)
      && scan_tile_matches(decl.results[0], extent, dtype_token);
}

bool scan_argument_types_match(
  const TileIRForeignType* arguments, size_t num_arguments, int64_t extent, const std::string& dtype_token)
{
  if (!arguments || num_arguments != scan_expected_arguments || !scan_tile_matches(arguments[0], extent, dtype_token)
      || !scan_tile_matches(arguments[scan_output_argument_index], extent, dtype_token))
  {
    return false;
  }
  for (size_t index = 1; index < scan_output_argument_index; ++index)
  {
    if (!rank_zero_tile_matches(arguments[index], dtype_token))
    {
      return false;
    }
  }
  return true;
}

MetadataResult parse_dtype_token(
  const FamilyDescriptor& entry, TileIRDataType typed_dtype, const std::string& dtype_token, std::string* out)
{
  if (!out)
  {
    return internal_error(entry, "null output for dtype-token validation");
  }
  out->clear();
  const std::string typed_token = to_string(typed_dtype);
  auto token_is_supported       = [](const std::string& token) {
    std::string cpp_type;
    std::string extra_include;
    std::string zero_expr;
    return decode_dtype_suffix(token, &cpp_type, &extra_include, &zero_expr);
  };
  if (typed_token.empty())
  {
    if (typed_dtype.kind == TILEIR_DTYPE_INVALID)
    {
      return dtype_error(entry, "collapsed LP dtype is invalid");
    }
    if (token_is_supported(dtype_token))
    {
      return dtype_error(entry,
                         "unknown collapsed LP dtype enum=" + std::to_string(static_cast<int>(typed_dtype.kind))
                           + " contradicts known DTypeToken=" + dtype_token);
    }
    return unsupported_error(
      entry, "collapsed LP dtype enum=" + std::to_string(static_cast<int>(typed_dtype.kind)) + " is not implemented");
  }
  auto token_matches_typed_dtype = [&](const std::string& token) {
    return token_is_supported(token)
        && collapsed_tileir_dtype_token(token) == collapsed_tileir_dtype_token(typed_token);
  };

  if (!token_is_supported(dtype_token))
  {
    return unsupported_error(entry, "DTypeToken=" + dtype_token + " is not implemented");
  }
  if (!token_matches_typed_dtype(dtype_token))
  {
    return dtype_error(entry, "DTypeToken=" + dtype_token + " contradicts collapsed LP dtype=" + typed_token);
  }
  *out = dtype_token;
  return MetadataResult::success();
}

bool declaration_symbol_matches_metadata(
  TileIRStringView decl_symbol, const FamilyDescriptor& entry, bool require_concrete_symbol)
{
  return declaration_symbol_matches_family(entry, decl_symbol.data, decl_symbol.len, require_concrete_symbol);
}

bool reduce_decl_types_match(
  const TileIRForeignDecl& decl, int64_t extent, int64_t output_extent, const std::string& dtype_token)
{
  return decl.params && decl.num_params == 2 && decl.results && decl.num_results == 1
      && scan_tile_matches(decl.params[0], extent, dtype_token)
      && scan_tile_matches(decl.params[1], output_extent, dtype_token)
      && scan_tile_matches(decl.results[0], output_extent, dtype_token);
}

bool reduce_decl_matches(
  const TileIRForeignDecl& decl, int64_t extent, int64_t output_extent, const std::string& dtype_token)
{
  return reduce_decl_types_match(decl, extent, output_extent, dtype_token);
}

bool reduce_operation_is_supported(FamilyId family, const std::string& dtype_token, const std::string& binary_op)
{
  if (family == FamilyId::sum || family == FamilyId::sum_batched)
  {
    return binary_op.empty();
  }
  std::string unused_op_expression;
  if (!resolve_binary_op_token(binary_op, &unused_op_expression))
  {
    return false;
  }
  const bool bitwise = binary_op == "bit_and" || binary_op == "bit_or" || binary_op == "bit_xor";
  return !bitwise || is_integer_dtype_suffix(dtype_token);
}

PrimitiveSpec make_reduce_spec(
  const FamilyDescriptor& entry,
  const std::string& dtype_token,
  const std::string& algorithm,
  const std::string& binary_op,
  int64_t batches)
{
  ReduceSpec spec;
  spec.generic   = entry.family == FamilyId::reduce || entry.family == FamilyId::reduce_batched;
  spec.batches   = batches;
  spec.dtype     = dtype_token;
  spec.algorithm = algorithm;
  spec.binary_op = binary_op;
  return spec;
}

template <class CallView>
MetadataResult make_reduce_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool generic              = entry.family == FamilyId::reduce || entry.family == FamilyId::reduce_batched;
  const bool batched              = entry.family == FamilyId::sum_batched || entry.family == FamilyId::reduce_batched;
  const size_t base_constants     = generic ? 4 : 3;
  const size_t expected_constants = base_constants + 1 + (batched ? 1 : 0);
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::sum && entry.family != FamilyId::reduce && !batched)
  {
    return internal_error(entry, "reduce parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_constants || !call.arguments() || call.argument_count() != 2
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent  = 0;
  int64_t batches = 0;
  TileIRDataType dtype{};
  std::string algorithm;
  std::string binary_op;
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant("T", &dtype)
      || !call.string_constant("Algorithm", &algorithm) || (generic && !call.string_constant("BinaryOp", &binary_op))
      || !call.string_constant("DTypeToken", &explicit_dtype_token)
      || (batched && !call.int_constant("Batches", &batches)))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (batched && (batches <= 0 || extent % batches != 0))
  {
    return constant_error(entry, "Batches must be positive and divide N");
  }
  const int64_t output_extent = batched ? batches : extent;
  if (!reduce_decl_matches(*declaration, extent, output_extent, dtype_token)
      || !scan_tile_matches(call.arguments()[0], extent, dtype_token)
      || !scan_tile_matches(call.arguments()[1], output_extent, dtype_token)
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], output_extent, dtype_token)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (resolve_reduce_algorithm_token(algorithm).empty()
      || !reduce_operation_is_supported(entry.family, dtype_token, binary_op))
  {
    return unsupported_error(entry, "algorithm or reduction operation is not implemented for this dtype");
  }

  return make_logical_request(
    entry.family,
    make_reduce_spec(entry, dtype_token, algorithm, binary_op, batches),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool histogram_decl_types_match(const TileIRForeignDecl& decl, int64_t extent, int64_t bins)
{
  return decl.params && decl.num_params == 2 && decl.results && decl.num_results == 1
      && scan_tile_matches(decl.params[0], extent, "u8") && scan_tile_matches(decl.params[1], bins, "i32")
      && scan_tile_matches(decl.results[0], bins, "i32");
}

bool histogram_decl_matches(const TileIRForeignDecl& decl, int64_t extent, int64_t bins)
{
  return histogram_decl_types_match(decl, extent, bins);
}

MetadataResult histogram_dtype_token_is_supported(
  const FamilyDescriptor& entry, const std::string& explicit_dtype_token, const TileIRForeignType& input_type)
{
  const std::string lowered_dtype = to_string(input_type.dtype);
  if (input_type.kind != TILEIR_TYPE_TILE || collapsed_tileir_dtype_token(lowered_dtype) != "i8")
  {
    return type_error(entry, "Histogram input must be a tile with the collapsed i8 dtype");
  }
  if (explicit_dtype_token != "u8")
  {
    std::string cpp_type;
    std::string extra_include;
    std::string zero_expr;
    if (!decode_dtype_suffix(explicit_dtype_token, &cpp_type, &extra_include, &zero_expr))
    {
      return unsupported_error(entry, "Histogram DTypeToken=" + explicit_dtype_token + " is not implemented");
    }
    return dtype_error(entry, "Histogram DTypeToken=" + explicit_dtype_token + " contradicts the required u8 dtype");
  }
  return MetadataResult::success();
}

PrimitiveSpec make_histogram_spec(const FamilyDescriptor& entry, int64_t bins)
{
  HistogramSpec spec;
  spec.fixed_256 = entry.family == FamilyId::histogram_256;
  spec.bins      = static_cast<int>(bins);
  spec.dtype     = "u8";
  return spec;
}

template <class CallView>
MetadataResult make_histogram_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool fixed_bins                = entry.family == FamilyId::histogram_256;
  const size_t expected_constants      = fixed_bins ? 2 : 3;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::histogram && entry.family != FamilyId::histogram_256)
  {
    return internal_error(entry, "histogram parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_constants || !call.arguments() || call.argument_count() != 2
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent = 0;
  int64_t bins   = fixed_bins ? 256 : 0;
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || (!fixed_bins && !call.int_constant("Bins", &bins))
      || !call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  if (bins <= 0)
  {
    return constant_error(entry, "Bins must be positive");
  }

  const MetadataResult dtype_result =
    histogram_dtype_token_is_supported(entry, explicit_dtype_token, call.arguments()[0]);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }

  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!histogram_decl_matches(*declaration, extent, bins) || !scan_tile_matches(call.arguments()[0], extent, "u8")
      || !scan_tile_matches(call.arguments()[1], bins, "i32")
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], bins, "i32")))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (bins > 256)
  {
    return unsupported_error(entry, "Bins exceeds the provider limit of 256");
  }

  return make_logical_request(
    entry.family, make_histogram_spec(entry, bins), extent, {target_sm, parse_cache_epoch_from_env()}, out);
}

bool run_length_decode_decl_types_match(
  const TileIRForeignDecl& decl,
  int64_t run_extent,
  int64_t decoded_extent,
  const std::string& value_dtype_token,
  const std::string& length_dtype_token,
  bool with_offsets)
{
  const size_t expected_params  = with_offsets ? 5 : 4;
  const size_t expected_results = with_offsets ? 2 : 1;
  return decl.params && decl.num_params == expected_params && decl.results && decl.num_results == expected_results
      && scan_tile_matches(decl.params[0], run_extent, value_dtype_token)
      && scan_tile_matches(decl.params[1], run_extent, length_dtype_token)
      && rank_zero_tile_matches(decl.params[2], length_dtype_token)
      && scan_tile_matches(decl.params[3], decoded_extent, value_dtype_token)
      && (!with_offsets || scan_tile_matches(decl.params[4], decoded_extent, length_dtype_token))
      && scan_tile_matches(decl.results[0], decoded_extent, value_dtype_token)
      && (!with_offsets || scan_tile_matches(decl.results[1], decoded_extent, length_dtype_token));
}

bool run_length_decode_decl_matches(
  const TileIRForeignDecl& decl,
  int64_t run_extent,
  int64_t decoded_extent,
  const std::string& value_dtype_token,
  const std::string& length_dtype_token,
  bool with_offsets)
{
  return run_length_decode_decl_types_match(
    decl, run_extent, decoded_extent, value_dtype_token, length_dtype_token, with_offsets);
}

bool run_length_decode_argument_types_match(
  const TileIRForeignType* arguments,
  size_t num_arguments,
  int64_t run_extent,
  int64_t decoded_extent,
  const std::string& value_dtype_token,
  const std::string& length_dtype_token,
  bool with_offsets)
{
  const size_t expected_arguments = with_offsets ? 5 : 4;
  return arguments && num_arguments == expected_arguments
      && scan_tile_matches(arguments[0], run_extent, value_dtype_token)
      && scan_tile_matches(arguments[1], run_extent, length_dtype_token)
      && rank_zero_tile_matches(arguments[2], length_dtype_token)
      && scan_tile_matches(arguments[3], decoded_extent, value_dtype_token)
      && (!with_offsets || scan_tile_matches(arguments[4], decoded_extent, length_dtype_token));
}

PrimitiveSpec make_run_length_decode_spec(
  const FamilyDescriptor& entry,
  int64_t decoded_extent,
  const std::string& value_dtype_token,
  const std::string& length_dtype_token)
{
  RunLengthDecodeSpec spec;
  spec.with_offsets   = entry.family == FamilyId::run_length_decode_with_offsets;
  spec.value_dtype    = value_dtype_token;
  spec.length_dtype   = length_dtype_token;
  spec.decoded_extent = decoded_extent;
  return spec;
}

template <class CallView>
MetadataResult make_run_length_decode_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool with_offsets              = entry.family == FamilyId::run_length_decode_with_offsets;
  const size_t expected_arguments      = with_offsets ? 5 : 4;
  const size_t expected_results        = with_offsets ? 2 : 1;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::run_length_decode && entry.family != FamilyId::run_length_decode_with_offsets)
  {
    return internal_error(entry, "run-length-decode parser was paired with an incompatible family descriptor");
  }
  if (!declaration || call.constant_count() != 6 || !call.arguments() || call.argument_count() != expected_arguments
      || (!CallView::is_resolved && (!call.results() || call.result_count() != expected_results)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t run_extent     = 0;
  int64_t decoded_extent = 0;
  TileIRDataType value_dtype{};
  TileIRDataType length_dtype{};
  std::string explicit_value_dtype_token;
  std::string explicit_length_dtype_token;
  if (!call.int_constant("RunN", &run_extent) || !call.int_constant("DecodedN", &decoded_extent)
      || !call.dtype_constant("ItemT", &value_dtype) || !call.dtype_constant("LengthT", &length_dtype)
      || !call.string_constant("ItemDTypeToken", &explicit_value_dtype_token)
      || !call.string_constant("LengthDTypeToken", &explicit_length_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  std::string value_dtype_token;
  std::string length_dtype_token;

  const MetadataResult value_dtype_result =
    parse_dtype_token(entry, value_dtype, explicit_value_dtype_token, &value_dtype_token);
  if (!value_dtype_result.is_supported())
  {
    return value_dtype_result;
  }
  const MetadataResult length_dtype_result =
    parse_dtype_token(entry, length_dtype, explicit_length_dtype_token, &length_dtype_token);
  if (!length_dtype_result.is_supported())
  {
    return length_dtype_result;
  }
  const auto topology = validate_logical_extents(entry.family, run_extent, decoded_extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol contradicts the selected family");
  }
  if (!run_length_decode_decl_matches(
        *declaration, run_extent, decoded_extent, value_dtype_token, length_dtype_token, with_offsets)
      || !run_length_decode_argument_types_match(
        call.arguments(),
        call.argument_count(),
        run_extent,
        decoded_extent,
        value_dtype_token,
        length_dtype_token,
        with_offsets)
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], decoded_extent, value_dtype_token))
      || (!CallView::is_resolved && with_offsets
          && !scan_tile_matches(call.results()[1], decoded_extent, length_dtype_token)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (length_dtype_token != "u32" && length_dtype_token != "u64")
  {
    return unsupported_error(entry, "length dtype must be u32 or u64");
  }

  return make_logical_request(
    entry.family,
    make_run_length_decode_spec(entry, decoded_extent, value_dtype_token, length_dtype_token),
    run_extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool discontinuity_decl_types_match(
  const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token, bool heads_and_tails)
{
  const size_t output_count = heads_and_tails ? 2 : 1;
  if (!decl.params || decl.num_params != output_count + 1 || !decl.results || decl.num_results != output_count
      || !scan_tile_matches(decl.params[0], extent, dtype_token))
  {
    return false;
  }
  for (size_t index = 0; index < output_count; ++index)
  {
    if (!scan_tile_matches(decl.params[index + 1], extent, "i32")
        || !scan_tile_matches(decl.results[index], extent, "i32"))
    {
      return false;
    }
  }
  return true;
}

bool discontinuity_decl_matches(
  const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token, bool heads_and_tails)
{
  return discontinuity_decl_types_match(decl, extent, dtype_token, heads_and_tails);
}

bool discontinuity_mode_is_supported(FamilyId family, const std::string& mode)
{
  if (family == FamilyId::discontinuity_flags)
  {
    return mode == "heads" || mode == "tails";
  }
  return family == FamilyId::discontinuity_heads_and_tails && mode == "heads_and_tails";
}

PrimitiveSpec
make_discontinuity_spec(const FamilyDescriptor& entry, const std::string& dtype_token, const std::string& mode)
{
  DiscontinuitySpec spec;
  spec.heads_and_tails = entry.family == FamilyId::discontinuity_heads_and_tails;
  spec.mode            = mode;
  spec.dtype           = dtype_token;
  return spec;
}

template <class CallView>
MetadataResult make_discontinuity_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool heads_and_tails           = entry.family == FamilyId::discontinuity_heads_and_tails;
  const size_t output_count            = heads_and_tails ? 2 : 1;
  const size_t expected_constants      = 4;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::discontinuity_flags && entry.family != FamilyId::discontinuity_heads_and_tails)
  {
    return internal_error(entry, "discontinuity parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_constants || !call.arguments()
      || call.argument_count() != output_count + 1
      || (!CallView::is_resolved && (!call.results() || call.result_count() != output_count)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent = 0;
  TileIRDataType dtype{};
  std::string mode;
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant("T", &dtype) || !call.string_constant("Mode", &mode)
      || !call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  if (!discontinuity_mode_is_supported(entry.family, mode))
  {
    if (mode == "heads" || mode == "tails" || mode == "heads_and_tails")
    {
      return symbol_error(entry, "Mode contradicts the selected discontinuity family");
    }
  }

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!discontinuity_decl_matches(*declaration, extent, dtype_token, heads_and_tails)
      || !scan_tile_matches(call.arguments()[0], extent, dtype_token))
  {
    return type_error(entry, "declaration or input argument contradicts the resolved metadata");
  }
  for (size_t index = 0; index < output_count; ++index)
  {
    if (!scan_tile_matches(call.arguments()[index + 1], extent, "i32")
        || (!CallView::is_resolved && !scan_tile_matches(call.results()[index], extent, "i32")))
    {
      return type_error(entry, "output argument or result contradicts the resolved metadata");
    }
  }

  if (!discontinuity_mode_is_supported(entry.family, mode))
  {
    return unsupported_error(entry, "Mode=" + mode + " is not implemented");
  }

  return make_logical_request(
    entry.family,
    make_discontinuity_spec(entry, dtype_token, mode),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool adjacent_difference_decl_types_match(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return decl.params && decl.num_params == 2 && decl.results && decl.num_results == 1
      && scan_tile_matches(decl.params[0], extent, dtype_token)
      && scan_tile_matches(decl.params[1], extent, dtype_token)
      && scan_tile_matches(decl.results[0], extent, dtype_token);
}

bool adjacent_difference_decl_matches(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return adjacent_difference_decl_types_match(decl, extent, dtype_token);
}

bool adjacent_difference_mode_is_supported(const std::string& mode)
{
  return mode == "subtract_left" || mode == "subtract_right";
}

PrimitiveSpec
make_adjacent_difference_spec(const FamilyDescriptor& entry, const std::string& dtype_token, const std::string& mode)
{
  static_cast<void>(entry);
  AdjacentDifferenceSpec spec;
  spec.mode  = mode;
  spec.dtype = dtype_token;
  return spec;
}

template <class CallView>
MetadataResult make_adjacent_difference_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const size_t expected_constants      = 4;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::adjacent_difference)
  {
    return internal_error(entry, "adjacent-difference parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_constants || !call.arguments() || call.argument_count() != 2
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent = 0;
  TileIRDataType dtype{};
  std::string mode;
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant("T", &dtype) || !call.string_constant("Mode", &mode)
      || !call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const bool declaration_matches = adjacent_difference_decl_matches(*declaration, extent, dtype_token);
  const auto topology            = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!declaration_matches || !scan_tile_matches(call.arguments()[0], extent, dtype_token)
      || !scan_tile_matches(call.arguments()[1], extent, dtype_token)
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], extent, dtype_token)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (!adjacent_difference_mode_is_supported(mode))
  {
    return unsupported_error(entry, "Mode=" + mode + " is not implemented");
  }

  return make_logical_request(
    entry.family,
    make_adjacent_difference_spec(entry, dtype_token, mode),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool shuffle_decl_types_match(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return decl.params && decl.num_params == 3 && decl.results && decl.num_results == 1
      && scan_tile_matches(decl.params[0], extent, dtype_token) && rank_zero_tile_matches(decl.params[1], "i32")
      && scan_tile_matches(decl.params[2], extent, dtype_token)
      && scan_tile_matches(decl.results[0], extent, dtype_token);
}

bool shuffle_decl_matches(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return shuffle_decl_types_match(decl, extent, dtype_token);
}

bool shuffle_mode_is_supported(const std::string& mode)
{
  return mode == "up" || mode == "down" || mode == "offset" || mode == "rotate";
}

PrimitiveSpec make_shuffle_spec(const std::string& dtype_token, const std::string& mode)
{
  ShuffleSpec spec;
  spec.mode  = mode;
  spec.dtype = dtype_token;
  return spec;
}

template <class CallView>
MetadataResult make_shuffle_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  constexpr size_t expected_arguments  = 3;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::shuffle)
  {
    return internal_error(entry, "shuffle parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != 4 || !call.arguments() || call.argument_count() != expected_arguments
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }
  constexpr size_t output_index = 2;

  int64_t extent = 0;
  TileIRDataType dtype{};
  std::string mode;
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant("T", &dtype) || !call.string_constant("Mode", &mode)
      || !call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!shuffle_decl_matches(*declaration, extent, dtype_token)
      || !scan_tile_matches(call.arguments()[0], extent, dtype_token)
      || !rank_zero_tile_matches(call.arguments()[1], "i32")
      || !scan_tile_matches(call.arguments()[output_index], extent, dtype_token)
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], extent, dtype_token)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (!shuffle_mode_is_supported(mode))
  {
    return unsupported_error(entry, "Mode=" + mode + " is not implemented");
  }

  return make_logical_request(
    entry.family, make_shuffle_spec(dtype_token, mode), extent, {target_sm, parse_cache_epoch_from_env()}, out);
}

bool exchange_decl_types_match(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return decl.params && decl.num_params == 1 && decl.results && decl.num_results == 1
      && scan_tile_matches(decl.params[0], extent, dtype_token)
      && scan_tile_matches(decl.results[0], extent, dtype_token);
}

bool exchange_decl_matches(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return exchange_decl_types_match(decl, extent, dtype_token);
}

bool exchange_mode_is_supported(const std::string& mode)
{
  return mode == "blocked_to_striped" || mode == "striped_to_blocked";
}

PrimitiveSpec make_exchange_spec(
  const FamilyDescriptor& entry, const std::string& dtype_token, const std::string& mode, int64_t block_threads)
{
  static_cast<void>(entry);
  ExchangeSpec spec;
  spec.block_threads = block_threads;
  spec.mode          = mode;
  spec.dtype         = dtype_token;
  return spec;
}

template <class CallView>
MetadataResult make_exchange_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::exchange)
  {
    return internal_error(entry, "exchange parser was paired with an incompatible family descriptor");
  }
  if (!declaration || call.constant_count() != 5 || !call.arguments() || call.argument_count() != 1
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent         = 0;
  int64_t exchange_width = 0;
  TileIRDataType dtype{};
  std::string mode;
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant("T", &dtype)
      || !call.int_constant("BlockThreads", &exchange_width) || !call.string_constant("Mode", &mode)
      || !call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }

  if (exchange_width <= 0 || extent <= 0 || extent % exchange_width != 0)
  {
    return topology_error(
      entry, "exchange requires positive BlockThreads dividing N because topology determines the permutation");
  }

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol contradicts the selected family");
  }
  if (!exchange_decl_matches(*declaration, extent, dtype_token)
      || !scan_tile_matches(call.arguments()[0], extent, dtype_token)
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], extent, dtype_token)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (!exchange_mode_is_supported(mode))
  {
    return unsupported_error(entry, "Mode=" + mode + " is not implemented");
  }

  return make_logical_request(
    entry.family,
    make_exchange_spec(entry, dtype_token, mode, exchange_width),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool radix_rank_argument_types_match(
  const TileIRForeignType* arguments, size_t num_arguments, int64_t extent, const std::string& dtype_token)
{
  return arguments && num_arguments == 4 && scan_tile_matches(arguments[0], extent, dtype_token)
      && rank_zero_tile_matches(arguments[1], "i32") && rank_zero_tile_matches(arguments[2], "i32")
      && scan_tile_matches(arguments[3], extent, "i32");
}

bool radix_rank_result_types_match(const TileIRForeignType* results, size_t num_results, int64_t extent)
{
  return results && num_results == 1 && scan_tile_matches(results[0], extent, "i32");
}

bool radix_rank_decl_types_match(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return radix_rank_argument_types_match(decl.params, decl.num_params, extent, dtype_token)
      && radix_rank_result_types_match(decl.results, decl.num_results, extent);
}

bool radix_rank_decl_matches(const TileIRForeignDecl& decl, int64_t extent, const std::string& dtype_token)
{
  return radix_rank_decl_types_match(decl, extent, dtype_token);
}

bool radix_rank_tuning_is_supported(
  int64_t extent,
  int64_t radix_bits,
  int64_t descending,
  int64_t memoize_outer_scan,
  const std::string& inner_scan_algorithm,
  const std::string& smem_config)
{
  return extent > 0 && extent <= std::numeric_limits<unsigned short>::max() && radix_bits >= 1 && radix_bits <= 16
      && (descending == 0 || descending == 1) && (memoize_outer_scan == 0 || memoize_outer_scan == 1)
      && !resolve_scan_algorithm_token(inner_scan_algorithm).empty()
      && (smem_config == "four_byte" || smem_config == "eight_byte");
}

PrimitiveSpec make_radix_rank_spec(
  int64_t radix_bits,
  int64_t descending,
  int64_t memoize_outer_scan,
  const std::string& inner_scan_algorithm,
  const std::string& smem_config,
  const std::string& dtype_token)
{
  RadixRankSpec spec;
  spec.dtype                = dtype_token;
  spec.descending           = descending != 0;
  spec.radix_bits           = static_cast<int>(radix_bits);
  spec.memoize_outer_scan   = memoize_outer_scan != 0;
  spec.inner_scan_algorithm = inner_scan_algorithm;
  spec.smem_config          = smem_config;
  return spec;
}

template <class CallView>
MetadataResult make_radix_rank_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  constexpr size_t expected_constants  = 8;
  constexpr size_t expected_arguments  = 4;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::radix_rank)
  {
    return internal_error(entry, "radix-rank parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_constants || !call.arguments()
      || call.argument_count() != expected_arguments
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent             = 0;
  int64_t radix_bits         = 0;
  int64_t descending         = 0;
  int64_t memoize_outer_scan = 1;
  TileIRDataType dtype{};
  std::string inner_scan_algorithm = "warp_scans";
  std::string smem_config          = "four_byte";
  std::string explicit_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant("T", &dtype)
      || !call.int_constant("RadixBits", &radix_bits) || !call.int_constant("Descending", &descending)
      || !call.int_constant("MemoizeOuterScan", &memoize_outer_scan)
      || !call.string_constant("InnerScanAlgorithm", &inner_scan_algorithm)
      || !call.string_constant("SmemConfig", &smem_config)
      || !call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }
  if (extent <= 0)
  {
    return constant_error(entry, "N must be positive");
  }
  if (descending != 0 && descending != 1)
  {
    return constant_error(entry, "Descending must be boolean");
  }
  if (memoize_outer_scan != 0 && memoize_outer_scan != 1)
  {
    return constant_error(entry, "MemoizeOuterScan must be boolean");
  }
  const bool tuning_supported = radix_rank_tuning_is_supported(
    extent, radix_bits, descending, memoize_outer_scan, inner_scan_algorithm, smem_config);

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const bool integral_dtype = is_integral_rank_dtype_suffix(dtype_token);
  const auto topology       = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!radix_rank_decl_matches(*declaration, extent, dtype_token)
      || !radix_rank_argument_types_match(call.arguments(), call.argument_count(), extent, dtype_token)
      || (!CallView::is_resolved && !radix_rank_result_types_match(call.results(), call.result_count(), extent)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (!integral_dtype)
  {
    return unsupported_error(entry, "radix rank requires an integral key dtype");
  }
  if (!tuning_supported)
  {
    return unsupported_error(entry,
                             "radix-rank capacity, algorithm, radix width, or shared-memory tuning is unavailable");
  }

  return make_logical_request(
    entry.family,
    make_radix_rank_spec(radix_bits, descending, memoize_outer_scan, inner_scan_algorithm, smem_config, dtype_token),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

PrimitiveSpec make_scan_spec(
  const std::string& dtype_token,
  const std::string& mode,
  const std::string& algorithm,
  const std::string& scan_op,
  int64_t initial_present,
  int64_t prefix_present)
{
  ScanSpec spec;
  spec.generic         = scan_op != "sum" || initial_present != 0;
  spec.dtype           = dtype_token;
  spec.mode            = mode;
  spec.algorithm       = algorithm;
  spec.scan_op         = scan_op;
  spec.initial_present = initial_present != 0;
  spec.prefix_present  = prefix_present != 0;
  return spec;
}

bool scan_mode_and_presence_are_supported(
  const std::string& mode, const std::string& scan_op, int64_t initial_present, int64_t prefix_present)
{
  if ((initial_present != 0 && initial_present != 1) || (prefix_present != 0 && prefix_present != 1)
      || (initial_present != 0 && prefix_present != 0))
  {
    return false;
  }
  if (mode == "inclusive")
  {
    return initial_present == 0;
  }
  if (mode != "exclusive")
  {
    return false;
  }
  return scan_op == "sum" || initial_present != 0 || prefix_present != 0;
}

template <class CallView>
MetadataResult make_scan_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  constexpr size_t expected_constants  = scan_expected_constants;
  constexpr size_t expected_arguments  = scan_expected_arguments;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::scan)
  {
    return internal_error(entry, "scan parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_constants || !call.arguments()
      || call.argument_count() != expected_arguments
      || (!CallView::is_resolved && (!call.results() || call.result_count() != 1)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t n               = 0;
  int64_t initial_present = 0;
  int64_t prefix_present  = 0;
  std::string mode;
  std::string algorithm;
  std::string scan_op;
  std::string explicit_dtype_token;
  TileIRDataType dtype{};
  const bool common_constants_match =
    call.int_constant("N", &n) && call.dtype_constant("T", &dtype) && call.string_constant("Mode", &mode)
    && call.string_constant("Algorithm", &algorithm) && call.string_constant("ScanOp", &scan_op)
    && call.int_constant("InitialPresent", &initial_present) && call.int_constant("PrefixPresent", &prefix_present);
  if (!common_constants_match)
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }
  if (!call.string_constant("DTypeToken", &explicit_dtype_token))
  {
    return constant_error(entry, "DTypeToken does not match its declared kind");
  }

  std::string dtype_token;
  const MetadataResult dtype_result = parse_dtype_token(entry, dtype, explicit_dtype_token, &dtype_token);
  if (!dtype_result.is_supported())
  {
    return dtype_result;
  }
  const bool mode_supported      = mode == "inclusive" || mode == "exclusive";
  const bool algorithm_supported = !resolve_scan_algorithm_token(algorithm).empty();
  const bool scan_op_supported =
    scan_op == "sum" || reduce_operation_is_supported(FamilyId::reduce, dtype_token, scan_op);
  if ((initial_present != 0 && initial_present != 1) || (prefix_present != 0 && prefix_present != 1))
  {
    return constant_error(entry, "InitialPresent and PrefixPresent must be boolean");
  }
  if (mode_supported && scan_op_supported
      && !scan_mode_and_presence_are_supported(mode, scan_op, initial_present, prefix_present))
  {
    return constant_error(entry, "scan mode, initial value, and prefix value are inconsistent");
  }
  const auto topology = validate_logical_extents(entry.family, n);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!scan_decl_matches(*declaration, n, dtype_token)
      || !scan_argument_types_match(call.arguments(), call.argument_count(), n, dtype_token)
      || (!CallView::is_resolved && !scan_tile_matches(call.results()[0], n, dtype_token)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (!mode_supported)
  {
    return unsupported_error(entry, "Mode=" + mode + " is not implemented");
  }
  if (!algorithm_supported)
  {
    return unsupported_error(entry, "Algorithm=" + algorithm + " is not implemented");
  }
  if (!scan_op_supported)
  {
    return unsupported_error(entry, "ScanOp=" + scan_op + " is not implemented for this dtype");
  }

  return make_logical_request(
    entry.family,
    make_scan_spec(dtype_token, mode, algorithm, scan_op, initial_present, prefix_present),
    n,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool merge_sort_scalar_tile_matches(const TileIRForeignType& type, const std::string& dtype_token)
{
  if (type.kind != TILEIR_TYPE_TILE
      || collapsed_tileir_dtype_token(to_string(type.dtype)) != collapsed_tileir_dtype_token(dtype_token))
  {
    return false;
  }
  return type.num_shape == 0;
}

bool merge_sort_argument_types_match(
  const TileIRForeignType* arguments,
  size_t num_arguments,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  const size_t expected_arguments = pairs ? 6 : 4;
  if (!arguments || num_arguments != expected_arguments || !scan_tile_matches(arguments[0], extent, key_dtype_token))
  {
    return false;
  }
  const size_t scalar_begin = pairs ? 2 : 1;
  if (pairs && !scan_tile_matches(arguments[1], extent, value_dtype_token))
  {
    return false;
  }
  if (!merge_sort_scalar_tile_matches(arguments[scalar_begin], "i32")
      || !merge_sort_scalar_tile_matches(arguments[scalar_begin + 1], key_dtype_token))
  {
    return false;
  }
  const size_t output_begin = scalar_begin + 2;
  return scan_tile_matches(arguments[output_begin], extent, key_dtype_token)
      && (!pairs || scan_tile_matches(arguments[output_begin + 1], extent, value_dtype_token));
}

bool merge_sort_result_types_match(
  const TileIRForeignType* results,
  size_t num_results,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  const size_t expected_results = pairs ? 2 : 1;
  return results && num_results == expected_results && scan_tile_matches(results[0], extent, key_dtype_token)
      && (!pairs || scan_tile_matches(results[1], extent, value_dtype_token));
}

bool merge_sort_decl_types_match(
  const TileIRForeignDecl& decl,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  return merge_sort_argument_types_match(decl.params, decl.num_params, extent, key_dtype_token, value_dtype_token, pairs)
      && merge_sort_result_types_match(
           decl.results, decl.num_results, extent, key_dtype_token, value_dtype_token, pairs);
}

bool merge_sort_decl_matches(
  const TileIRForeignDecl& decl,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  return merge_sort_decl_types_match(decl, extent, key_dtype_token, value_dtype_token, pairs);
}

PrimitiveSpec make_merge_sort_spec(
  const FamilyDescriptor& entry,
  int64_t descending,
  int64_t partial_tile,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token)
{
  MergeSortSpec spec;
  spec.pairs        = entry.family == FamilyId::merge_sort_pairs;
  spec.key_dtype    = key_dtype_token;
  spec.value_dtype  = value_dtype_token;
  spec.descending   = descending != 0;
  spec.partial_tile = partial_tile != 0;
  return spec;
}

template <class CallView>
MetadataResult make_merge_sort_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool pairs                     = entry.family == FamilyId::merge_sort_pairs;
  const size_t base_consts             = pairs ? 5 : 4;
  const size_t expected_consts         = base_consts + (pairs ? 2 : 1);
  const size_t expected_args           = pairs ? 6 : 4;
  const size_t expected_results        = pairs ? 2 : 1;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::merge_sort_keys && entry.family != FamilyId::merge_sort_pairs)
  {
    return internal_error(entry, "merge-sort parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_consts || !call.arguments()
      || call.argument_count() != expected_args
      || (!CallView::is_resolved && (!call.results() || call.result_count() != expected_results)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent       = 0;
  int64_t descending   = 0;
  int64_t partial_tile = 0;
  TileIRDataType key_dtype{};
  TileIRDataType value_dtype{};
  std::string explicit_key_dtype_token;
  std::string explicit_value_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant(pairs ? "KeyT" : "T", &key_dtype)
      || (pairs && !call.dtype_constant("ValueT", &value_dtype)) || !call.int_constant("Descending", &descending)
      || !call.int_constant("PartialTile", &partial_tile)
      || !call.string_constant(pairs ? "KeyDTypeToken" : "DTypeToken", &explicit_key_dtype_token)
      || (pairs && !call.string_constant("ValueDTypeToken", &explicit_value_dtype_token)))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }
  if ((descending != 0 && descending != 1) || (partial_tile != 0 && partial_tile != 1))
  {
    return constant_error(entry, "Descending and PartialTile must be boolean");
  }

  std::string key_dtype_token;
  std::string value_dtype_token;
  const MetadataResult key_dtype_result =
    parse_dtype_token(entry, key_dtype, explicit_key_dtype_token, &key_dtype_token);
  if (!key_dtype_result.is_supported())
  {
    return key_dtype_result;
  }
  if (pairs)
  {
    const MetadataResult value_dtype_result =
      parse_dtype_token(entry, value_dtype, explicit_value_dtype_token, &value_dtype_token);
    if (!value_dtype_result.is_supported())
    {
      return value_dtype_result;
    }
  }

  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!merge_sort_decl_matches(*declaration, extent, key_dtype_token, value_dtype_token, pairs)
      || !merge_sort_argument_types_match(
        call.arguments(), call.argument_count(), extent, key_dtype_token, value_dtype_token, pairs)
      || (!CallView::is_resolved
          && !merge_sort_result_types_match(
            call.results(), call.result_count(), extent, key_dtype_token, value_dtype_token, pairs)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  return make_logical_request(
    entry.family,
    make_merge_sort_spec(entry, descending, partial_tile, key_dtype_token, value_dtype_token),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool radix_sort_scalar_tile_matches(const TileIRForeignType& type)
{
  if (type.kind != TILEIR_TYPE_TILE || collapsed_tileir_dtype_token(to_string(type.dtype)) != "i32")
  {
    return false;
  }
  return type.num_shape == 0;
}

bool radix_sort_argument_types_match(
  const TileIRForeignType* arguments,
  size_t num_arguments,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  const size_t expected_arguments = pairs ? 6 : 4;
  if (!arguments || num_arguments != expected_arguments || !scan_tile_matches(arguments[0], extent, key_dtype_token))
  {
    return false;
  }
  const size_t scalar_begin = pairs ? 2 : 1;
  if (pairs && !scan_tile_matches(arguments[1], extent, value_dtype_token))
  {
    return false;
  }
  if (!radix_sort_scalar_tile_matches(arguments[scalar_begin])
      || !radix_sort_scalar_tile_matches(arguments[scalar_begin + 1]))
  {
    return false;
  }
  const size_t output_begin = scalar_begin + 2;
  return scan_tile_matches(arguments[output_begin], extent, key_dtype_token)
      && (!pairs || scan_tile_matches(arguments[output_begin + 1], extent, value_dtype_token));
}

bool radix_sort_result_types_match(
  const TileIRForeignType* results,
  size_t num_results,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  const size_t expected_results = pairs ? 2 : 1;
  return results && num_results == expected_results && scan_tile_matches(results[0], extent, key_dtype_token)
      && (!pairs || scan_tile_matches(results[1], extent, value_dtype_token));
}

bool radix_sort_decl_types_match(
  const TileIRForeignDecl& decl,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  return radix_sort_argument_types_match(decl.params, decl.num_params, extent, key_dtype_token, value_dtype_token, pairs)
      && radix_sort_result_types_match(
           decl.results, decl.num_results, extent, key_dtype_token, value_dtype_token, pairs);
}

bool radix_sort_decl_matches(
  const TileIRForeignDecl& decl,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  return radix_sort_decl_types_match(decl, extent, key_dtype_token, value_dtype_token, pairs);
}

bool radix_sort_tuning_is_supported(
  int64_t descending,
  int64_t radix_bits,
  int64_t memoize_outer_scan,
  const std::string& inner_scan_algorithm,
  const std::string& smem_config)
{
  return (descending == 0 || descending == 1) && radix_bits >= 1 && radix_bits <= kMaxRadixSortBits
      && (memoize_outer_scan == 0 || memoize_outer_scan == 1)
      && !resolve_scan_algorithm_token(inner_scan_algorithm).empty() && !resolve_smem_config_token(smem_config).empty();
}

PrimitiveSpec make_radix_sort_spec(
  const FamilyDescriptor& entry,
  int64_t descending,
  int64_t radix_bits,
  int64_t memoize_outer_scan,
  const std::string& inner_scan_algorithm,
  const std::string& smem_config,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token)
{
  RadixSortSpec spec;
  spec.pairs                = entry.family == FamilyId::radix_sort_pairs;
  spec.key_dtype            = key_dtype_token;
  spec.value_dtype          = value_dtype_token;
  spec.descending           = descending != 0;
  spec.radix_bits           = static_cast<int>(radix_bits);
  spec.memoize_outer_scan   = memoize_outer_scan != 0;
  spec.inner_scan_algorithm = inner_scan_algorithm;
  spec.smem_config          = smem_config;
  return spec;
}

template <class CallView>
MetadataResult make_radix_sort_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool pairs                     = entry.family == FamilyId::radix_sort_pairs;
  const size_t base_consts             = pairs ? 8 : 7;
  const size_t expected_consts         = base_consts + (pairs ? 2 : 1);
  const size_t expected_args           = pairs ? 6 : 4;
  const size_t expected_results        = pairs ? 2 : 1;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::radix_sort_keys && entry.family != FamilyId::radix_sort_pairs)
  {
    return internal_error(entry, "radix-sort parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_consts || !call.arguments()
      || call.argument_count() != expected_args
      || (!CallView::is_resolved && (!call.results() || call.result_count() != expected_results)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent             = 0;
  int64_t descending         = 0;
  int64_t radix_bits         = 0;
  int64_t memoize_outer_scan = 0;
  std::string inner_scan_algorithm;
  std::string smem_config;
  std::string explicit_key_dtype_token;
  std::string explicit_value_dtype_token;
  if (!call.int_constant("N", &extent) || !call.int_constant("Descending", &descending)
      || !call.int_constant("RadixBits", &radix_bits) || !call.int_constant("MemoizeOuterScan", &memoize_outer_scan)
      || !call.string_constant("InnerScanAlgorithm", &inner_scan_algorithm)
      || !call.string_constant("SmemConfig", &smem_config)
      || !call.string_constant(pairs ? "KeyDTypeToken" : "DTypeToken", &explicit_key_dtype_token)
      || (pairs && !call.string_constant("ValueDTypeToken", &explicit_value_dtype_token)))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }
  if (descending != 0 && descending != 1)
  {
    return constant_error(entry, "Descending must be boolean");
  }
  if (memoize_outer_scan != 0 && memoize_outer_scan != 1)
  {
    return constant_error(entry, "MemoizeOuterScan must be boolean");
  }
  const bool tuning_supported =
    radix_sort_tuning_is_supported(descending, radix_bits, memoize_outer_scan, inner_scan_algorithm, smem_config);
  if (extent > 65535)
  {
    return topology_error(entry, "Radix Sort N must be at most 65535");
  }

  TileIRDataType key_dtype{};
  TileIRDataType value_dtype{};
  if (!call.dtype_constant(pairs ? "KeyT" : "T", &key_dtype) || (pairs && !call.dtype_constant("ValueT", &value_dtype)))
  {
    return constant_error(entry, "key or value dtype does not match its declared kind");
  }
  std::string key_dtype_token;
  std::string value_dtype_token;
  const MetadataResult key_dtype_result =
    parse_dtype_token(entry, key_dtype, explicit_key_dtype_token, &key_dtype_token);
  if (!key_dtype_result.is_supported())
  {
    return key_dtype_result;
  }
  if (pairs)
  {
    const MetadataResult value_dtype_result =
      parse_dtype_token(entry, value_dtype, explicit_value_dtype_token, &value_dtype_token);
    if (!value_dtype_result.is_supported())
    {
      return value_dtype_result;
    }
  }
  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!radix_sort_decl_matches(*declaration, extent, key_dtype_token, value_dtype_token, pairs)
      || !radix_sort_argument_types_match(
        call.arguments(), call.argument_count(), extent, key_dtype_token, value_dtype_token, pairs)
      || (!CallView::is_resolved
          && !radix_sort_result_types_match(
            call.results(), call.result_count(), extent, key_dtype_token, value_dtype_token, pairs)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  if (!tuning_supported)
  {
    return unsupported_error(entry, "radix width, scan algorithm, or shared-memory tuning is unavailable");
  }

  return make_logical_request(
    entry.family,
    make_radix_sort_spec(
      entry,
      descending,
      radix_bits,
      memoize_outer_scan,
      inner_scan_algorithm,
      smem_config,
      key_dtype_token,
      value_dtype_token),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

bool topk_scalar_tile_matches(const TileIRForeignType& type)
{
  if (type.kind != TILEIR_TYPE_TILE || to_string(type.dtype) != "i32")
  {
    return false;
  }
  return type.num_shape == 0;
}

bool topk_argument_types_match(
  const TileIRForeignType* arguments,
  size_t num_arguments,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  const size_t expected_arguments = pairs ? 8 : 6;
  if (!arguments || num_arguments != expected_arguments || !scan_tile_matches(arguments[0], extent, key_dtype_token))
  {
    return false;
  }
  const size_t scalar_begin = pairs ? 2 : 1;
  const size_t scalar_end   = scalar_begin + 4;
  if (pairs && !scan_tile_matches(arguments[1], extent, value_dtype_token))
  {
    return false;
  }
  for (size_t i = scalar_begin; i < scalar_end; ++i)
  {
    if (!topk_scalar_tile_matches(arguments[i]))
    {
      return false;
    }
  }
  const size_t output_begin = scalar_end;
  return scan_tile_matches(arguments[output_begin], extent, key_dtype_token)
      && (!pairs || scan_tile_matches(arguments[output_begin + 1], extent, value_dtype_token));
}

bool topk_result_types_match(
  const TileIRForeignType* results,
  size_t num_results,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  const size_t expected_results = pairs ? 2 : 1;
  return results && num_results == expected_results && scan_tile_matches(results[0], extent, key_dtype_token)
      && (!pairs || scan_tile_matches(results[1], extent, value_dtype_token));
}

bool topk_decl_types_match(
  const TileIRForeignDecl& decl,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  return topk_argument_types_match(decl.params, decl.num_params, extent, key_dtype_token, value_dtype_token, pairs)
      && topk_result_types_match(decl.results, decl.num_results, extent, key_dtype_token, value_dtype_token, pairs);
}

bool topk_decl_matches(
  const TileIRForeignDecl& decl,
  int64_t extent,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token,
  bool pairs)
{
  return topk_decl_types_match(decl, extent, key_dtype_token, value_dtype_token, pairs);
}

PrimitiveSpec make_topk_spec(
  const FamilyDescriptor& entry,
  int64_t descending,
  int64_t full_tile,
  const std::string& key_dtype_token,
  const std::string& value_dtype_token)
{
  TopKSpec spec;
  spec.pairs       = entry.family == FamilyId::topk_pairs;
  spec.key_dtype   = key_dtype_token;
  spec.value_dtype = value_dtype_token;
  spec.descending  = descending != 0;
  spec.full_tile   = full_tile != 0;
  return spec;
}

template <class CallView>
MetadataResult make_topk_metadata(
  const CallView& call, const FamilyDescriptor& entry, TargetArchitecture target_sm, std::optional<LogicalRequest>* out)
{
  const bool pairs                     = entry.family == FamilyId::topk_pairs;
  const size_t base_consts             = pairs ? 5 : 4;
  const size_t expected_consts         = base_consts + (pairs ? 2 : 1);
  const size_t expected_args           = pairs ? 8 : 6;
  const size_t expected_results        = pairs ? 2 : 1;
  const TileIRForeignDecl* declaration = call.declaration();
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  if (entry.family != FamilyId::topk_keys && entry.family != FamilyId::topk_pairs)
  {
    return internal_error(entry, "TopK parser was paired with a different family");
  }
  if (!declaration || call.constant_count() != expected_consts || !call.arguments()
      || call.argument_count() != expected_args
      || (!CallView::is_resolved && (!call.results() || call.result_count() != expected_results)))
  {
    return schema_error(entry, "foreign declaration, constant, argument, or result count does not match the ABI");
  }

  int64_t extent     = 0;
  int64_t descending = 0;
  int64_t full_tile  = 0;
  TileIRDataType key_dtype{};
  TileIRDataType value_dtype{};
  std::string explicit_key_dtype_token;
  std::string explicit_value_dtype_token;
  if (!call.int_constant("N", &extent) || !call.dtype_constant(pairs ? "KeyT" : "T", &key_dtype)
      || (pairs && !call.dtype_constant("ValueT", &value_dtype)) || !call.int_constant("Descending", &descending)
      || !call.int_constant("FullTile", &full_tile)
      || !call.string_constant(pairs ? "KeyDTypeToken" : "DTypeToken", &explicit_key_dtype_token)
      || (pairs && !call.string_constant("ValueDTypeToken", &explicit_value_dtype_token)))
  {
    return constant_error(entry, "constant values do not match their declared kinds");
  }
  if ((descending != 0 && descending != 1) || (full_tile != 0 && full_tile != 1))
  {
    return constant_error(entry, "Descending and FullTile must be boolean");
  }

  std::string key_dtype_token;
  std::string value_dtype_token;
  const MetadataResult key_dtype_result =
    parse_dtype_token(entry, key_dtype, explicit_key_dtype_token, &key_dtype_token);
  if (!key_dtype_result.is_supported())
  {
    return key_dtype_result;
  }
  if (pairs)
  {
    const MetadataResult value_dtype_result =
      parse_dtype_token(entry, value_dtype, explicit_value_dtype_token, &value_dtype_token);
    if (!value_dtype_result.is_supported())
    {
      return value_dtype_result;
    }
  }
  const auto topology = validate_logical_extents(entry.family, extent);
  if (!topology.is_supported())
  {
    return topology;
  }
  if (!declaration_symbol_matches_metadata(declaration->symbol, entry, CallView::is_resolved))
  {
    return symbol_error(entry, "declaration symbol does not match the selected family");
  }
  if (!topk_decl_matches(*declaration, extent, key_dtype_token, value_dtype_token, pairs)
      || !topk_argument_types_match(
        call.arguments(), call.argument_count(), extent, key_dtype_token, value_dtype_token, pairs)
      || (!CallView::is_resolved
          && !topk_result_types_match(
            call.results(), call.result_count(), extent, key_dtype_token, value_dtype_token, pairs)))
  {
    return type_error(entry, "declaration, call arguments, or results contradict the resolved metadata");
  }

  return make_logical_request(
    entry.family,
    make_topk_spec(entry, descending, full_tile, key_dtype_token, value_dtype_token),
    extent,
    {target_sm, parse_cache_epoch_from_env()},
    out);
}

// Check the ABI structure shared by all operation-family parsers.
// Discovery retains named constant parameters; resolved declarations have
// already substituted them and must not retain that schema. Operand access
// modes survive both phases and must still match the generated declaration.
MetadataResult
validate_declaration_schema(const FamilyDescriptor& entry, const TileIRForeignDecl& declaration, bool resolved)
{
  const auto* const schema = entry.schema;
  if (!schema)
  {
    return internal_error(entry, "registered family has no generated declaration schema");
  }
  if (resolved)
  {
    if (declaration.num_const_params != 0)
    {
      return schema_error(entry, "resolved declaration retains a constant-parameter schema after concretization");
    }
  }
  else
  {
    if (!declaration.const_params || declaration.num_const_params != schema->num_constants)
    {
      return schema_error(entry, "constant-parameter or access schema does not match the ABI");
    }
    for (size_t i = 0; i < schema->num_constants; ++i)
    {
      if (declaration.const_params[i].kind != schema->constants[i].kind
          || !string_view_equals(declaration.const_params[i].name, schema->constants[i].name))
      {
        return schema_error(entry, "constant-parameter or access schema does not match the ABI");
      }
    }
  }
  if (!declaration.param_access || declaration.num_param_access != schema->num_operands)
  {
    return schema_error(entry, "constant-parameter or access schema does not match the ABI");
  }
  for (size_t i = 0; i < schema->num_operands; ++i)
  {
    if (declaration.param_access[i] != schema->access[i])
    {
      return schema_error(entry, "constant-parameter or access schema does not match the ABI");
    }
  }
  return MetadataResult::success();
}

template <class CallView>
MetadataResult bind_call_request_impl(
  const CallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_architecture,
  std::optional<LogicalRequest>* out)
{
  if (!out)
  {
    return internal_error(entry, "null logical request output");
  }
  out->reset();
  if (!entry.parse_metadata || !target_architecture.valid())
  {
    return internal_error(entry, "invalid family parser or pre-resolved target architecture");
  }
  const auto* const declaration = call.declaration();
  if (!declaration)
  {
    return schema_error(entry, "call has no foreign declaration");
  }
  const auto schema = validate_declaration_schema(entry, *declaration, CallView::is_resolved);
  if (!schema.is_supported())
  {
    return schema;
  }

  return entry.parse_metadata(MetadataCallView{call}, entry, target_architecture, out);
}

// Select one concrete topology after logical binding has succeeded.
// Discovery can use a candidate override when no worker count is known. A
// resolved call must instead agree with the selected scope and width. Only a
// fully selected request with usable include paths is returned to codegen.
template <class CallView>
MetadataResult materialize_call_request(
  const CallView& call,
  const FamilyDescriptor& entry,
  const LogicalRequest& demand,
  int64_t auto_block_threads_override,
  CallMetadata* out)
{
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  *out            = CallMetadata{};
  int64_t threads = call.execution_threads() > 0 ? call.execution_threads() : auto_block_threads_override;
  if constexpr (!CallView::is_resolved)
  {
    if (const auto* const reduce = std::get_if<ReduceSpec>(&demand.primitive());
        reduce && reduce->batches > 0 && reduce->algorithm == "warp_reductions")
    {
      threads = 32;
    }
    // A logical 32-element merge sort has a warp-scoped realization even in
    // kernels whose surrounding CTA contains more than one warp.
    if (std::holds_alternative<MergeSortSpec>(demand.primitive()) && demand.extent() == 32)
    {
      threads = 32;
    }
  }
  const auto selected = select_implementation(demand, threads, &out->request);
  if (!selected.is_supported())
  {
    return selected;
  }
  if constexpr (CallView::is_resolved)
  {
    const auto& implementation = *out->request;
    const bool scope_matches =
      uses_warp_execution_scope(implementation) ? call.has_resolved_warp_scope() : call.has_resolved_cta_scope();
    if (!scope_matches)
    {
      *out = CallMetadata{};
      return metadata_error(
        MetadataStatus::unsupported,
        entry,
        "[LP-TOPOLOGY-001]",
        "resolved execution scope does not match the implementation's participation scope");
    }
    if (!call.topology_matches(implementation.topology().block_threads))
    {
      *out = CallMetadata{};
      return topology_error(entry, "resolved execution scope contradicts BlockThreads");
    }
  }
  else if (std::holds_alternative<ExchangeSpec>(demand.primitive())
           && !call.topology_matches(out->request->topology().block_threads))
  {
    const int width = out->request->topology().block_threads;
    *out            = CallMetadata{};
    return topology_error(
      entry,
      "exchange BlockThreads=" + std::to_string(width)
        + " contradicts the compiler's execution width=" + std::to_string(call.execution_threads()));
  }
  out->include_paths = resolve_provider_include_paths();
  if (out->include_paths.empty())
  {
    *out = CallMetadata{};
    return unsupported_error(entry, "provider include paths are unavailable");
  }
  return MetadataResult::success();
}
} // namespace

MetadataResult
resolve_request_target(TileIRStringView architecture, const FamilyDescriptor& entry, TargetArchitecture* out)
{
  return resolve_request_target_impl(architecture, entry, out);
}

MetadataResult bind_call_request(
  const TileIRLPCallsite& callsite,
  const TileIRLPCompilationContext& context,
  const FamilyDescriptor& entry,
  TargetArchitecture target,
  std::optional<LogicalRequest>* out)
{
  return bind_call_request_impl(DiscoveryCallView{callsite, context}, entry, target, out);
}

MetadataResult materialize_call_request(
  const TileIRLPCallsite& callsite,
  const TileIRLPCompilationContext& context,
  const FamilyDescriptor& entry,
  const LogicalRequest& demand,
  int64_t candidate_threads,
  CallMetadata* out)
{
  return materialize_call_request(DiscoveryCallView{callsite, context}, entry, demand, candidate_threads, out);
}

MetadataResult parse_call_metadata(
  const TileIRLPCallsite& callsite,
  const TileIRLPCompilationContext& context,
  const FamilyDescriptor& entry,
  int64_t candidate_threads,
  TargetArchitecture target,
  CallMetadata* out)
{
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  *out = CallMetadata{};
  std::optional<LogicalRequest> demand;
  const auto bound = bind_call_request(callsite, context, entry, target, &demand);
  return bound.is_supported()
         ? materialize_call_request(callsite, context, entry, *demand, candidate_threads, out)
         : bound;
}

MetadataResult parse_call_metadata(
  const TileIRLPCallsite& callsite,
  const TileIRLPCompilationContext& context,
  const FamilyDescriptor& entry,
  int64_t candidate_threads,
  CallMetadata* out)
{
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  *out = CallMetadata{};
  TargetArchitecture target;
  const auto resolved = resolve_request_target(context.target_architecture, entry, &target);
  return resolved.is_supported()
         ? parse_call_metadata(callsite, context, entry, candidate_threads, target, out)
         : resolved;
}

MetadataResult
parse_call_metadata(const TileIRResolvedCall& call, const FamilyDescriptor& entry, size_t call_index, CallMetadata* out)
{
  static_cast<void>(call_index);
  if (!out)
  {
    return internal_error(entry, "null metadata output");
  }
  *out = CallMetadata{};
  TargetArchitecture target;
  const auto resolved = resolve_request_target(call.resolved_execution_context.architecture, entry, &target);
  if (!resolved.is_supported())
  {
    return resolved;
  }
  std::optional<LogicalRequest> demand;
  const auto bound = bind_call_request_impl(ResolvedCallView{call}, entry, target, &demand);
  return bound.is_supported() ? materialize_call_request(ResolvedCallView{call}, entry, *demand, 0, out) : bound;
}

bool argument_is_scalar_by_value(
  const FamilyDescriptor& entry, size_t argument_index, const TileIRForeignType& type, TileIRLPAccessKind access)
{
  if (access != TILEIR_LP_ACCESS_IN)
  {
    return false;
  }
  static_cast<void>(entry);
  static_cast<void>(argument_index);
  return type.kind == TILEIR_TYPE_TILE && type.num_shape == 0;
}

namespace
{
// Resolve family constant names through the canonical schema's ordered slots.
// This lets the same family parser read discovery expressions and resolved
// arguments even after the compiler has removed parameter names. The schema
// has static storage; the inherited view still borrows the callback's call.
template <class CallView>
class DeclaredCallView : public CallView
{
public:
  DeclaredCallView(const CallView& call, const generated::DeclarationSchema& schema)
      : CallView(call)
      , schema_(schema)
  {}

  bool int_constant(std::string_view name, int64_t* value) const
  {
    const auto index = constant_index(name);
    return index < schema_.num_constants && CallView::int_constant(index, value);
  }
  bool dtype_constant(std::string_view name, TileIRDataType* value) const
  {
    const auto index = constant_index(name);
    return index < schema_.num_constants && CallView::dtype_constant(index, value);
  }
  bool string_constant(std::string_view name, std::string* value) const
  {
    const auto index = constant_index(name);
    return index < schema_.num_constants && CallView::string_constant(index, value);
  }

private:
  size_t constant_index(std::string_view name) const
  {
    for (size_t index = 0; index < schema_.num_constants; ++index)
    {
      if (name == schema_.constants[index].name)
      {
        return index;
      }
    }
    return schema_.num_constants;
  }
  const generated::DeclarationSchema& schema_;
};

// Run a family parser and discard its tentative result on failure.
// The empty optional prevents a later candidate from reusing a partially bound
// request after malformed or unsupported metadata was encountered.
template <class Parser>
MetadataResult parse_with_reset(
  const MetadataCallView& call,
  const generated::DeclarationSchema& schema,
  std::optional<LogicalRequest>* out,
  Parser&& parser)
{
  if (!out)
  {
    return MetadataResult::internal_error("[LP-PROVIDER-INTERNAL-001] null metadata output");
  }
  out->reset();
  MetadataResult result = call.visit([&](const auto& concrete) {
    return parser(DeclaredCallView{concrete, schema});
  });
  if (!result.is_supported())
  {
    out->reset();
  }
  return result;
}
} // namespace

MetadataResult parse_reduce_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_reduce_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_scan_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_scan_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_histogram_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_histogram_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_run_length_decode_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_run_length_decode_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_discontinuity_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_discontinuity_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_adjacent_difference_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_adjacent_difference_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_shuffle_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_shuffle_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_exchange_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_exchange_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_radix_rank_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_radix_rank_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_merge_sort_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_merge_sort_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_radix_sort_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_radix_sort_metadata(concrete_call, entry, target_sm, out);
  });
}

MetadataResult parse_topk_metadata(
  const MetadataCallView& call,
  const FamilyDescriptor& entry,
  TargetArchitecture target_sm,
  std::optional<LogicalRequest>* out)
{
  return parse_with_reset(call, *entry.schema, out, [&](const auto& concrete_call) {
    return make_topk_metadata(concrete_call, entry, target_sm, out);
  });
}
} // namespace cuda::coop::tile::detail::provider
