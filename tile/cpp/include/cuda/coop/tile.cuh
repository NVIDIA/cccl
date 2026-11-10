//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef CUDA_COOP_TILE_CUH
#define CUDA_COOP_TILE_CUH

#include <cuda_tile.h>

#if !defined(__CUDACC_TILE__) || !defined(__CUDACC_TILE_PREVIEW_SIMT__)
#  error "cuda::coop::tile requires nvcc --enable-tile --tile-preview=simt"
#else

#  include <cuda_bf16.h>
#  include <cuda_fp16.h>

#  include <cuda/coop/__tile/foreign_declarations.cuh>
#  include <cuda/functional>
#  include <cuda/std/functional>
#  include <cuda/std/type_traits>
#  include <cuda/std/utility>

// The common CCCL headers may disable preview Tile support. Keep the
// execution-space annotation local to this explicitly enabled frontend.
#  if defined(__CUDACC_TILE__)
#    define _CCCL_COOP_TILE_API __tile__ _CCCL_VISIBILITY_HIDDEN _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
#  else
#    define _CCCL_COOP_TILE_API _CCCL_TILE_API
#  endif

/**
 * @brief Cooperative primitives on CUDA Tile values.
 *
 * Calls return new Tile values and leave their input values unchanged. Numeric
 * elements are signed and unsigned 8-, 16-, 32-, and 64-bit integers, __half,
 * __nv_bfloat16, float, and double, subject to each operation's restrictions.
 * Boolean and structured element types are unsupported. Pair inputs require
 * equal extents; their element types may differ. Accumulation does not widen.
 * Reductions accept a compile-time axis; other primitives consume one-dimensional tiles.
 *
 * The compiler allocates temporary storage and selects physical participation.
 * Every call in a kernel must admit a compatible worker count. Operations apply
 * independently to each logical block; they do not combine results across blocks.
 */
namespace cuda::coop::tile
{
namespace ct  = ::cuda::tiles;
namespace cte = ::cuda::tiles::experimental;

/**
 * @brief CUB algorithm used to compute block prefixes.
 */
enum class scan_algorithm
{
  raking,
  raking_memoize,
  warp_scans,
};

/**
 * @brief CUB algorithm used to combine a block of values.
 */
enum class reduce_algorithm
{
  raking_commutative_only,
  raking,
  warp_reductions,
};

/**
 * @brief Shared-memory bank width used by radix operations.
 */
enum class radix_smem_config
{
  four_byte,
  eight_byte,
};

/**
 * @brief Whether each prefix includes its current input element.
 */
enum class scan_mode
{
  exclusive,
  inclusive,
};

/**
 * @brief Select a scan algorithm; physical participation remains compiler-selected.
 */
template <scan_algorithm Algorithm = scan_algorithm::raking>
struct scan_policy
{};

/**
 * @brief Select a reduction algorithm; physical participation remains compiler-selected.
 */
template <reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
struct reduce_policy
{};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct topk_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile topk_policy accepts no options");
};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct histogram_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile histogram_policy accepts no options");
};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct run_length_decode_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile run_length_decode_policy accepts no options");
};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct discontinuity_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile discontinuity_policy accepts no options");
};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct adjacent_difference_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile adjacent_difference_policy accepts no options");
};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct shuffle_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile shuffle_policy accepts no options");
};

/**
 * @brief Select the semantic permutation width. It must be a multiple of 32 in [32, 1024] and divide the
 * input extent.
 */
template <size_t BlockThreads>
struct exchange_policy
{
  static_assert(BlockThreads >= 32 && BlockThreads <= 1024 && BlockThreads % 32 == 0,
                "cuda::coop::tile Exchange requires explicit block_threads "
                "that are a multiple of 32 in [32, 1024]");
};

/**
 * @brief Select a radix-digit capacity in [1, 16], direction, scan strategy, and shared-memory bank
 * width.
 */
template <size_t RadixBits                  = 4,
          bool Descending                   = false,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
struct radix_rank_policy
{
  static_assert(RadixBits >= 1 && RadixBits <= 16, "cuda::coop::tile Radix Rank radix_bits must be in [1, 16]");
};

/**
 * @brief Empty policy for this family. Only the empty template argument list is supported; it allocates
 * no scratch storage.
 */
template <typename... Options>
struct merge_sort_policy
{
  static_assert(sizeof...(Options) == 0, "cuda::coop::tile merge_sort_policy accepts no options");
};

/**
 * @brief Select radix bits per pass in [1, 8], scan strategy, and shared-memory bank width. The bit
 * interval can span several passes.
 */
template <size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
struct radix_sort_policy
{
  static_assert(RadixBits >= 1 && RadixBits <= 8, "cuda::coop::tile Radix Sort radix_bits must be in [1, 8]");
};

/// @brief Selected keys followed by their payloads; only the selected prefix is defined.
template <typename KeyT, typename ValueT, size_t N>
using topk_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>, ct::tile<ValueT, ct::shape<N>>>;

/// @brief Sorted keys followed by their associated payloads.
template <typename KeyT, typename ValueT, size_t N>
using merge_sort_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>, ct::tile<ValueT, ct::shape<N>>>;

/// @brief Radix-sorted keys followed by their associated payloads.
template <typename KeyT, typename ValueT, size_t N>
using radix_sort_pair_result = cte::multi_outputs<ct::tile<KeyT, ct::shape<N>>, ct::tile<ValueT, ct::shape<N>>>;

/// @brief Head flags followed by tail flags, both with int elements.
template <size_t N>
using discontinuity_pair_result = cte::multi_outputs<ct::tile<int, ct::shape<N>>, ct::tile<int, ct::shape<N>>>;

/// @brief Decoded values followed by relative offsets with the run-length element type.
template <typename ValueT, typename LengthT, size_t N>
using run_length_decode_with_offsets_result =
  cte::multi_outputs<ct::tile<ValueT, ct::shape<N>>, ct::tile<LengthT, ct::shape<N>>>;

namespace detail
{
template <typename T>
struct cpp_dtype_token
{
  inline static constexpr bool supported = false;
};

#  define _CCCL_COOP_TILE_DTYPE_TOKEN(_Type, _Token)                     \
    template <>                                                          \
    struct cpp_dtype_token<_Type>                                        \
    {                                                                    \
      inline static constexpr bool supported = true;                     \
      inline static constexpr auto value     = cte::fixedString{_Token}; \
    }

_CCCL_COOP_TILE_DTYPE_TOKEN(signed char, "i8");
_CCCL_COOP_TILE_DTYPE_TOKEN(unsigned char, "u8");
_CCCL_COOP_TILE_DTYPE_TOKEN(short, "i16");
_CCCL_COOP_TILE_DTYPE_TOKEN(unsigned short, "u16");
_CCCL_COOP_TILE_DTYPE_TOKEN(int, "i32");
_CCCL_COOP_TILE_DTYPE_TOKEN(unsigned int, "u32");
_CCCL_COOP_TILE_DTYPE_TOKEN(long long, "i64");
_CCCL_COOP_TILE_DTYPE_TOKEN(unsigned long long, "u64");
_CCCL_COOP_TILE_DTYPE_TOKEN(long, sizeof(long) == 4 ? "i32" : "i64");
_CCCL_COOP_TILE_DTYPE_TOKEN(unsigned long, sizeof(unsigned long) == 4 ? "u32" : "u64");
_CCCL_COOP_TILE_DTYPE_TOKEN(__half, "f16");
_CCCL_COOP_TILE_DTYPE_TOKEN(__nv_bfloat16, "bf16");
_CCCL_COOP_TILE_DTYPE_TOKEN(float, "f32");
_CCCL_COOP_TILE_DTYPE_TOKEN(double, "f64");

#  undef _CCCL_COOP_TILE_DTYPE_TOKEN

template <typename T>
inline constexpr bool supported_element_v = cpp_dtype_token<T>::supported;

template <typename T>
_CCCL_CONSTEVAL auto checked_cpp_dtype_token() noexcept
{
  static_assert(supported_element_v<T>,
                "cuda::coop::tile supports signed and unsigned 8-, 16-, "
                "32-, and 64-bit integers, f16, bf16, f32, and f64");
  if constexpr (supported_element_v<T>)
  {
    return cpp_dtype_token<T>::value;
  }
  else
  {
    return cte::fixedString{""};
  }
}

template <typename T>
inline constexpr auto cpp_dtype_token_v = checked_cpp_dtype_token<T>();

template <scan_algorithm Algorithm>
struct radix_sort_scan_algorithm_token;

template <>
struct radix_sort_scan_algorithm_token<scan_algorithm::raking>
{
  inline static constexpr auto value = cte::fixedString{"raking"};
};

template <>
struct radix_sort_scan_algorithm_token<scan_algorithm::raking_memoize>
{
  inline static constexpr auto value = cte::fixedString{"raking_memoize"};
};

template <>
struct radix_sort_scan_algorithm_token<scan_algorithm::warp_scans>
{
  inline static constexpr auto value = cte::fixedString{"warp_scans"};
};

template <scan_algorithm Algorithm>
inline constexpr auto radix_sort_scan_algorithm_token_v = radix_sort_scan_algorithm_token<Algorithm>::value;

template <radix_smem_config SmemConfig>
struct radix_smem_config_token;

template <>
struct radix_smem_config_token<radix_smem_config::four_byte>
{
  inline static constexpr auto value = cte::fixedString{"four_byte"};
};

template <>
struct radix_smem_config_token<radix_smem_config::eight_byte>
{
  inline static constexpr auto value = cte::fixedString{"eight_byte"};
};

template <radix_smem_config SmemConfig>
inline constexpr auto radix_smem_config_token_v = radix_smem_config_token<SmemConfig>::value;

template <typename T, typename OpT>
inline constexpr bool compatible_functor_v = ::cuda::std::is_void_v<OpT> || ::cuda::std::is_same_v<T, OpT>;

template <typename T, typename OpT>
_CCCL_CONSTEVAL void validate_functor() noexcept
{
  static_assert(compatible_functor_v<T, OpT>,
                "cuda::coop::tile functor value_type must be void or exactly "
                "match the Tile dtype");
}

template <typename ControlT>
inline constexpr bool shape_one_i32_control_v = ::cuda::std::is_same_v<ControlT, ct::tile<int, ct::shape<1>>>;

template <typename ControlT>
inline constexpr bool compatible_i32_control_v =
  shape_one_i32_control_v<ControlT> || ::cuda::std::is_convertible_v<ControlT, int>;

template <typename T, typename ControlT>
inline constexpr bool shape_one_value_control_v = ::cuda::std::is_same_v<ControlT, ct::tile<T, ct::shape<1>>>;

template <typename T, typename ControlT>
inline constexpr bool compatible_value_control_v =
  shape_one_value_control_v<T, ControlT> || ::cuda::std::is_convertible_v<ControlT, T>;

template <typename LengthT>
inline constexpr bool run_length_decode_length_type_v =
  ::cuda::std::is_same_v<LengthT, unsigned int> || ::cuda::std::is_same_v<LengthT, unsigned long long>
  || ::cuda::std::is_same_v<LengthT, unsigned long>;

template <typename LengthT, typename ControlT>
inline constexpr bool compatible_run_length_decode_offset_v =
  shape_one_value_control_v<LengthT, ControlT>
  || (::cuda::std::is_integral_v<ControlT> && ::cuda::std::is_convertible_v<ControlT, LengthT>);

template <typename ControlT>
[[nodiscard]] _CCCL_COOP_TILE_API inline int normalize_i32_control(ControlT control) noexcept
{
  static_assert(compatible_i32_control_v<ControlT>,
                "cuda::coop::tile integer controls must be scalars convertible to int or shape-one int Tiles");
  return static_cast<int>(control);
}

template <typename T, typename ControlT>
[[nodiscard]] _CCCL_COOP_TILE_API inline T normalize_value_control(ControlT control) noexcept
{
  static_assert(compatible_value_control_v<T, ControlT>,
                "cuda::coop::tile value controls must be scalars convertible to the Tile dtype or shape-one Tiles "
                "with the Tile dtype");
  return static_cast<T>(control);
}

template <typename LengthT, typename ControlT>
[[nodiscard]] _CCCL_COOP_TILE_API inline LengthT normalize_run_length_decode_offset(ControlT control) noexcept
{
  static_assert(compatible_run_length_decode_offset_v<LengthT, ControlT>,
                "cuda::coop::tile Run-Length Decode window offset must be an integral scalar convertible to the "
                "length dtype or a shape-one Tile with the length dtype");
  return static_cast<LengthT>(control);
}

template <typename T, size_t RadixBits>
[[nodiscard]] _CCCL_COOP_TILE_API inline int default_radix_rank_end(int begin_bit) noexcept
{
  constexpr int key_bits   = static_cast<int>(sizeof(T) * 8);
  constexpr int radix_bits = static_cast<int>(RadixBits);
  return begin_bit >= key_bits - radix_bits ? key_bits : begin_bit + radix_bits;
}

template <cte::fixedString Mode,
          cte::fixedString ScanOp,
          int InitialPresent,
          int PrefixPresent,
          typename T,
          size_t N,
          scan_algorithm Algorithm>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
scan_call(ct::tile<T, ct::shape<N>> input, T runtime_initial, T runtime_prefix) noexcept
{
  static_assert(N > 0, "cuda::coop::tile scan expects a non-empty tile");
  static_assert(InitialPresent == 0 || InitialPresent == 1);
  static_assert(PrefixPresent == 0 || PrefixPresent == 1);
  static_assert(!(InitialPresent && PrefixPresent),
                "cuda::coop::tile scan cannot set both an initial value and a prefix value");
  auto runtime_initial_tile = ct::tile{runtime_initial};
  auto runtime_prefix_tile  = ct::tile{runtime_prefix};
  if constexpr (Algorithm == scan_algorithm::raking)
  {
    return detail::scan_foreign<N, T, Mode, "raking", ScanOp, InitialPresent, PrefixPresent, cpp_dtype_token_v<T>>(
      input, runtime_initial_tile, runtime_prefix_tile);
  }
  else if constexpr (Algorithm == scan_algorithm::raking_memoize)
  {
    return detail::
      scan_foreign<N, T, Mode, "raking_memoize", ScanOp, InitialPresent, PrefixPresent, cpp_dtype_token_v<T>>(
        input, runtime_initial_tile, runtime_prefix_tile);
  }
  else
  {
    static_assert(Algorithm == scan_algorithm::warp_scans);
    return detail::scan_foreign<N, T, Mode, "warp_scans", ScanOp, InitialPresent, PrefixPresent, cpp_dtype_token_v<T>>(
      input, runtime_initial_tile, runtime_prefix_tile);
  }
}

template <typename T, size_t N, size_t Batches, reduce_algorithm Algorithm>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<Batches>>
sum_batched_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile sum expects a non-empty tile");
  if constexpr (Algorithm == reduce_algorithm::raking_commutative_only)
  {
    return detail::sum_batched_foreign<N, T, "raking_commutative_only", cpp_dtype_token_v<T>, Batches>(input);
  }
  else if constexpr (Algorithm == reduce_algorithm::raking)
  {
    return detail::sum_batched_foreign<N, T, "raking", cpp_dtype_token_v<T>, Batches>(input);
  }
  else
  {
    static_assert(Algorithm == reduce_algorithm::warp_reductions);
    return detail::sum_batched_foreign<N, T, "warp_reductions", cpp_dtype_token_v<T>, Batches>(input);
  }
}

template <cte::fixedString BinaryOp, bool IntegralOnly, typename T, size_t N, size_t Batches, reduce_algorithm Algorithm>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<Batches>>
reduce_batched_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile reduce expects a non-empty tile");
  static_assert(!IntegralOnly || ::cuda::std::is_integral_v<T>,
                "cuda::coop::tile bitwise reductions require an integer "
                "tile dtype");
  if constexpr (Algorithm == reduce_algorithm::raking_commutative_only)
  {
    return detail::reduce_batched_foreign<N, T, "raking_commutative_only", BinaryOp, cpp_dtype_token_v<T>, Batches>(
      input);
  }
  else if constexpr (Algorithm == reduce_algorithm::raking)
  {
    return detail::reduce_batched_foreign<N, T, "raking", BinaryOp, cpp_dtype_token_v<T>, Batches>(input);
  }
  else
  {
    static_assert(Algorithm == reduce_algorithm::warp_reductions);
    return detail::reduce_batched_foreign<N, T, "warp_reductions", BinaryOp, cpp_dtype_token_v<T>, Batches>(input);
  }
}

template <size_t Axis, typename Tile, size_t... Indices>
[[nodiscard]] _CCCL_COOP_TILE_API inline auto
reduction_rows(Tile input, ::cuda::std::index_sequence<Indices...>) noexcept
{
  // Retained axes keep their original order. Each resulting row becomes one
  // logical reduction in the native provider's batched declaration.
  const auto rows = ct::permute(input, ct::dimension_map<(Indices < Axis ? Indices : Indices + 1)..., Axis>{});
  return ct::reshape(rows, ct::shape<ct::tile_size_v<Tile>>{});
}

template <size_t Bins, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<Bins>>
histogram_call(ct::tile<unsigned char, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Histogram expects a non-empty input tile");
  static_assert(Bins >= 32 && Bins <= 256 && Bins % 32 == 0,
                "cuda::coop::tile Histogram bins must be a multiple of 32 in [32, 256]");
  return detail::histogram_foreign<N, Bins, cpp_dtype_token_v<unsigned char>>(input);
}

template <size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<256>>
histogram_256_call(ct::tile<unsigned char, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Histogram-256 expects a non-empty input tile");
  return detail::histogram_256_foreign<N, cpp_dtype_token_v<unsigned char>>(input);
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t RunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<ValueT, ct::shape<DecodedN>> run_length_decode_call(
  ct::tile<ValueT, ct::shape<RunN>> run_values,
  ct::tile<LengthT, ct::shape<RunN>> run_lengths,
  LengthT decoded_window_offset) noexcept
{
  static_assert(RunN > 0, "cuda::coop::tile Run-Length Decode expects non-empty run tiles");
  static_assert(DecodedN > 0, "cuda::coop::tile Run-Length Decode expects a non-empty decoded tile");
  static_assert(RunN <= 2147483647, "cuda::coop::tile Run-Length Decode run extent must fit in int");
  static_assert(DecodedN <= 2147483647, "cuda::coop::tile Run-Length Decode decoded extent must fit in int");
  static_assert(run_length_decode_length_type_v<LengthT>,
                "cuda::coop::tile Run-Length Decode length dtype must be u32 or u64");

  if constexpr (run_length_decode_length_type_v<LengthT>)
  {
    auto runtime_offset_tile = ct::tile{decoded_window_offset};
    return detail::
      run_length_decode_foreign<RunN, DecodedN, ValueT, LengthT, cpp_dtype_token_v<ValueT>, cpp_dtype_token_v<LengthT>>(
        run_values, run_lengths, runtime_offset_tile);
  }
  else
  {
    return {};
  }
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t RunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline run_length_decode_with_offsets_result<ValueT, LengthT, DecodedN>
run_length_decode_with_offsets_call(ct::tile<ValueT, ct::shape<RunN>> run_values,
                                    ct::tile<LengthT, ct::shape<RunN>> run_lengths,
                                    LengthT decoded_window_offset) noexcept
{
  static_assert(RunN > 0, "cuda::coop::tile Run-Length Decode expects non-empty run tiles");
  static_assert(DecodedN > 0, "cuda::coop::tile Run-Length Decode expects a non-empty decoded tile");
  static_assert(RunN <= 2147483647, "cuda::coop::tile Run-Length Decode run extent must fit in int");
  static_assert(DecodedN <= 2147483647, "cuda::coop::tile Run-Length Decode decoded extent must fit in int");
  static_assert(run_length_decode_length_type_v<LengthT>,
                "cuda::coop::tile Run-Length Decode length dtype must be u32 or u64");

  if constexpr (run_length_decode_length_type_v<LengthT>)
  {
    auto runtime_offset_tile = ct::tile{decoded_window_offset};
    return detail::run_length_decode_with_offsets_foreign<
      RunN,
      DecodedN,
      ValueT,
      LengthT,
      cpp_dtype_token_v<ValueT>,
      cpp_dtype_token_v<LengthT>>(run_values, run_lengths, runtime_offset_tile);
  }
  else
  {
    return {};
  }
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<ValueT, ct::shape<DecodedN>> run_length_decode_checked(
  ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
  ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
  LengthT decoded_window_offset) noexcept
{
  static_assert(ValueRunN == LengthRunN,
                "cuda::coop::tile Run-Length Decode value and length tiles must have matching extents");
  if constexpr (ValueRunN == LengthRunN)
  {
    return detail::run_length_decode_call<DecodedN, ValueT, LengthT, ValueRunN>(
      run_values, run_lengths, decoded_window_offset);
  }
  else
  {
    return {};
  }
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline run_length_decode_with_offsets_result<ValueT, LengthT, DecodedN>
run_length_decode_with_offsets_checked(ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
                                       ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
                                       LengthT decoded_window_offset) noexcept
{
  static_assert(ValueRunN == LengthRunN,
                "cuda::coop::tile Run-Length Decode value and length tiles must have matching extents");
  if constexpr (ValueRunN == LengthRunN)
  {
    return detail::run_length_decode_with_offsets_call<DecodedN, ValueT, LengthT, ValueRunN>(
      run_values, run_lengths, decoded_window_offset);
  }
  else
  {
    return {};
  }
}

template <cte::fixedString Mode, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>>
discontinuity_flags_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Discontinuity expects a non-empty tile");
  return detail::discontinuity_flags_foreign<N, T, Mode, cpp_dtype_token_v<T>>(input);
}

template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline discontinuity_pair_result<N>
discontinuity_heads_and_tails_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Discontinuity expects a non-empty tile");
  return detail::discontinuity_heads_and_tails_foreign<N, T, "heads_and_tails", cpp_dtype_token_v<T>>(input);
}

template <cte::fixedString Mode, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
adjacent_difference_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Adjacent Difference expects a non-empty tile");
  return detail::adjacent_difference_foreign<N, T, Mode, cpp_dtype_token_v<T>>(input);
}

template <cte::fixedString Mode, bool SingleItem, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_call(ct::tile<T, ct::shape<N>> input, int distance) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Shuffle expects a non-empty tile");
  static_assert(!SingleItem || (N >= 32 && N <= 1024 && N % 32 == 0),
                "cuda::coop::tile shuffle_offset and shuffle_rotate require "
                "one item per thread and an extent that is a multiple of 32 "
                "in [32, 1024]");
  return detail::shuffle_foreign<N, T, Mode, cpp_dtype_token_v<T>>(input, distance);
}

template <cte::fixedString Mode, typename T, size_t N, size_t BlockThreads>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exchange_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Exchange expects a non-empty tile");
  static_assert(N % BlockThreads == 0,
                "cuda::coop::tile Exchange tile extent must be divisible by "
                "the requested block_threads");

  return detail::exchange_foreign<N, T, BlockThreads, Mode, cpp_dtype_token_v<T>>(input);
}

template <cte::fixedString InnerScanAlgorithm,
          typename T,
          size_t N,
          size_t RadixBits,
          bool Descending,
          bool MemoizeOuterScan,
          radix_smem_config SmemConfig>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>>
radix_rank_smem_call(ct::tile<T, ct::shape<N>> keys, int begin_bit, int end_bit) noexcept
{
  if constexpr (SmemConfig == radix_smem_config::four_byte)
  {
    return detail::radix_rank_foreign<
      N,
      T,
      RadixBits,
      static_cast<size_t>(Descending),
      static_cast<size_t>(MemoizeOuterScan),
      InnerScanAlgorithm,
      "four_byte",
      cpp_dtype_token_v<T>>(keys, begin_bit, end_bit);
  }
  else
  {
    static_assert(SmemConfig == radix_smem_config::eight_byte);
    return detail::radix_rank_foreign<
      N,
      T,
      RadixBits,
      static_cast<size_t>(Descending),
      static_cast<size_t>(MemoizeOuterScan),
      InnerScanAlgorithm,
      "eight_byte",
      cpp_dtype_token_v<T>>(keys, begin_bit, end_bit);
  }
}

template <typename T,
          size_t N,
          size_t RadixBits,
          bool Descending,
          bool MemoizeOuterScan,
          scan_algorithm InnerScanAlgorithm,
          radix_smem_config SmemConfig>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>>
radix_rank_call(ct::tile<T, ct::shape<N>> keys, int begin_bit, int end_bit) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Radix Rank expects a non-empty tile");
  static_assert(N <= 65535, "cuda::coop::tile Radix Rank tile extent must be at most 65535");
  static_assert(::cuda::std::is_integral_v<T> && !::cuda::std::is_same_v<T, bool>,
                "cuda::coop::tile Radix Rank key dtype must be a non-bool integer");

  if constexpr (InnerScanAlgorithm == scan_algorithm::raking)
  {
    return detail::radix_rank_smem_call<"raking", T, N, RadixBits, Descending, MemoizeOuterScan, SmemConfig>(
      keys, begin_bit, end_bit);
  }
  else if constexpr (InnerScanAlgorithm == scan_algorithm::raking_memoize)
  {
    return detail::radix_rank_smem_call<"raking_memoize", T, N, RadixBits, Descending, MemoizeOuterScan, SmemConfig>(
      keys, begin_bit, end_bit);
  }
  else
  {
    static_assert(InnerScanAlgorithm == scan_algorithm::warp_scans);
    return detail::radix_rank_smem_call<"warp_scans", T, N, RadixBits, Descending, MemoizeOuterScan, SmemConfig>(
      keys, begin_bit, end_bit);
  }
}

template <size_t Descending, size_t PartialTile, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
merge_sort_keys_call(ct::tile<T, ct::shape<N>> keys, int valid_items, T oob_default) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Merge Sort expects a non-empty tile");
  static_assert(N <= 2147483647, "cuda::coop::tile Merge Sort tile extent must fit in int");
  auto runtime_oob_default_tile = ct::tile{oob_default};
  return detail::merge_sort_keys_foreign<N, T, Descending, PartialTile, cpp_dtype_token_v<T>>(
    keys, valid_items, runtime_oob_default_tile);
}

template <size_t Descending, size_t PartialTile, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, N> merge_sort_pairs_call(
  ct::tile<KeyT, ct::shape<N>> keys, ct::tile<ValueT, ct::shape<N>> values, int valid_items, KeyT oob_default) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Merge Sort expects non-empty tiles");
  static_assert(N <= 2147483647, "cuda::coop::tile Merge Sort tile extent must fit in int");
  auto runtime_oob_default_tile = ct::tile{oob_default};
  return detail::
    merge_sort_pairs_foreign<N, KeyT, ValueT, Descending, PartialTile, cpp_dtype_token_v<KeyT>, cpp_dtype_token_v<ValueT>>(
      keys, values, valid_items, runtime_oob_default_tile);
}

template <size_t Descending, size_t FullTile, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
topk_keys_call(ct::tile<T, ct::shape<N>> keys, int k, int valid_items, int begin_bit, int end_bit) noexcept
{
  static_assert(N > 0, "cuda::coop::tile TopK expects a non-empty tile");
  static_assert(N <= 2147483647, "cuda::coop::tile TopK tile extent must fit in int");
  return detail::topk_keys_foreign<N, T, Descending, FullTile, cpp_dtype_token_v<T>>(
    keys, k, valid_items, begin_bit, end_bit);
}

template <size_t Descending, size_t FullTile, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> topk_pairs_call(
  ct::tile<KeyT, ct::shape<N>> keys,
  ct::tile<ValueT, ct::shape<N>> values,
  int k,
  int valid_items,
  int begin_bit,
  int end_bit) noexcept
{
  static_assert(N > 0, "cuda::coop::tile TopK expects a non-empty tile");
  static_assert(N <= 2147483647, "cuda::coop::tile TopK tile extent must fit in int");
  return detail::
    topk_pairs_foreign<N, KeyT, ValueT, Descending, FullTile, cpp_dtype_token_v<KeyT>, cpp_dtype_token_v<ValueT>>(
      keys, values, k, valid_items, begin_bit, end_bit);
}

template <size_t Descending,
          typename T,
          size_t N,
          size_t RadixBits,
          bool MemoizeOuterScan,
          scan_algorithm InnerScanAlgorithm,
          radix_smem_config SmemConfig>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
radix_sort_keys_call(ct::tile<T, ct::shape<N>> keys, int begin_bit, int end_bit) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Radix Sort expects a non-empty tile");
  static_assert(N <= 65535, "cuda::coop::tile Radix Sort tile extent must be at most 65535");
  return detail::radix_sort_keys_foreign<
    N,
    T,
    Descending,
    RadixBits,
    MemoizeOuterScan ? 1 : 0,
    radix_sort_scan_algorithm_token_v<InnerScanAlgorithm>,
    radix_smem_config_token_v<SmemConfig>,
    cpp_dtype_token_v<T>>(keys, begin_bit, end_bit);
}

template <size_t Descending,
          typename KeyT,
          typename ValueT,
          size_t N,
          size_t RadixBits,
          bool MemoizeOuterScan,
          scan_algorithm InnerScanAlgorithm,
          radix_smem_config SmemConfig>
[[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, N> radix_sort_pairs_call(
  ct::tile<KeyT, ct::shape<N>> keys, ct::tile<ValueT, ct::shape<N>> values, int begin_bit, int end_bit) noexcept
{
  static_assert(N > 0, "cuda::coop::tile Radix Sort expects non-empty tiles");
  static_assert(N <= 65535, "cuda::coop::tile Radix Sort tile extent must be at most 65535");
  return detail::radix_sort_pairs_foreign<
    N,
    KeyT,
    ValueT,
    Descending,
    RadixBits,
    MemoizeOuterScan ? 1 : 0,
    radix_sort_scan_algorithm_token_v<InnerScanAlgorithm>,
    radix_smem_config_token_v<SmemConfig>,
    cpp_dtype_token_v<KeyT>,
    cpp_dtype_token_v<ValueT>>(keys, values, begin_bit, end_bit);
}
} // namespace detail
} // namespace cuda::coop::tile

#  undef _CCCL_COOP_TILE_API

#endif // CUDA C++ Tile SIMT preview

#endif // CUDA_COOP_TILE_CUH
