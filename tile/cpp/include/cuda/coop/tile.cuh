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
 *
 * Scratch storage grows exponentially with RadixBits. Values within this range can fail compilation
 * depending on the worker count and target resources.
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
 *
 * Scratch storage grows exponentially with RadixBits. Values within this range can fail compilation
 * depending on the worker count and target resources.
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile scan expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Histogram expects a positive input tile size divisible by 32");
  static_assert(Bins >= 32 && Bins <= 256 && Bins % 32 == 0,
                "cuda::coop::tile Histogram bins must be a multiple of 32 in [32, 256]");
  return detail::histogram_foreign<N, Bins, cpp_dtype_token_v<unsigned char>>(input);
}

template <size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<256>>
histogram_256_call(ct::tile<unsigned char, ct::shape<N>> input) noexcept
{
  static_assert(N > 0 && N % 32 == 0,
                "cuda::coop::tile Histogram-256 expects a positive input tile size divisible by 32");
  return detail::histogram_256_foreign<N, cpp_dtype_token_v<unsigned char>>(input);
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t RunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<ValueT, ct::shape<DecodedN>> run_length_decode_call(
  ct::tile<ValueT, ct::shape<RunN>> run_values,
  ct::tile<LengthT, ct::shape<RunN>> run_lengths,
  LengthT decoded_window_offset) noexcept
{
  static_assert(RunN > 0 && RunN % 32 == 0,
                "cuda::coop::tile Run-Length Decode expects a positive run tile size divisible by 32");
  static_assert(DecodedN > 0 && DecodedN % 32 == 0,
                "cuda::coop::tile Run-Length Decode expects a positive decoded tile size divisible by 32");
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
  static_assert(RunN > 0 && RunN % 32 == 0,
                "cuda::coop::tile Run-Length Decode expects a positive run tile size divisible by 32");
  static_assert(DecodedN > 0 && DecodedN % 32 == 0,
                "cuda::coop::tile Run-Length Decode expects a positive decoded tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Discontinuity expects a positive tile size divisible by 32");
  return detail::discontinuity_flags_foreign<N, T, Mode, cpp_dtype_token_v<T>>(input);
}

template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline discontinuity_pair_result<N>
discontinuity_heads_and_tails_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Discontinuity expects a positive tile size divisible by 32");
  return detail::discontinuity_heads_and_tails_foreign<N, T, "heads_and_tails", cpp_dtype_token_v<T>>(input);
}

template <cte::fixedString Mode, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
adjacent_difference_call(ct::tile<T, ct::shape<N>> input) noexcept
{
  static_assert(N > 0 && N % 32 == 0,
                "cuda::coop::tile Adjacent Difference expects a positive tile size divisible by 32");
  return detail::adjacent_difference_foreign<N, T, Mode, cpp_dtype_token_v<T>>(input);
}

template <cte::fixedString Mode, bool SingleItem, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_call(ct::tile<T, ct::shape<N>> input, int distance) noexcept
{
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Shuffle expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Radix Rank expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Merge Sort expects a positive tile size divisible by 32");
  static_assert(N <= 2147483647, "cuda::coop::tile Merge Sort tile extent must fit in int");
  auto runtime_oob_default_tile = ct::tile{oob_default};
  return detail::merge_sort_keys_foreign<N, T, Descending, PartialTile, cpp_dtype_token_v<T>>(
    keys, valid_items, runtime_oob_default_tile);
}

template <size_t Descending, size_t PartialTile, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, N> merge_sort_pairs_call(
  ct::tile<KeyT, ct::shape<N>> keys, ct::tile<ValueT, ct::shape<N>> values, int valid_items, KeyT oob_default) noexcept
{
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Merge Sort expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile TopK expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile TopK expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Radix Sort expects a positive tile size divisible by 32");
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
  static_assert(N > 0 && N % 32 == 0, "cuda::coop::tile Radix Sort expects a positive tile size divisible by 32");
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

/**
 * @brief Return the sum of preceding elements, with zero at position zero.
 *
 * The result preserves the input shape and dtype.
 * All Scan calls require `N` to be a positive multiple of 32.
 */
template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exclusive_sum(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> = {}) noexcept
{
  return detail::scan_call<"exclusive", "sum", 0, 0, T, N, Algorithm>(input, T{}, T{});
}

/**
 * @brief Return an exclusive sum starting with a runtime prefix.
 *
 * The prefix is a scalar convertible to the input dtype or a shape-one Tile of that dtype.
 */
template <typename T, size_t N, typename PrefixT, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exclusive_sum(ct::tile<T, ct::shape<N>> input, PrefixT prefix_value, scan_policy<Algorithm> = {}) noexcept
{
  return detail::scan_call<"exclusive", "sum", 0, 1, T, N, Algorithm>(
    input, T{}, detail::normalize_value_control<T>(prefix_value));
}

/**
 * @brief Return the sum through each input position, preserving the input shape and dtype.
 */
template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
inclusive_sum(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> = {}) noexcept
{
  return detail::scan_call<"inclusive", "sum", 0, 0, T, N, Algorithm>(input, T{}, T{});
}

/**
 * @brief Return an inclusive sum including a runtime prefix before the first input.
 *
 * The prefix is a scalar convertible to the input dtype or a shape-one Tile of that dtype.
 */
template <typename T, size_t N, typename PrefixT, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
inclusive_sum(ct::tile<T, ct::shape<N>> input, PrefixT prefix_value, scan_policy<Algorithm> = {}) noexcept
{
  return detail::scan_call<"inclusive", "sum", 0, 1, T, N, Algorithm>(
    input, T{}, detail::normalize_value_control<T>(prefix_value));
}

/**
 * @brief Select the mode with `scan_mode`.
 *
 * The overload without an operation computes addition. An operation may follow `input`; a
 * seed or prefix, when present, follows the operation. In exclusive mode this is the initial
 * value; in inclusive mode it is a prefix combined with the first input. Exclusive non-addition
 * requires a seed. Supported operations are plus, multiplies, minimum,
 * maximum, bit_and, bit_or, and bit_xor. Typed functors match the element type; void functors are
 * also accepted. Bitwise operations require integral input. Seeds are scalars convertible to the
 * input dtype or shape-one Tiles of exactly that dtype.
 */
template <scan_mode Mode = scan_mode::exclusive, typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
scan(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> policy = {}) noexcept
{
  if constexpr (Mode == scan_mode::exclusive)
  {
    return ::cuda::coop::tile::exclusive_sum(input, policy);
  }
  else
  {
    static_assert(Mode == scan_mode::inclusive);
    return ::cuda::coop::tile::inclusive_sum(input, policy);
  }
}

/**
 * @brief Compute an exclusive prefix, preserving shape and dtype.
 *
 * The overload without an operation computes addition with a zero seed. Addition also accepts a
 * seed immediately after `input`. With an operation, supply it after `input` and the seed after
 * the operation. Only addition can omit its seed. Seeds accept a scalar convertible to `T` or
 * `ct::tile<T, ct::shape<1>>` with exactly the input dtype.
 */
template <typename T, size_t N, typename OpT, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exclusive_scan(ct::tile<T, ct::shape<N>> input, ::cuda::std::plus<OpT>, scan_policy<Algorithm> policy = {}) noexcept
{
  detail::validate_functor<T, OpT>();
  return ::cuda::coop::tile::exclusive_sum(input, policy);
}

template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exclusive_scan(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> policy = {}) noexcept
{
  return ::cuda::coop::tile::exclusive_sum(input, policy);
}

template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> exclusive_scan(
  ct::tile<T, ct::shape<N>> input, ::cuda::std::type_identity_t<T> initial_value, scan_policy<Algorithm> = {}) noexcept
{
  return detail::scan_call<"exclusive", "sum", 1, 0, T, N, Algorithm>(input, initial_value, T{});
}

template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> exclusive_scan(
  ct::tile<T, ct::shape<N>> input, ct::tile<T, ct::shape<1>> initial_value, scan_policy<Algorithm> policy = {}) noexcept
{
  const T scalar_initial = static_cast<T>(initial_value);
  return ::cuda::coop::tile::exclusive_scan(input, scalar_initial, policy);
}

#  define _CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(_Functor, _Token, _IntegralOnly)                   \
    template <typename T, size_t N, typename OpT, scan_algorithm Algorithm = scan_algorithm::raking> \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> inclusive_scan(               \
      ct::tile<T, ct::shape<N>> input, _Functor<OpT>, scan_policy<Algorithm> = {}) noexcept          \
    {                                                                                                \
      detail::validate_functor<T, OpT>();                                                            \
      static_assert(!_IntegralOnly || ::cuda::std::is_integral_v<T>,                                 \
                    "cuda::coop::tile bitwise scans require an integer tile dtype");                 \
      return detail::scan_call<"inclusive", _Token, 0, 0, T, N, Algorithm>(input, T{}, T{});         \
    }

_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::std::plus, "sum", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::std::multiplies, "mul", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::minimum, "min", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::maximum, "max", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::std::bit_and, "bit_and", true)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::std::bit_or, "bit_or", true)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD(::cuda::std::bit_xor, "bit_xor", true)

#  undef _CCCL_COOP_TILE_INCLUSIVE_SCAN_OVERLOAD

#  define _CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(_Functor, _Token, _IntegralOnly)                              \
    template <typename T, size_t N, typename OpT, typename PrefixT, scan_algorithm Algorithm = scan_algorithm::raking> \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> inclusive_scan(                                 \
      ct::tile<T, ct::shape<N>> input, _Functor<OpT>, PrefixT prefix_value, scan_policy<Algorithm> = {}) noexcept      \
    {                                                                                                                  \
      detail::validate_functor<T, OpT>();                                                                              \
      static_assert(!_IntegralOnly || ::cuda::std::is_integral_v<T>,                                                   \
                    "cuda::coop::tile bitwise scans require an integer tile dtype");                                   \
      return detail::scan_call<"inclusive", _Token, 0, 1, T, N, Algorithm>(                                            \
        input, T{}, detail::normalize_value_control<T>(prefix_value));                                                 \
    }

_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::std::plus, "sum", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::std::multiplies, "mul", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::minimum, "min", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::maximum, "max", false)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::std::bit_and, "bit_and", true)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::std::bit_or, "bit_or", true)
_CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD(::cuda::std::bit_xor, "bit_xor", true)

#  undef _CCCL_COOP_TILE_INCLUSIVE_SCAN_PREFIX_OVERLOAD

/**
 * @brief Compute an inclusive prefix, preserving shape and dtype.
 *
 * Omitting the operation selects addition. An operation may follow `input`, followed by an optional
 * runtime prefix. Inclusive scans accept prefixes but no initial value.
 */
template <typename T, size_t N, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
inclusive_scan(ct::tile<T, ct::shape<N>> input, scan_policy<Algorithm> policy = {}) noexcept
{
  return ::cuda::coop::tile::inclusive_sum(input, policy);
}

template <typename T, size_t N, typename PrefixT, scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
inclusive_scan(ct::tile<T, ct::shape<N>> input, PrefixT prefix_value, scan_policy<Algorithm> policy = {}) noexcept
{
  return ::cuda::coop::tile::inclusive_sum(input, prefix_value, policy);
}

#  define _CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(_Functor, _Token, _IntegralOnly)             \
    template <typename T, size_t N, typename OpT, scan_algorithm Algorithm = scan_algorithm::raking>   \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> exclusive_scan(                 \
      ct::tile<T, ct::shape<N>> input,                                                                 \
      _Functor<OpT>,                                                                                   \
      ::cuda::std::type_identity_t<T> initial_value,                                                   \
      scan_policy<Algorithm> = {}) noexcept                                                            \
    {                                                                                                  \
      detail::validate_functor<T, OpT>();                                                              \
      static_assert(!_IntegralOnly || ::cuda::std::is_integral_v<T>,                                   \
                    "cuda::coop::tile bitwise scans require an integer tile dtype");                   \
      return detail::scan_call<"exclusive", _Token, 1, 0, T, N, Algorithm>(input, initial_value, T{}); \
    }                                                                                                  \
    template <typename T, size_t N, typename OpT, scan_algorithm Algorithm = scan_algorithm::raking>   \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> exclusive_scan(                 \
      ct::tile<T, ct::shape<N>> input,                                                                 \
      _Functor<OpT> op,                                                                                \
      ct::tile<T, ct::shape<1>> initial_value,                                                         \
      scan_policy<Algorithm> policy = {}) noexcept                                                     \
    {                                                                                                  \
      const T scalar_initial = static_cast<T>(initial_value);                                          \
      return ::cuda::coop::tile::exclusive_scan(input, op, scalar_initial, policy);                    \
    }

_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::std::plus, "sum", false)
_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::std::multiplies, "mul", false)
_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::minimum, "min", false)
_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::maximum, "max", false)
_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::std::bit_and, "bit_and", true)
_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::std::bit_or, "bit_or", true)
_CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD(::cuda::std::bit_xor, "bit_xor", true)

#  undef _CCCL_COOP_TILE_EXCLUSIVE_SCAN_INITIAL_OVERLOAD

#  define _CCCL_COOP_TILE_SCAN_OVERLOADS(_Functor, _Token, _IntegralOnly, _ExclusiveWithoutInitial)      \
    template <scan_mode Mode = scan_mode::exclusive,                                                     \
              typename T,                                                                                \
              size_t N,                                                                                  \
              typename OpT,                                                                              \
                                                                                                         \
              scan_algorithm Algorithm = scan_algorithm::raking>                                         \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> scan(                             \
      ct::tile<T, ct::shape<N>> input, _Functor<OpT>, scan_policy<Algorithm> = {}) noexcept              \
    {                                                                                                    \
      detail::validate_functor<T, OpT>();                                                                \
      static_assert(!_IntegralOnly || ::cuda::std::is_integral_v<T>,                                     \
                    "cuda::coop::tile bitwise scans require an integer tile dtype");                     \
      if constexpr (Mode == scan_mode::exclusive)                                                        \
      {                                                                                                  \
        static_assert(_ExclusiveWithoutInitial || Mode != scan_mode::exclusive,                          \
                      "cuda::coop::tile exclusive scans with this operation require an initial value");  \
        return detail::scan_call<"exclusive", _Token, 0, 0, T, N, Algorithm>(input, T{}, T{});           \
      }                                                                                                  \
      else                                                                                               \
      {                                                                                                  \
        static_assert(Mode == scan_mode::inclusive);                                                     \
        return detail::scan_call<"inclusive", _Token, 0, 0, T, N, Algorithm>(input, T{}, T{});           \
      }                                                                                                  \
    }                                                                                                    \
    template <scan_mode Mode = scan_mode::exclusive,                                                     \
              typename T,                                                                                \
              size_t N,                                                                                  \
              typename OpT,                                                                              \
                                                                                                         \
              scan_algorithm Algorithm = scan_algorithm::raking>                                         \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> scan(                             \
      ct::tile<T, ct::shape<N>> input,                                                                   \
      _Functor<OpT>,                                                                                     \
      ::cuda::std::type_identity_t<T> initial_value,                                                     \
      scan_policy<Algorithm> = {}) noexcept                                                              \
    {                                                                                                    \
      detail::validate_functor<T, OpT>();                                                                \
      static_assert(!_IntegralOnly || ::cuda::std::is_integral_v<T>,                                     \
                    "cuda::coop::tile bitwise scans require an integer tile dtype");                     \
      if constexpr (Mode == scan_mode::exclusive)                                                        \
      {                                                                                                  \
        return detail::scan_call<"exclusive", _Token, 1, 0, T, N, Algorithm>(input, initial_value, T{}); \
      }                                                                                                  \
      else                                                                                               \
      {                                                                                                  \
        static_assert(Mode == scan_mode::inclusive);                                                     \
        return detail::scan_call<"inclusive", _Token, 0, 1, T, N, Algorithm>(input, T{}, initial_value); \
      }                                                                                                  \
    }                                                                                                    \
    template <scan_mode Mode = scan_mode::exclusive,                                                     \
              typename T,                                                                                \
              size_t N,                                                                                  \
              typename OpT,                                                                              \
                                                                                                         \
              scan_algorithm Algorithm = scan_algorithm::raking>                                         \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> scan(                             \
      ct::tile<T, ct::shape<N>> input,                                                                   \
      _Functor<OpT> op,                                                                                  \
      ct::tile<T, ct::shape<1>> initial_value,                                                           \
      scan_policy<Algorithm> policy = {}) noexcept                                                       \
    {                                                                                                    \
      const T scalar_initial = static_cast<T>(initial_value);                                            \
      return ::cuda::coop::tile::scan<Mode>(input, op, scalar_initial, policy);                          \
    }

_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::std::plus, "sum", false, true)
_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::std::multiplies, "mul", false, false)
_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::minimum, "min", false, false)
_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::maximum, "max", false, false)
_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::std::bit_and, "bit_and", true, false)
_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::std::bit_or, "bit_or", true, false)
_CCCL_COOP_TILE_SCAN_OVERLOADS(::cuda::std::bit_xor, "bit_xor", true, false)

#  undef _CCCL_COOP_TILE_SCAN_OVERLOADS

template <scan_mode Mode = scan_mode::exclusive,
          typename T,
          size_t N,
          typename ControlT,
          scan_algorithm Algorithm = scan_algorithm::raking>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
scan(ct::tile<T, ct::shape<N>> input, ControlT seed_or_prefix, scan_policy<Algorithm> policy = {}) noexcept
{
  const T scalar_value = detail::normalize_value_control<T>(seed_or_prefix);
  return ::cuda::coop::tile::scan<Mode>(input, ::cuda::std::plus<>{}, scalar_value, policy);
}

/**
 * @brief Return the sum repeated across the input shape, using the input dtype for accumulation.
 *
 * `T` must be a supported numeric type. For valid Tile shapes, `N` must be
 * 1, 2, 4, 8, 16 or a positive multiple of 32.
 */
template <typename T, size_t N, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
  requires(detail::supported_element_v<T>)
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
sum(ct::tile<T, ct::shape<N>> input, reduce_policy<Algorithm> = {}) noexcept
{
  return ct::broadcast(detail::sum_batched_call<T, N, 1, Algorithm>(input), ct::shape<N>{});
}

/**
 * @brief Sum along a compile-time axis, retaining that dimension with extent one.
 *
 * `Axis` is nonnegative and less than the input rank. The input dtype is used
 * for accumulation and output. Independent reductions execute through native batching.
 * For valid Tile shapes, the reduced-axis extent must be 1, 2, 4, 8, 16 or a positive multiple of 32.
 */
template <size_t Axis, typename T, typename Shape, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
  requires(detail::supported_element_v<T> && Axis < Shape::rank())
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::reduction_result_t<ct::tile<T, Shape>, Axis>
sum(ct::tile<T, Shape> input, reduce_policy<Algorithm> = {}) noexcept
{
  constexpr size_t count   = ct::tile_size_v<ct::tile<T, Shape>>;
  constexpr size_t batches = count / Shape::static_extent(Axis);
  using Result             = ct::reduction_result_t<ct::tile<T, Shape>, Axis>;
  const auto rows          = detail::reduction_rows<Axis>(input, ::cuda::std::make_index_sequence<Shape::rank() - 1>{});
  return ct::reshape(detail::sum_batched_call<T, count, batches, Algorithm>(rows), typename Result::shape_type{});
}

/**
 * @brief Repeat the aggregate across the input shape.
 *
 * Omitting the operation selects addition; otherwise an operation follows `input`. `T` must be
 * numeric and a typed operation must match it. Integer bitwise operations additionally require
 * integral `T`. For valid Tile shapes, `N` must be 1, 2, 4, 8, 16 or a positive multiple of 32.
 */
template <typename T, size_t N, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
  requires(detail::supported_element_v<T>)
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
reduce(ct::tile<T, ct::shape<N>> input, reduce_policy<Algorithm> policy = {}) noexcept
{
  return ::cuda::coop::tile::sum(input, policy);
}

template <typename T, size_t N, typename OpT, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
  requires(detail::supported_element_v<T> && detail::compatible_functor_v<T, OpT>)
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
reduce(ct::tile<T, ct::shape<N>> input, ::cuda::std::plus<OpT>, reduce_policy<Algorithm> policy = {}) noexcept
{
  detail::validate_functor<T, OpT>();
  return ::cuda::coop::tile::sum(input, policy);
}

/**
 * @brief Reduce along a compile-time axis, retaining that dimension with extent one.
 *
 * Omitting the operation selects addition. Supported operations and dtype
 * constraints match the one-dimensional broadcast overloads. For valid Tile shapes,
 * the reduced-axis extent must be 1, 2, 4, 8, 16 or a positive multiple of 32.
 */
template <size_t Axis, typename T, typename Shape, reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
  requires(detail::supported_element_v<T> && Axis < Shape::rank())
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::reduction_result_t<ct::tile<T, Shape>, Axis>
reduce(ct::tile<T, Shape> input, reduce_policy<Algorithm> policy = {}) noexcept
{
  return ::cuda::coop::tile::sum<Axis>(input, policy);
}

template <size_t Axis,
          typename T,
          typename Shape,
          typename OpT,
          reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>
  requires(detail::supported_element_v<T> && detail::compatible_functor_v<T, OpT> && Axis < Shape::rank())
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::reduction_result_t<ct::tile<T, Shape>, Axis>
reduce(ct::tile<T, Shape> input, ::cuda::std::plus<OpT>, reduce_policy<Algorithm> policy = {}) noexcept
{
  return ::cuda::coop::tile::sum<Axis>(input, policy);
}

#  define _CCCL_COOP_TILE_REDUCE_OVERLOAD(_Functor, _Token, _IntegralOnly)                                          \
    template <typename T,                                                                                           \
              size_t N,                                                                                             \
              typename OpT,                                                                                         \
                                                                                                                    \
              reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>                                       \
      requires(detail::supported_element_v<T> && detail::compatible_functor_v<T, OpT>                               \
               && (!_IntegralOnly || ::cuda::std::is_integral_v<T>) )                                               \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> reduce(                                      \
      ct::tile<T, ct::shape<N>> input, _Functor<OpT>, reduce_policy<Algorithm> = {}) noexcept                       \
    {                                                                                                               \
      detail::validate_functor<T, OpT>();                                                                           \
      return ct::broadcast(detail::reduce_batched_call<_Token, _IntegralOnly, T, N, 1, Algorithm>(input),           \
                           ct::shape<N>{});                                                                         \
    }                                                                                                               \
    template <size_t Axis,                                                                                          \
              typename T,                                                                                           \
              typename Shape,                                                                                       \
              typename OpT,                                                                                         \
              reduce_algorithm Algorithm = reduce_algorithm::warp_reductions>                                       \
      requires(detail::supported_element_v<T> && detail::compatible_functor_v<T, OpT>                               \
               && (!_IntegralOnly || ::cuda::std::is_integral_v<T>) && Axis < Shape::rank())                        \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::reduction_result_t<ct::tile<T, Shape>, Axis> reduce(               \
      ct::tile<T, Shape> input, _Functor<OpT>, reduce_policy<Algorithm> = {}) noexcept                              \
    {                                                                                                               \
      constexpr size_t count   = ct::tile_size_v<ct::tile<T, Shape>>;                                               \
      constexpr size_t batches = count / Shape::static_extent(Axis);                                                \
      using Result             = ct::reduction_result_t<ct::tile<T, Shape>, Axis>;                                  \
      const auto rows = detail::reduction_rows<Axis>(input, ::cuda::std::make_index_sequence<Shape::rank() - 1>{}); \
      return ct::reshape(detail::reduce_batched_call<_Token, _IntegralOnly, T, count, batches, Algorithm>(rows),    \
                         typename Result::shape_type{});                                                            \
    }

_CCCL_COOP_TILE_REDUCE_OVERLOAD(::cuda::std::multiplies, "mul", false)
_CCCL_COOP_TILE_REDUCE_OVERLOAD(::cuda::minimum, "min", false)
_CCCL_COOP_TILE_REDUCE_OVERLOAD(::cuda::maximum, "max", false)
_CCCL_COOP_TILE_REDUCE_OVERLOAD(::cuda::std::bit_and, "bit_and", true)
_CCCL_COOP_TILE_REDUCE_OVERLOAD(::cuda::std::bit_or, "bit_or", true)
_CCCL_COOP_TILE_REDUCE_OVERLOAD(::cuda::std::bit_xor, "bit_xor", true)

#  undef _CCCL_COOP_TILE_REDUCE_OVERLOAD

/**
 * @brief Count unsigned-byte samples into an `int` Tile of extent `Bins`.
 *
 * `N` must be a positive multiple of 32.
 * `T` must be `unsigned char` and each sample must be less than `Bins`. `Bins` is 32, 64, 128, or
 * 256, as required by the CUDA Tile shape rules. The selected worker count must divide both
 * `N` and `Bins`. Counts start at zero for each call.
 */
template <size_t Bins = 256, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<Bins>>
histogram(ct::tile<T, ct::shape<N>> input, histogram_policy<> = {}) noexcept
{
  if constexpr (::cuda::std::is_same_v<T, unsigned char>)
  {
    return detail::histogram_call<Bins, N>(input);
  }
  else
  {
    static_assert(::cuda::std::is_same_v<T, unsigned char>,
                  "cuda::coop::tile Histogram input dtype must be unsigned char (u8)");
    return {};
  }
}

/**
 * @brief Count unsigned-byte samples into all 256 bins.
 *
 * This is the fixed-bin counterpart of `histogram<256>` with the same input and result contract.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<256>>
histogram_256(ct::tile<T, ct::shape<N>> input, histogram_policy<> = {}) noexcept
{
  if constexpr (::cuda::std::is_same_v<T, unsigned char>)
  {
    return detail::histogram_256_call<N>(input);
  }
  else
  {
    static_assert(::cuda::std::is_same_v<T, unsigned char>,
                  "cuda::coop::tile Histogram-256 input dtype must be unsigned char (u8)");
    return {};
  }
}

/**
 * @brief Expand runs into a value Tile of extent `DecodedN`.
 *
 * Input extents must match. `LengthT` is `unsigned int`, `unsigned long`, or `unsigned long long`;
 * run-length sums must fit that type. Zero-length runs may only follow all positive runs.
 * An overload accepts `decoded_window_offset` between
 * `run_lengths` and the policy: an integral scalar convertible to `LengthT` or `ct::tile<LengthT,
 * ct::shape<1>>`. The default offset is zero. Use a nonnegative representable offset. Output
 * beyond the expanded sequence is zero.
 *
 * Both input extents and `DecodedN` must be positive multiples of 32.
 */
template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<ValueT, ct::shape<DecodedN>> run_length_decode(
  ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
  ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
  run_length_decode_policy<> = {}) noexcept
{
  return detail::run_length_decode_checked<DecodedN, ValueT, LengthT, ValueRunN, LengthRunN>(
    run_values, run_lengths, LengthT{});
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN, typename OffsetControlT>
  requires(detail::compatible_run_length_decode_offset_v<LengthT, OffsetControlT>)
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<ValueT, ct::shape<DecodedN>> run_length_decode(
  ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
  ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
  OffsetControlT decoded_window_offset,
  run_length_decode_policy<> = {}) noexcept
{
  const LengthT scalar_offset = detail::normalize_run_length_decode_offset<LengthT>(decoded_window_offset);
  return detail::run_length_decode_checked<DecodedN, ValueT, LengthT, ValueRunN, LengthRunN>(
    run_values, run_lengths, scalar_offset);
}

/**
 * @brief Return decoded values and relative offsets within their runs, in that order.
 *
 * The inputs, window-offset overload, and capacity follow `run_length_decode`. Offsets have
 * element type `LengthT`; padded positions contain its maximum value. Both input and output
 * extents must admit a common worker count.
 */
template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN>
[[nodiscard]] _CCCL_COOP_TILE_API inline run_length_decode_with_offsets_result<ValueT, LengthT, DecodedN>
run_length_decode_with_offsets(ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
                               ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
                               run_length_decode_policy<> = {}) noexcept
{
  return detail::run_length_decode_with_offsets_checked<DecodedN, ValueT, LengthT, ValueRunN, LengthRunN>(
    run_values, run_lengths, LengthT{});
}

template <size_t DecodedN, typename ValueT, typename LengthT, size_t ValueRunN, size_t LengthRunN, typename OffsetControlT>
  requires(detail::compatible_run_length_decode_offset_v<LengthT, OffsetControlT>)
[[nodiscard]] _CCCL_COOP_TILE_API inline run_length_decode_with_offsets_result<ValueT, LengthT, DecodedN>
run_length_decode_with_offsets(
  ct::tile<ValueT, ct::shape<ValueRunN>> run_values,
  ct::tile<LengthT, ct::shape<LengthRunN>> run_lengths,
  OffsetControlT decoded_window_offset,
  run_length_decode_policy<> = {}) noexcept
{
  const LengthT scalar_offset = detail::normalize_run_length_decode_offset<LengthT>(decoded_window_offset);
  return detail::run_length_decode_with_offsets_checked<DecodedN, ValueT, LengthT, ValueRunN, LengthRunN>(
    run_values, run_lengths, scalar_offset);
}

/**
 * @brief Return an `int` flag Tile: one at the first position and where a value differs from its
 * predecessor; zero elsewhere.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>>
discontinuity_flag_heads(ct::tile<T, ct::shape<N>> input, discontinuity_policy<> = {}) noexcept
{
  return detail::discontinuity_flags_call<"heads", T, N>(input);
}

/**
 * @brief Return an `int` flag Tile: one at the last position and where a value differs from its
 * successor; zero elsewhere.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>>
discontinuity_flag_tails(ct::tile<T, ct::shape<N>> input, discontinuity_policy<> = {}) noexcept
{
  return detail::discontinuity_flags_call<"tails", T, N>(input);
}

/**
 * @brief Return two `int` flag Tiles, heads first and tails second.
 *
 * Use a structured binding. All run boundaries are local to the input Tile.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline discontinuity_pair_result<N>
discontinuity_flag_heads_and_tails(ct::tile<T, ct::shape<N>> input, discontinuity_policy<> = {}) noexcept
{
  return detail::discontinuity_heads_and_tails_call<T, N>(input);
}

/**
 * @brief Return the current value minus its predecessor; preserve the first input value.
 *
 * Shape and dtype are unchanged. No custom operation or external predecessor is accepted.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
adjacent_difference_subtract_left(ct::tile<T, ct::shape<N>> input, adjacent_difference_policy<> = {}) noexcept
{
  return detail::adjacent_difference_call<"subtract_left", T, N>(input);
}

/**
 * @brief Return the current value minus its successor; preserve the last input value.
 *
 * Shape and dtype are unchanged. No custom operation or external successor is accepted.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
adjacent_difference_subtract_right(ct::tile<T, ct::shape<N>> input, adjacent_difference_policy<> = {}) noexcept
{
  return detail::adjacent_difference_call<"subtract_right", T, N>(input);
}

/**
 * @brief Read the preceding element and preserve the first input element.
 *
 * Shape and dtype are unchanged. Multiple items per physical worker are supported.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_up(ct::tile<T, ct::shape<N>> input, shuffle_policy<> = {}) noexcept
{
  return detail::shuffle_call<"up", false, T, N>(input, 1);
}

/**
 * @brief Read the following element and preserve the last input element.
 *
 * Shape and dtype are unchanged. Multiple items per physical worker are supported.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_down(ct::tile<T, ct::shape<N>> input, shuffle_policy<> = {}) noexcept
{
  return detail::shuffle_call<"down", false, T, N>(input, 1);
}

/**
 * @brief Return `input[i + distance]` when the source is in range; otherwise preserve `input[i]`.
 *
 * Require `-N < distance < N` and one physical worker per logical item. Another overload accepts
 * `ct::tile<int, ct::shape<1>>` for `distance`.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_offset(ct::tile<T, ct::shape<N>> input, int distance = 1, shuffle_policy<> = {}) noexcept
{
  return detail::shuffle_call<"offset", true, T, N>(input, distance);
}

template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_offset(ct::tile<T, ct::shape<N>> input, ct::tile<int, ct::shape<1>> distance, shuffle_policy<> = {}) noexcept
{
  return detail::shuffle_call<"offset", true, T, N>(input, detail::normalize_i32_control(distance));
}

/**
 * @brief Return `input[(i + distance) % N]`.
 *
 * Require `0 <= distance < N` and one physical worker per logical item. Another overload accepts
 * `ct::tile<int, ct::shape<1>>` for `distance`.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_rotate(ct::tile<T, ct::shape<N>> input, int distance = 1, shuffle_policy<> = {}) noexcept
{
  return detail::shuffle_call<"rotate", true, T, N>(input, distance);
}

template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
shuffle_rotate(ct::tile<T, ct::shape<N>> input, ct::tile<int, ct::shape<1>> distance, shuffle_policy<> = {}) noexcept
{
  return detail::shuffle_call<"rotate", true, T, N>(input, detail::normalize_i32_control(distance));
}

/**
 * @brief For `B = BlockThreads` and `I = N / B`, return `output[t * I + i] = input[i * B + t]`.
 *
 * Supply `exchange_policy<B>{}` explicitly so `B` can be deduced. The default function argument
 * does not supply a default width.
 */
template <typename T, size_t N, size_t BlockThreads>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exchange_striped_to_blocked(ct::tile<T, ct::shape<N>> input, exchange_policy<BlockThreads> = {}) noexcept
{
  // The Tile fragment map is blocked. CUB's BlockedToStriped permutation
  // therefore converts values that are logically striped into blocked order.
  return detail::exchange_call<"blocked_to_striped", T, N, BlockThreads>(input);
}

/**
 * @brief Return the inverse permutation, `output[i * B + t] = input[t * I + i]`.
 *
 * Supply the same explicit `exchange_policy<B>{}` as the forward conversion.
 */
template <typename T, size_t N, size_t BlockThreads>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
exchange_blocked_to_striped(ct::tile<T, ct::shape<N>> input, exchange_policy<BlockThreads> = {}) noexcept
{
  // The inverse conversion likewise uses CUB's StripedToBlocked permutation.
  return detail::exchange_call<"striped_to_blocked", T, N, BlockThreads>(input);
}

/**
 * @brief Return an `int` destination rank per original input item for one stable radix digit.
 *
 * Keys must be integral. Overloads accept no bounds, `begin_bit` alone, or `begin_bit, end_bit`
 * before the policy. Scalar controls use `int`; one or both bounds may instead be shape-one `int`
 * Tiles. The omitted end is `min(key_bits, begin_bit + RadixBits)`. Signed keys have their sign bit flipped
 * before extraction, matching Radix Sort's signed-key ordering. Unsigned keys use their original bits.
 * Require N <= 65535, 0 <= begin_bit < end_bit <= key_bits, and end_bit - begin_bit <= RadixBits.
 * Invalid runtime bounds trap.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T,
          size_t N,
          size_t RadixBits                  = 4,
          bool Descending                   = false,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>> radix_rank_keys(
  ct::tile<T, ct::shape<N>> keys,
  radix_rank_policy<RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> = {}) noexcept
{
  return detail::radix_rank_call<T, N, RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>(
    keys, 0, detail::default_radix_rank_end<T, RadixBits>(0));
}

template <typename T,
          size_t N,
          size_t RadixBits                  = 4,
          bool Descending                   = false,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>> radix_rank_keys(
  ct::tile<T, ct::shape<N>> keys,
  int begin_bit,
  radix_rank_policy<RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> = {}) noexcept
{
  return detail::radix_rank_call<T, N, RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>(
    keys, begin_bit, detail::default_radix_rank_end<T, RadixBits>(begin_bit));
}

template <typename T,
          size_t N,
          size_t RadixBits                  = 4,
          bool Descending                   = false,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>> radix_rank_keys(
  ct::tile<T, ct::shape<N>> keys,
  int begin_bit,
  int end_bit,
  radix_rank_policy<RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> = {}) noexcept
{
  return detail::radix_rank_call<T, N, RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>(
    keys, begin_bit, end_bit);
}

template <typename T,
          typename BeginControlT,
          size_t N,
          size_t RadixBits                  = 4,
          bool Descending                   = false,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
  requires(detail::compatible_i32_control_v<BeginControlT> && detail::shape_one_i32_control_v<BeginControlT>)
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>> radix_rank_keys(
  ct::tile<T, ct::shape<N>> keys,
  BeginControlT begin_bit,
  radix_rank_policy<RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return ::cuda::coop::tile::radix_rank_keys(keys, detail::normalize_i32_control(begin_bit), policy);
}

template <typename T,
          typename BeginControlT,
          typename EndControlT,
          size_t N,
          size_t RadixBits                  = 4,
          bool Descending                   = false,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
  requires(detail::compatible_i32_control_v<BeginControlT> && detail::compatible_i32_control_v<EndControlT>
           && (detail::shape_one_i32_control_v<BeginControlT> || detail::shape_one_i32_control_v<EndControlT>) )
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<int, ct::shape<N>> radix_rank_keys(
  ct::tile<T, ct::shape<N>> keys,
  BeginControlT begin_bit,
  EndControlT end_bit,
  radix_rank_policy<RadixBits, Descending, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return ::cuda::coop::tile::radix_rank_keys(
    keys, detail::normalize_i32_control(begin_bit), detail::normalize_i32_control(end_bit), policy);
}

namespace detail
{
template <size_t Descending, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
merge_sort_keys(ct::tile<T, ct::shape<N>> keys, merge_sort_policy<>) noexcept
{
  return detail::merge_sort_keys_call<Descending, 0, T, N>(keys, static_cast<int>(N), T{});
}

template <size_t Descending, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> merge_sort_keys(
  ct::tile<T, ct::shape<N>> keys,
  int valid_items,
  ::cuda::std::type_identity_t<T> oob_default,
  merge_sort_policy<>) noexcept
{
  return detail::merge_sort_keys_call<Descending, 1, T, N>(keys, valid_items, oob_default);
}

template <size_t Descending, size_t PartialTile, typename KeyT, typename ValueT, size_t KeyN, size_t ValueN>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs_checked(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  int valid_items,
  KeyT oob_default,
  merge_sort_policy<>) noexcept
{
  static_assert(KeyN == ValueN, "cuda::coop::tile Merge Sort key and value tiles must have matching extents");
  if constexpr (KeyN == ValueN)
  {
    return detail::merge_sort_pairs_call<Descending, PartialTile, KeyT, ValueT, KeyN>(
      keys, values, valid_items, oob_default);
  }
  else
  {
    return {};
  }
}

template <size_t Descending, typename KeyT, typename ValueT, size_t KeyN, size_t ValueN>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys, ct::tile<ValueT, ct::shape<ValueN>> values, merge_sort_policy<> policy) noexcept
{
  return detail::merge_sort_pairs_checked<Descending, 0>(keys, values, static_cast<int>(KeyN), KeyT{}, policy);
}

template <size_t Descending, typename KeyT, typename ValueT, size_t KeyN, size_t ValueN>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  int valid_items,
  ::cuda::std::type_identity_t<KeyT> oob_default,
  merge_sort_policy<> policy) noexcept
{
  return detail::merge_sort_pairs_checked<Descending, 1>(keys, values, valid_items, oob_default, policy);
}
} // namespace detail

/**
 * @brief Return keys in ascending numeric order.
 *
 * Overloads optionally accept `cuda::std::less<CompareT>` or `cuda::std::greater<CompareT>` after
 * `keys`, followed optionally by the pair `valid_items, oob_default`. The count is an int scalar
 * or shape-one int Tile; the padding key is convertible to the key dtype or a shape-one Tile of
 * that dtype. Require 0 <= valid_items <= N and a strict padding sentinel when consuming the valid
 * prefix.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
merge_sort_keys(ct::tile<T, ct::shape<N>> keys, merge_sort_policy<> policy = {}) noexcept
{
  return detail::merge_sort_keys<0>(keys, policy);
}

template <typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> merge_sort_keys(
  ct::tile<T, ct::shape<N>> keys,
  int valid_items,
  ::cuda::std::type_identity_t<T> oob_default,
  merge_sort_policy<> policy = {}) noexcept
{
  return detail::merge_sort_keys<0>(keys, valid_items, oob_default, policy);
}

#  define _CCCL_COOP_TILE_MERGE_SORT_KEYS_COMPARE_OVERLOADS(_Compare, _Descending)                               \
    template <typename T, size_t N, typename CompareT>                                                           \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> merge_sort_keys(                          \
      ct::tile<T, ct::shape<N>> keys, _Compare<CompareT>, merge_sort_policy<> policy = {}) noexcept              \
    {                                                                                                            \
      detail::validate_functor<T, CompareT>();                                                                   \
      return detail::merge_sort_keys<_Descending>(keys, policy);                                                 \
    }                                                                                                            \
    template <typename T, size_t N, typename CompareT>                                                           \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> merge_sort_keys(                          \
      ct::tile<T, ct::shape<N>> keys,                                                                            \
      _Compare<CompareT>,                                                                                        \
      int valid_items,                                                                                           \
      ::cuda::std::type_identity_t<T> oob_default,                                                               \
      merge_sort_policy<> policy = {}) noexcept                                                                  \
    {                                                                                                            \
      detail::validate_functor<T, CompareT>();                                                                   \
      return detail::merge_sort_keys<_Descending>(keys, valid_items, oob_default, policy);                       \
    }                                                                                                            \
    template <typename T, size_t N, typename CompareT, typename ValidItemsT, typename OobDefaultT>               \
      requires(                                                                                                  \
        detail::compatible_i32_control_v<ValidItemsT> && detail::compatible_value_control_v<T, OobDefaultT>      \
        && (detail::shape_one_i32_control_v<ValidItemsT> || detail::shape_one_value_control_v<T, OobDefaultT>) ) \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> merge_sort_keys(                          \
      ct::tile<T, ct::shape<N>> keys,                                                                            \
      _Compare<CompareT> compare,                                                                                \
      ValidItemsT valid_items,                                                                                   \
      OobDefaultT oob_default,                                                                                   \
      merge_sort_policy<> policy = {}) noexcept                                                                  \
    {                                                                                                            \
      const int scalar_valid_items = detail::normalize_i32_control(valid_items);                                 \
      const T scalar_oob_default   = detail::normalize_value_control<T>(oob_default);                            \
      return ::cuda::coop::tile::merge_sort_keys(keys, compare, scalar_valid_items, scalar_oob_default, policy); \
    }

_CCCL_COOP_TILE_MERGE_SORT_KEYS_COMPARE_OVERLOADS(::cuda::std::less, 0)
_CCCL_COOP_TILE_MERGE_SORT_KEYS_COMPARE_OVERLOADS(::cuda::std::greater, 1)

#  undef _CCCL_COOP_TILE_MERGE_SORT_KEYS_COMPARE_OVERLOADS

template <typename T, typename ValidItemsT, typename OobDefaultT, size_t N>
  requires(detail::compatible_i32_control_v<ValidItemsT> && detail::compatible_value_control_v<T, OobDefaultT>
           && (detail::shape_one_i32_control_v<ValidItemsT> || detail::shape_one_value_control_v<T, OobDefaultT>) )
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> merge_sort_keys(
  ct::tile<T, ct::shape<N>> keys,
  ValidItemsT valid_items,
  OobDefaultT oob_default,
  merge_sort_policy<> policy = {}) noexcept
{
  const int scalar_valid_items = detail::normalize_i32_control(valid_items);
  const T scalar_oob_default   = detail::normalize_value_control<T>(oob_default);
  return ::cuda::coop::tile::merge_sort_keys(keys, scalar_valid_items, scalar_oob_default, policy);
}

/**
 * @brief Sort keys and associated payloads, returning keys first.
 *
 * The overload forms match `merge_sort_keys`, with `values` immediately after `keys`. Key and
 * value extents must match; dtypes may differ. Partial sorting zeroes padded payloads. Equal keys
 * have no stability guarantee.
 *
 * Each input extent must be a positive multiple of 32.
 */
template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  merge_sort_policy<> policy = {}) noexcept
{
  return detail::merge_sort_pairs<0>(keys, values, policy);
}

template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN>
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  int valid_items,
  ::cuda::std::type_identity_t<KeyT> oob_default,
  merge_sort_policy<> policy = {}) noexcept
{
  return detail::merge_sort_pairs<0>(keys, values, valid_items, oob_default, policy);
}

#  define _CCCL_COOP_TILE_MERGE_SORT_PAIRS_COMPARE_OVERLOADS(_Compare, _Descending)                                 \
    template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN, typename CompareT>                        \
    [[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(           \
      ct::tile<KeyT, ct::shape<KeyN>> keys,                                                                         \
      ct::tile<ValueT, ct::shape<ValueN>> values,                                                                   \
      _Compare<CompareT>,                                                                                           \
      merge_sort_policy<> policy = {}) noexcept                                                                     \
    {                                                                                                               \
      detail::validate_functor<KeyT, CompareT>();                                                                   \
      return detail::merge_sort_pairs<_Descending>(keys, values, policy);                                           \
    }                                                                                                               \
    template <typename KeyT, typename ValueT, size_t KeyN, size_t ValueN, typename CompareT>                        \
    [[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(           \
      ct::tile<KeyT, ct::shape<KeyN>> keys,                                                                         \
      ct::tile<ValueT, ct::shape<ValueN>> values,                                                                   \
      _Compare<CompareT>,                                                                                           \
      int valid_items,                                                                                              \
      ::cuda::std::type_identity_t<KeyT> oob_default,                                                               \
      merge_sort_policy<> policy = {}) noexcept                                                                     \
    {                                                                                                               \
      detail::validate_functor<KeyT, CompareT>();                                                                   \
      return detail::merge_sort_pairs<_Descending>(keys, values, valid_items, oob_default, policy);                 \
    }                                                                                                               \
    template <typename KeyT,                                                                                        \
              typename ValueT,                                                                                      \
              size_t KeyN,                                                                                          \
              size_t ValueN,                                                                                        \
              typename CompareT,                                                                                    \
              typename ValidItemsT,                                                                                 \
              typename OobDefaultT>                                                                                 \
      requires(                                                                                                     \
        detail::compatible_i32_control_v<ValidItemsT> && detail::compatible_value_control_v<KeyT, OobDefaultT>      \
        && (detail::shape_one_i32_control_v<ValidItemsT> || detail::shape_one_value_control_v<KeyT, OobDefaultT>) ) \
    [[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(           \
      ct::tile<KeyT, ct::shape<KeyN>> keys,                                                                         \
      ct::tile<ValueT, ct::shape<ValueN>> values,                                                                   \
      _Compare<CompareT> compare,                                                                                   \
      ValidItemsT valid_items,                                                                                      \
      OobDefaultT oob_default,                                                                                      \
      merge_sort_policy<> policy = {}) noexcept                                                                     \
    {                                                                                                               \
      const int scalar_valid_items  = detail::normalize_i32_control(valid_items);                                   \
      const KeyT scalar_oob_default = detail::normalize_value_control<KeyT>(oob_default);                           \
      return ::cuda::coop::tile::merge_sort_pairs(                                                                  \
        keys, values, compare, scalar_valid_items, scalar_oob_default, policy);                                     \
    }

_CCCL_COOP_TILE_MERGE_SORT_PAIRS_COMPARE_OVERLOADS(::cuda::std::less, 0)
_CCCL_COOP_TILE_MERGE_SORT_PAIRS_COMPARE_OVERLOADS(::cuda::std::greater, 1)

#  undef _CCCL_COOP_TILE_MERGE_SORT_PAIRS_COMPARE_OVERLOADS

template <typename KeyT, typename ValueT, typename ValidItemsT, typename OobDefaultT, size_t KeyN, size_t ValueN>
  requires(detail::compatible_i32_control_v<ValidItemsT> && detail::compatible_value_control_v<KeyT, OobDefaultT>
           && (detail::shape_one_i32_control_v<ValidItemsT> || detail::shape_one_value_control_v<KeyT, OobDefaultT>) )
[[nodiscard]] _CCCL_COOP_TILE_API inline merge_sort_pair_result<KeyT, ValueT, KeyN> merge_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  ValidItemsT valid_items,
  OobDefaultT oob_default,
  merge_sort_policy<> policy = {}) noexcept
{
  const int scalar_valid_items  = detail::normalize_i32_control(valid_items);
  const KeyT scalar_oob_default = detail::normalize_value_control<KeyT>(oob_default);
  return ::cuda::coop::tile::merge_sort_pairs(keys, values, scalar_valid_items, scalar_oob_default, policy);
}

namespace detail
{
template <size_t Descending,
          typename T,
          size_t N,
          size_t RadixBits,
          bool MemoizeOuterScan,
          scan_algorithm InnerScanAlgorithm,
          radix_smem_config SmemConfig>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> radix_sort_keys(
  ct::tile<T, ct::shape<N>> keys,
  int begin_bit,
  int end_bit,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>) noexcept
{
  return detail::radix_sort_keys_call<Descending, T, N, RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>(
    keys, begin_bit, end_bit);
}

template <size_t Descending,
          typename KeyT,
          typename ValueT,
          size_t KeyN,
          size_t ValueN,
          size_t RadixBits,
          bool MemoizeOuterScan,
          scan_algorithm InnerScanAlgorithm,
          radix_smem_config SmemConfig>
[[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  int begin_bit,
  int end_bit,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>) noexcept
{
  static_assert(KeyN == ValueN, "cuda::coop::tile Radix Sort key and value tiles must have matching extents");
  if constexpr (KeyN == ValueN)
  {
    return detail::
      radix_sort_pairs_call<Descending, KeyT, ValueT, KeyN, RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig>(
        keys, values, begin_bit, end_bit);
  }
  else
  {
    return {};
  }
}
} // namespace detail

/**
 * @brief Sort keys in ascending radix order over the full key width.
 *
 * Overloads optionally accept `cuda::std::less<CompareT>` or `cuda::std::greater<CompareT>` after
 * `keys`, followed optionally by `begin_bit, end_bit`. The bounds and comparison may each be
 * omitted independently. Bounds are int scalars or shape-one int Tiles. Require N <= 65535 and 0
 * <= begin_bit <= end_bit <= key_bits.
 *
 * `N` must be a positive multiple of 32.
 */
template <typename T,
          size_t N,
          size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
radix_sort_keys(ct::tile<T, ct::shape<N>> keys,
                radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return detail::radix_sort_keys<0>(keys, 0, static_cast<int>(sizeof(T) * 8), policy);
}

template <typename T,
          size_t N,
          size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> radix_sort_keys(
  ct::tile<T, ct::shape<N>> keys,
  int begin_bit,
  int end_bit,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return detail::radix_sort_keys<0>(keys, begin_bit, end_bit, policy);
}

#  define _CCCL_COOP_TILE_RADIX_SORT_KEYS_COMPARE_OVERLOADS(_Compare, _Descending)                                   \
    template <typename T,                                                                                            \
              typename CompareT,                                                                                     \
              size_t N,                                                                                              \
                                                                                                                     \
              size_t RadixBits                  = 4,                                                                 \
              bool MemoizeOuterScan             = true,                                                              \
              scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,                                        \
              radix_smem_config SmemConfig      = radix_smem_config::four_byte>                                      \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> radix_sort_keys(                              \
      ct::tile<T, ct::shape<N>> keys,                                                                                \
      _Compare<CompareT>,                                                                                            \
      radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept           \
    {                                                                                                                \
      detail::validate_functor<T, CompareT>();                                                                       \
      return detail::radix_sort_keys<_Descending>(keys, 0, static_cast<int>(sizeof(T) * 8), policy);                 \
    }                                                                                                                \
    template <typename T,                                                                                            \
              typename CompareT,                                                                                     \
              size_t N,                                                                                              \
                                                                                                                     \
              size_t RadixBits                  = 4,                                                                 \
              bool MemoizeOuterScan             = true,                                                              \
              scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,                                        \
              radix_smem_config SmemConfig      = radix_smem_config::four_byte>                                      \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> radix_sort_keys(                              \
      ct::tile<T, ct::shape<N>> keys,                                                                                \
      _Compare<CompareT>,                                                                                            \
      int begin_bit,                                                                                                 \
      int end_bit,                                                                                                   \
      radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept           \
    {                                                                                                                \
      detail::validate_functor<T, CompareT>();                                                                       \
      return detail::radix_sort_keys<_Descending>(keys, begin_bit, end_bit, policy);                                 \
    }                                                                                                                \
    template <typename T,                                                                                            \
              typename CompareT,                                                                                     \
              typename BeginControlT,                                                                                \
              typename EndControlT,                                                                                  \
              size_t N,                                                                                              \
                                                                                                                     \
              size_t RadixBits                  = 4,                                                                 \
              bool MemoizeOuterScan             = true,                                                              \
              scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,                                        \
              radix_smem_config SmemConfig      = radix_smem_config::four_byte>                                      \
      requires(detail::compatible_i32_control_v<BeginControlT> && detail::compatible_i32_control_v<EndControlT>      \
               && (detail::shape_one_i32_control_v<BeginControlT> || detail::shape_one_i32_control_v<EndControlT>) ) \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> radix_sort_keys(                              \
      ct::tile<T, ct::shape<N>> keys,                                                                                \
      _Compare<CompareT> compare,                                                                                    \
      BeginControlT begin_bit,                                                                                       \
      EndControlT end_bit,                                                                                           \
      radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept           \
    {                                                                                                                \
      return ::cuda::coop::tile::radix_sort_keys(                                                                    \
        keys, compare, detail::normalize_i32_control(begin_bit), detail::normalize_i32_control(end_bit), policy);    \
    }

_CCCL_COOP_TILE_RADIX_SORT_KEYS_COMPARE_OVERLOADS(::cuda::std::less, 0)
_CCCL_COOP_TILE_RADIX_SORT_KEYS_COMPARE_OVERLOADS(::cuda::std::greater, 1)

#  undef _CCCL_COOP_TILE_RADIX_SORT_KEYS_COMPARE_OVERLOADS

template <typename T,
          typename BeginControlT,
          typename EndControlT,
          size_t N,
          size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
  requires(detail::compatible_i32_control_v<BeginControlT> && detail::compatible_i32_control_v<EndControlT>
           && (detail::shape_one_i32_control_v<BeginControlT> || detail::shape_one_i32_control_v<EndControlT>) )
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> radix_sort_keys(
  ct::tile<T, ct::shape<N>> keys,
  BeginControlT begin_bit,
  EndControlT end_bit,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return ::cuda::coop::tile::radix_sort_keys(
    keys, detail::normalize_i32_control(begin_bit), detail::normalize_i32_control(end_bit), policy);
}

/**
 * @brief Sort keys and associated values, preserving equal-digit input order.
 *
 * Its overload forms match `radix_sort_keys`, with `values` immediately after `keys`. Key and
 * value extents must match; dtypes may differ.
 *
 * Each input extent must be a positive multiple of 32.
 */
template <typename KeyT,
          typename ValueT,
          size_t KeyN,
          size_t ValueN,
          size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return detail::radix_sort_pairs<0>(keys, values, 0, static_cast<int>(sizeof(KeyT) * 8), policy);
}

template <typename KeyT,
          typename ValueT,
          size_t KeyN,
          size_t ValueN,
          size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
[[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  int begin_bit,
  int end_bit,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return detail::radix_sort_pairs<0>(keys, values, begin_bit, end_bit, policy);
}

#  define _CCCL_COOP_TILE_RADIX_SORT_PAIRS_COMPARE_OVERLOADS(_Compare, _Descending)                                  \
    template <typename KeyT,                                                                                         \
              typename ValueT,                                                                                       \
              typename CompareT,                                                                                     \
              size_t KeyN,                                                                                           \
              size_t ValueN,                                                                                         \
                                                                                                                     \
              size_t RadixBits                  = 4,                                                                 \
              bool MemoizeOuterScan             = true,                                                              \
              scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,                                        \
              radix_smem_config SmemConfig      = radix_smem_config::four_byte>                                      \
    [[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(            \
      ct::tile<KeyT, ct::shape<KeyN>> keys,                                                                          \
      ct::tile<ValueT, ct::shape<ValueN>> values,                                                                    \
      _Compare<CompareT>,                                                                                            \
      radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept           \
    {                                                                                                                \
      detail::validate_functor<KeyT, CompareT>();                                                                    \
      return detail::radix_sort_pairs<_Descending>(keys, values, 0, static_cast<int>(sizeof(KeyT) * 8), policy);     \
    }                                                                                                                \
    template <typename KeyT,                                                                                         \
              typename ValueT,                                                                                       \
              typename CompareT,                                                                                     \
              size_t KeyN,                                                                                           \
              size_t ValueN,                                                                                         \
                                                                                                                     \
              size_t RadixBits                  = 4,                                                                 \
              bool MemoizeOuterScan             = true,                                                              \
              scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,                                        \
              radix_smem_config SmemConfig      = radix_smem_config::four_byte>                                      \
    [[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(            \
      ct::tile<KeyT, ct::shape<KeyN>> keys,                                                                          \
      ct::tile<ValueT, ct::shape<ValueN>> values,                                                                    \
      _Compare<CompareT>,                                                                                            \
      int begin_bit,                                                                                                 \
      int end_bit,                                                                                                   \
      radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept           \
    {                                                                                                                \
      detail::validate_functor<KeyT, CompareT>();                                                                    \
      return detail::radix_sort_pairs<_Descending>(keys, values, begin_bit, end_bit, policy);                        \
    }                                                                                                                \
    template <typename KeyT,                                                                                         \
              typename ValueT,                                                                                       \
              typename CompareT,                                                                                     \
              typename BeginControlT,                                                                                \
              typename EndControlT,                                                                                  \
              size_t KeyN,                                                                                           \
              size_t ValueN,                                                                                         \
                                                                                                                     \
              size_t RadixBits                  = 4,                                                                 \
              bool MemoizeOuterScan             = true,                                                              \
              scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,                                        \
              radix_smem_config SmemConfig      = radix_smem_config::four_byte>                                      \
      requires(detail::compatible_i32_control_v<BeginControlT> && detail::compatible_i32_control_v<EndControlT>      \
               && (detail::shape_one_i32_control_v<BeginControlT> || detail::shape_one_i32_control_v<EndControlT>) ) \
    [[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(            \
      ct::tile<KeyT, ct::shape<KeyN>> keys,                                                                          \
      ct::tile<ValueT, ct::shape<ValueN>> values,                                                                    \
      _Compare<CompareT> compare,                                                                                    \
      BeginControlT begin_bit,                                                                                       \
      EndControlT end_bit,                                                                                           \
      radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept           \
    {                                                                                                                \
      return ::cuda::coop::tile::radix_sort_pairs(                                                                   \
        keys,                                                                                                        \
        values,                                                                                                      \
        compare,                                                                                                     \
        detail::normalize_i32_control(begin_bit),                                                                    \
        detail::normalize_i32_control(end_bit),                                                                      \
        policy);                                                                                                     \
    }

_CCCL_COOP_TILE_RADIX_SORT_PAIRS_COMPARE_OVERLOADS(::cuda::std::less, 0)
_CCCL_COOP_TILE_RADIX_SORT_PAIRS_COMPARE_OVERLOADS(::cuda::std::greater, 1)

#  undef _CCCL_COOP_TILE_RADIX_SORT_PAIRS_COMPARE_OVERLOADS

template <typename KeyT,
          typename ValueT,
          typename BeginControlT,
          typename EndControlT,
          size_t KeyN,
          size_t ValueN,
          size_t RadixBits                  = 4,
          bool MemoizeOuterScan             = true,
          scan_algorithm InnerScanAlgorithm = scan_algorithm::warp_scans,
          radix_smem_config SmemConfig      = radix_smem_config::four_byte>
  requires(detail::compatible_i32_control_v<BeginControlT> && detail::compatible_i32_control_v<EndControlT>
           && (detail::shape_one_i32_control_v<BeginControlT> || detail::shape_one_i32_control_v<EndControlT>) )
[[nodiscard]] _CCCL_COOP_TILE_API inline radix_sort_pair_result<KeyT, ValueT, KeyN> radix_sort_pairs(
  ct::tile<KeyT, ct::shape<KeyN>> keys,
  ct::tile<ValueT, ct::shape<ValueN>> values,
  BeginControlT begin_bit,
  EndControlT end_bit,
  radix_sort_policy<RadixBits, MemoizeOuterScan, InnerScanAlgorithm, SmemConfig> policy = {}) noexcept
{
  return ::cuda::coop::tile::radix_sort_pairs(
    keys, values, detail::normalize_i32_control(begin_bit), detail::normalize_i32_control(end_bit), policy);
}

namespace detail
{
template <size_t Descending, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
topk_keys(ct::tile<T, ct::shape<N>> keys, int k, topk_policy<>) noexcept
{
  return detail::topk_keys_call<Descending, 1, T, N>(keys, k, static_cast<int>(N), 0, static_cast<int>(sizeof(T) * 8));
}

template <size_t Descending, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
topk_keys(ct::tile<T, ct::shape<N>> keys, int k, int begin_bit, int end_bit, topk_policy<>) noexcept
{
  return detail::topk_keys_call<Descending, 1, T, N>(keys, k, static_cast<int>(N), begin_bit, end_bit);
}

template <size_t Descending, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
topk_keys(ct::tile<T, ct::shape<N>> keys, int k, int valid_items, topk_policy<>) noexcept
{
  return detail::topk_keys_call<Descending, 0, T, N>(keys, k, valid_items, 0, static_cast<int>(sizeof(T) * 8));
}

template <size_t Descending, typename T, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>>
topk_keys(ct::tile<T, ct::shape<N>> keys, int k, int valid_items, int begin_bit, int end_bit, topk_policy<>) noexcept
{
  return detail::topk_keys_call<Descending, 0, T, N>(keys, k, valid_items, begin_bit, end_bit);
}

template <size_t Descending, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N>
topk_pairs(ct::tile<KeyT, ct::shape<N>> keys, ct::tile<ValueT, ct::shape<N>> values, int k, topk_policy<>) noexcept
{
  return detail::topk_pairs_call<Descending, 1, KeyT, ValueT, N>(
    keys, values, k, static_cast<int>(N), 0, static_cast<int>(sizeof(KeyT) * 8));
}

template <size_t Descending, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> topk_pairs(
  ct::tile<KeyT, ct::shape<N>> keys,
  ct::tile<ValueT, ct::shape<N>> values,
  int k,
  int begin_bit,
  int end_bit,
  topk_policy<>) noexcept
{
  return detail::topk_pairs_call<Descending, 1, KeyT, ValueT, N>(
    keys, values, k, static_cast<int>(N), begin_bit, end_bit);
}

template <size_t Descending, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> topk_pairs(
  ct::tile<KeyT, ct::shape<N>> keys,
  ct::tile<ValueT, ct::shape<N>> values,
  int k,
  int valid_items,
  topk_policy<>) noexcept
{
  return detail::topk_pairs_call<Descending, 0, KeyT, ValueT, N>(
    keys, values, k, valid_items, 0, static_cast<int>(sizeof(KeyT) * 8));
}

template <size_t Descending, typename KeyT, typename ValueT, size_t N>
[[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> topk_pairs(
  ct::tile<KeyT, ct::shape<N>> keys,
  ct::tile<ValueT, ct::shape<N>> values,
  int k,
  int valid_items,
  int begin_bit,
  int end_bit,
  topk_policy<>) noexcept
{
  return detail::topk_pairs_call<Descending, 0, KeyT, ValueT, N>(keys, values, k, valid_items, begin_bit, end_bit);
}
} // namespace detail

#  define _CCCL_COOP_TILE_TOPK_KEYS_OVERLOADS(_Name, _Descending)                                            \
    template <typename T, size_t N>                                                                          \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> _Name(                                \
      ct::tile<T, ct::shape<N>> keys, int k, topk_policy<> policy = {}) noexcept                             \
    {                                                                                                        \
      return detail::topk_keys<_Descending>(keys, k, policy);                                                \
    }                                                                                                        \
    template <typename T, size_t N>                                                                          \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> _Name(                                \
      ct::tile<T, ct::shape<N>> keys, int k, int begin_bit, int end_bit, topk_policy<> policy = {}) noexcept \
    {                                                                                                        \
      return detail::topk_keys<_Descending>(keys, k, begin_bit, end_bit, policy);                            \
    }                                                                                                        \
    template <typename T, size_t N>                                                                          \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> _Name(                                \
      ct::tile<T, ct::shape<N>> keys, int k, int valid_items, topk_policy<> policy = {}) noexcept            \
    {                                                                                                        \
      return detail::topk_keys<_Descending>(keys, k, valid_items, policy);                                   \
    }                                                                                                        \
    template <typename T, size_t N>                                                                          \
    [[nodiscard]] _CCCL_COOP_TILE_API inline ct::tile<T, ct::shape<N>> _Name(                                \
      ct::tile<T, ct::shape<N>> keys,                                                                        \
      int k,                                                                                                 \
      int valid_items,                                                                                       \
      int begin_bit,                                                                                         \
      int end_bit,                                                                                           \
      topk_policy<> policy = {}) noexcept                                                                    \
    {                                                                                                        \
      return detail::topk_keys<_Descending>(keys, k, valid_items, begin_bit, end_bit, policy);               \
    }

/**
 * @brief Select the largest `k` keys.
 *
 * The output retains input capacity; only its first `min(k, valid_items)` positions are defined and their order is
 * unspecified. After k, overloads accept valid_items, begin_bit and end_bit, both controls together,
 * or neither. Controls are int scalars, with 0 <= k, valid_items <= N and 0 <= begin_bit < end_bit
 * <= key_bits. Omitted controls select the full input and key width.
 *
 * `N` must be a positive multiple of 32.
 */
_CCCL_COOP_TILE_TOPK_KEYS_OVERLOADS(topk_max_keys, 1)
/**
 * @brief Select the smallest `k` keys.
 *
 * The output retains input capacity; only its first `min(k, valid_items)` positions are defined and their order is
 * unspecified. After k, overloads accept valid_items, begin_bit and end_bit, both controls together,
 * or neither. Controls are int scalars, with 0 <= k, valid_items <= N and 0 <= begin_bit < end_bit
 * <= key_bits. Omitted controls select the full input and key width.
 *
 * `N` must be a positive multiple of 32.
 */
_CCCL_COOP_TILE_TOPK_KEYS_OVERLOADS(topk_min_keys, 0)

#  undef _CCCL_COOP_TILE_TOPK_KEYS_OVERLOADS

#  define _CCCL_COOP_TILE_TOPK_PAIRS_OVERLOADS(_Name, _Descending)                                      \
    template <typename KeyT, typename ValueT, size_t N>                                                 \
    [[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> _Name(                   \
      ct::tile<KeyT, ct::shape<N>> keys,                                                                \
      ct::tile<ValueT, ct::shape<N>> values,                                                            \
      int k,                                                                                            \
      topk_policy<> policy = {}) noexcept                                                               \
    {                                                                                                   \
      return detail::topk_pairs<_Descending>(keys, values, k, policy);                                  \
    }                                                                                                   \
    template <typename KeyT, typename ValueT, size_t N>                                                 \
    [[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> _Name(                   \
      ct::tile<KeyT, ct::shape<N>> keys,                                                                \
      ct::tile<ValueT, ct::shape<N>> values,                                                            \
      int k,                                                                                            \
      int begin_bit,                                                                                    \
      int end_bit,                                                                                      \
      topk_policy<> policy = {}) noexcept                                                               \
    {                                                                                                   \
      return detail::topk_pairs<_Descending>(keys, values, k, begin_bit, end_bit, policy);              \
    }                                                                                                   \
    template <typename KeyT, typename ValueT, size_t N>                                                 \
    [[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> _Name(                   \
      ct::tile<KeyT, ct::shape<N>> keys,                                                                \
      ct::tile<ValueT, ct::shape<N>> values,                                                            \
      int k,                                                                                            \
      int valid_items,                                                                                  \
      topk_policy<> policy = {}) noexcept                                                               \
    {                                                                                                   \
      return detail::topk_pairs<_Descending>(keys, values, k, valid_items, policy);                     \
    }                                                                                                   \
    template <typename KeyT, typename ValueT, size_t N>                                                 \
    [[nodiscard]] _CCCL_COOP_TILE_API inline topk_pair_result<KeyT, ValueT, N> _Name(                   \
      ct::tile<KeyT, ct::shape<N>> keys,                                                                \
      ct::tile<ValueT, ct::shape<N>> values,                                                            \
      int k,                                                                                            \
      int valid_items,                                                                                  \
      int begin_bit,                                                                                    \
      int end_bit,                                                                                      \
      topk_policy<> policy = {}) noexcept                                                               \
    {                                                                                                   \
      return detail::topk_pairs<_Descending>(keys, values, k, valid_items, begin_bit, end_bit, policy); \
    }

/**
 * @brief Select the largest `k` keys with associated payloads.
 *
 * The output retains input capacity; only its first `min(k, valid_items)` positions are defined and their order is
 * unspecified. After k, overloads accept valid_items, begin_bit and end_bit, both controls together,
 * or neither. Controls are int scalars, with 0 <= k, valid_items <= N and 0 <= begin_bit < end_bit
 * <= key_bits. Omitted controls select the full input and key width. Pair results return keys
 * first; input shapes must match.
 *
 * `N` must be a positive multiple of 32.
 */
_CCCL_COOP_TILE_TOPK_PAIRS_OVERLOADS(topk_max_pairs, 1)
/**
 * @brief Select the smallest `k` keys with associated payloads.
 *
 * The output retains input capacity; only its first `min(k, valid_items)` positions are defined and their order is
 * unspecified. After k, overloads accept valid_items, begin_bit and end_bit, both controls together,
 * or neither. Controls are int scalars, with 0 <= k, valid_items <= N and 0 <= begin_bit < end_bit
 * <= key_bits. Omitted controls select the full input and key width. Pair results return keys
 * first; input shapes must match.
 *
 * `N` must be a positive multiple of 32.
 */
_CCCL_COOP_TILE_TOPK_PAIRS_OVERLOADS(topk_min_pairs, 0)

#  undef _CCCL_COOP_TILE_TOPK_PAIRS_OVERLOADS
} // namespace cuda::coop::tile

#  undef _CCCL_COOP_TILE_API

#endif // CUDA C++ Tile SIMT preview

#endif // CUDA_COOP_TILE_CUH
