// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_reduce_by_key.cuh>
#include <cub/agent/agent_rle.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/util_device.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/arch_traits.h>
#include <cuda/__device/compute_capability.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/concepts>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN

//! The algorithm used by the RLE-encode policy.
enum class RleAlgorithm
{
  lookback,
  lookahead
};

#if _CCCL_HOSTED()
namespace detail
{
[[nodiscard]] constexpr const char* to_string(RleAlgorithm algo) noexcept
{
  switch (algo)
  {
    case RleAlgorithm::lookback:
      return "RleAlgorithm::lookback";
    case RleAlgorithm::lookahead:
      return "RleAlgorithm::lookahead";
    default:
      return "<unknown RleAlgorithm>";
  }
}
} // namespace detail
#endif // _CCCL_HOSTED()

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, RleAlgorithm algo)
{
  return os << detail::to_string(algo);
}
#endif // _CCCL_HOSTED()

//! The lookback tuning policy for DeviceRunLengthEncode::Encode
struct RleLookbackPolicy
{
  int threads_per_block; //!< Number of threads in a CUDA block
  int items_per_thread; //!< Number of items processed per thread
  BlockLoadAlgorithm load_algorithm; //!< The @ref BlockLoadAlgorithm used for loading items from global memory
  CacheLoadModifier load_modifier; //!< The @ref CacheLoadModifier used for loading items from global memory
  BlockScanAlgorithm scan_algorithm; //!< The @ref BlockScanAlgorithm used for the prefix scan
  LookbackDelayPolicy lookback_delay; //!< The @ref LookbackDelayPolicy used for the lookback delay

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RleLookbackPolicy& lhs, const RleLookbackPolicy& rhs) noexcept
  {
    return lhs.threads_per_block == rhs.threads_per_block && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.scan_algorithm == rhs.scan_algorithm && lhs.lookback_delay == rhs.lookback_delay;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RleLookbackPolicy& lhs, const RleLookbackPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RleLookbackPolicy& p)
  {
    return os
        << "RleLookbackPolicy { .threads_per_block = " << p.threads_per_block << ", .items_per_thread = "
        << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .scan_algorithm = " << p.scan_algorithm << ", .lookback_delay = " << p.lookback_delay << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for the lookahead implementation of DeviceRunLengthEncode::Encode
struct RleLookaheadPolicy
{
  int items_per_thread; //!< Number of items each lane of a compute warp processes; a warp tile is
                        //!< 32 * items_per_thread items
  int compute_warps; //!< Number of compute warps; each processes one warp tile per pipeline generation
  int store_warps; //!< Number of store warps; must equal compute_warps
  int key_ring_stages; //!< Depth of the key staging ring: how many pipeline generations can be in flight
  // positions ring depth: positions are written at staging and consumed by store about 2 pipeline_gens later,
  // so it can be SHALLOWER than the keys ring and this buys room for more key_ring_stages
  int pos_ring_stages; //!< Depth of the run-positions ring; 2 * pos_ring_stages >= key_ring_stages must hold
  int poll_loads_per_lane; //!< Number of tile-state loads each poll-warp lane keeps in flight
  // when should compute warps stage?
  int flag_staging_threshold; //!< Runs per warp tile below which the compute warp stages raw head flags and the
                              //!< store warp decodes positions itself, instead of staging precomputed positions

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int warp_tile_size() const noexcept
  {
    return 32 * items_per_thread;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int tile_size() const noexcept
  {
    return compute_warps * warp_tile_size();
  }

  // store buffers one key + one length per reg-buf round in registers
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int buf_per_lane() const noexcept
  {
    return ((flag_staging_threshold - 1 + 31) / 32 > 0) ? (flag_staging_threshold - 1 + 31) / 32 : 1;
  }

  // for each input tile, we need to store the keys and in-tile positions
  // for in tile position we can just do unsigned int16 since tile size is never bigger than 2^16
  // each key slot carries slot_pad extra leading elements
  // we overcopy one 16B chunk to the left, so that we get the last tiles boundary element
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int slot_pad(int key_size) const noexcept
  {
    return 16 / key_size; // elements; 16 bytes = cp_async_bulk quantum
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr int slot_stride(int key_size, int key_align) const noexcept
  {
    return tile_size() + slot_pad(key_size) + (key_align < 16 ? 16 / key_size : 0);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::size_t
  dyn_smem_bytes(int key_size, int key_align) const noexcept
  {
    return static_cast<::cuda::std::size_t>(key_ring_stages) * slot_stride(key_size, key_align) * key_size
         + static_cast<::cuda::std::size_t>(pos_ring_stages) * tile_size() * sizeof(short);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const RleLookaheadPolicy& lhs, const RleLookaheadPolicy& rhs) noexcept
  {
    return lhs.items_per_thread == rhs.items_per_thread && lhs.compute_warps == rhs.compute_warps
        && lhs.store_warps == rhs.store_warps && lhs.key_ring_stages == rhs.key_ring_stages
        && lhs.pos_ring_stages == rhs.pos_ring_stages && lhs.poll_loads_per_lane == rhs.poll_loads_per_lane
        && lhs.flag_staging_threshold == rhs.flag_staging_threshold;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const RleLookaheadPolicy& lhs, const RleLookaheadPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RleLookaheadPolicy& p)
  {
    return os
        << "RleLookaheadPolicy { .items_per_thread = " << p.items_per_thread << ", .compute_warps = " << p.compute_warps
        << ", .store_warps = " << p.store_warps << ", .key_ring_stages = " << p.key_ring_stages
        << ", .pos_ring_stages = " << p.pos_ring_stages << ", .poll_loads_per_lane = " << p.poll_loads_per_lane
        << ", .flag_staging_threshold = " << p.flag_staging_threshold << " }";
  }
#endif // _CCCL_HOSTED()
};

//! The tuning policy for all algorithms in @ref DeviceRunLengthEncode
struct RleEncodePolicy
{
  RleAlgorithm algorithm; //!< The RLE-encode algorithm to use
  RleLookbackPolicy lookback; //!< The lookback policy; must be valid even when algorithm is @p lookahead, because it
                              //!< also drives the streaming fallback (device-side callers and non-viable types)
  RleLookaheadPolicy lookahead; //!< The lookahead policy (used when algorithm is @p lookahead, otherwise ignored)

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const RleEncodePolicy& lhs, const RleEncodePolicy& rhs) noexcept
  {
    return lhs.lookback == rhs.lookback && lhs.lookahead == rhs.lookahead && lhs.algorithm == rhs.algorithm;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const RleEncodePolicy& lhs, const RleEncodePolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const RleEncodePolicy& p)
  {
    return os << "RleEncodePolicy { .algorithm = " << p.algorithm << ", .lookback = " << p.lookback
              << ", .lookahead = " << p.lookahead << " }";
  }
#endif // _CCCL_HOSTED()
};

namespace detail::rle::encode
{
// TODO(bgruber): remove in CCCL 4.0 when we drop the CUB dispatchers
template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm80_tuning;

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<640>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<900>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1080>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1075>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<630>;
};

template <class LengthT>
struct sm80_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

// TODO(bgruber): remove in CCCL 4.0 when we drop the CUB dispatchers
template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm90_tuning;

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<620>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<775>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<284, 480>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 19;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<515>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<428, 930>;
};

template <class LengthT>
struct sm90_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

// TODO(bgruber): remove in CCCL 4.0 when we drop the CUB dispatchers
template <class LengthT,
          class KeyT,
          primitive_length PrimitiveLength = is_primitive_length<LengthT>(),
          primitive_key PrimitiveKey       = is_primitive_key<KeyT>(),
          length_size LengthSize           = classify_length_size<LengthT>(),
          key_size KeySize                 = classify_key_size<KeyT>()>
struct sm100_tuning;

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_1>
{
  // ipt_14.tpb_256.trp_0.ld_1.ns_468.dcid_7.l2w_300 1.202228  1.126160  1.197973  1.307692
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backon_constructor_t<468, 300>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  // ipt_14.tpb_224.trp_0.ld_0.ns_376.dcid_7.l2w_420 1.123754  1.002404  1.113839  1.274882
  static constexpr int threads                       = 224;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_constructor_t<376, 420>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  // ipt_14.tpb_256.trp_0.ld_1.ns_956.dcid_7.l2w_70 1.134395  1.071951  1.137008  1.169419
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backon_constructor_t<956, 70>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_188.dcid_2.l2w_765 1.100140  1.020069  1.116462  1.345506
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<188, 765>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class LengthT>
// struct sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {
//   static constexpr int threads = 128;
//   static constexpr int items = 11;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor = detail::fixed_delay_constructor_t<428, 930>;
// };

// template <class LengthT>
// struct sm100_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
//     : sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {};
#endif

// TODO(bgruber): remove in CCCL 4.0 when we drop all CUB dispatchers
// this policy is passed to DispatchStreamingReduceByKey and then to DeviceReduceByKeyKernel
template <class LengthT, class KeyT>
struct policy_hub
{
  static constexpr int max_input_bytes      = static_cast<int>((::cuda::std::max) (sizeof(KeyT), sizeof(LengthT)));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(LengthT);

  template <CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 6;
    static constexpr int items =
      (max_input_bytes <= 8)
        ? 6
        : ::cuda::std::clamp(
            ::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes), 1, nominal_4B_items_per_thread);
    using ReduceByKeyPolicyT =
      agent_reduce_by_key_policy<128,
                                 items,
                                 BLOCK_LOAD_DIRECT,
                                 LoadModifier,
                                 BLOCK_SCAN_WARP_SCANS,
                                 default_reduce_by_key_delay_constructor_t<LengthT, int>>;
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy500
  using DefaultPolicy500 = DefaultPolicy<LOAD_LDG>;

  struct Policy500
      : DefaultPolicy500
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> agent_reduce_by_key_policy<Tuning::threads,
                                  Tuning::items,
                                  Tuning::load_algorithm,
                                  LOAD_DEFAULT,
                                  BLOCK_SCAN_WARP_SCANS,
                                  typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) -> typename DefaultPolicy<LOAD_DEFAULT>::ReduceByKeyPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    using ReduceByKeyPolicyT = decltype(select_agent_policy<sm80_tuning<LengthT, KeyT>>(0));
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy860
  using DefaultPolicy860 = DefaultPolicy<LOAD_LDG>;

  struct Policy860
      : DefaultPolicy860
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ReduceByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<LengthT, KeyT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> agent_reduce_by_key_policy<Tuning::threads,
                                    Tuning::items,
                                    Tuning::load_algorithm,
                                    Tuning::load_modifier,
                                    BLOCK_SCAN_WARP_SCANS,
                                    typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy100(long) -> typename Policy900::ReduceByKeyPolicyT;

    using ReduceByKeyPolicyT = decltype(select_agent_policy100<sm100_tuning<LengthT, KeyT>>(0));
  };

  using MaxPolicy = Policy1000;
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept rle_encode_policy_selector = detail::policy_selector<T, RleEncodePolicy>;
#endif // _CCCL_HAS_CONCEPTS()

// TODO(bgruber): remove in CCCL 4.0 when we drop the RLE dispatchers
using reduce_by_key::policy_selector_from_hub;

struct policy_selector
{
  int length_size;
  int key_size;
  type_t key_t;

  // TODO(bgruber): we want to get rid of the following three and just assume by default that types behave "primitive".
  // This opts a lot more types into the tunings we have. We can do this when we publish the public tuning API, because
  // then users can opt-out of tunings again
  bool length_is_primitive;
  bool length_is_trivially_copyable;
  bool key_is_primitive;
  int key_align;
  bool key_is_trivially_copyable;
  bool input_contiguous;
  bool unique_out_contiguous;
  bool lengths_out_contiguous;
  bool num_runs_out_contiguous;
  bool input_matches_unique_type;
  bool offset_is_i32_or_i64;

  _CCCL_HOST_DEVICE_API constexpr auto __make_default_policy(CacheLoadModifier load_mod) const -> RleLookbackPolicy
  {
    constexpr int nominal_4B_items_per_thread = 6;
    const int combined_input_bytes            = length_size + key_size;
    const int max_input_bytes                 = (::cuda::std::max) (length_size, key_size);
    const int items_per_thread =
      (max_input_bytes <= 8)
        ? 6
        : ::cuda::std::clamp(
            ::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes), 1, nominal_4B_items_per_thread);
    return RleLookbackPolicy{
      128,
      items_per_thread,
      BLOCK_LOAD_DIRECT,
      load_mod,
      BLOCK_SCAN_WARP_SCANS,
      default_reduce_by_key_delay_constructor_policy(
        length_size, int{sizeof(int)}, length_is_primitive || length_is_trivially_copyable, true)};
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto get_lookback_policy(::cuda::compute_capability cc) const
    -> RleLookbackPolicy
  {
    // if we don't have a tuning for SM100, fall back to SM90
    if (cc >= ::cuda::compute_capability{10, 0} && length_is_primitive && length_size == 4 && key_is_primitive)
    {
      if (key_size == 1)
      {
        return RleLookbackPolicy{
          256,
          14,
          BLOCK_LOAD_DIRECT,
          LOAD_CA,
          BLOCK_SCAN_WARP_SCANS,
          {LookbackDelayAlgorithm::exponential_backon, 468, 300}};
      }
      if (key_size == 2)
      {
        return RleLookbackPolicy{
          224,
          14,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {LookbackDelayAlgorithm::exponential_backon, 376, 420}};
      }
      if (key_size == 4)
      {
        return RleLookbackPolicy{
          256,
          14,
          BLOCK_LOAD_DIRECT,
          LOAD_CA,
          BLOCK_SCAN_WARP_SCANS,
          {LookbackDelayAlgorithm::exponential_backon, 956, 70}};
      }
      if (key_size == 8)
      {
        return RleLookbackPolicy{
          224,
          9,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {LookbackDelayAlgorithm::exponential_backoff, 188, 765}};
      }
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (length_is_primitive && length_size == 4)
      {
        if (key_is_primitive && key_size == 1)
        {
          return RleLookbackPolicy{
            256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {LookbackDelayAlgorithm::no_delay, 0, 620}};
        }
        if (key_is_primitive && key_size == 2)
        {
          return RleLookbackPolicy{
            128, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {LookbackDelayAlgorithm::no_delay, 0, 775}};
        }
        if (key_is_primitive && key_size == 4)
        {
          return RleLookbackPolicy{
            192,
            14,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 284, 480}};
        }
        if (key_is_primitive && key_size == 8)
        {
          return RleLookbackPolicy{
            128,
            19,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::no_delay, 0, 515}};
        }
        if (key_t == type_t::int128 || key_t == type_t::uint128)
        {
          return RleLookbackPolicy{
            128,
            11,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 428, 930}};
        }
      }

      // no tuning, use a default one
      return __make_default_policy(LOAD_DEFAULT);
    }

    if (cc >= ::cuda::compute_capability{8, 6})
    {
      return __make_default_policy(LOAD_LDG);
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      if (length_is_primitive && length_size == 4)
      {
        if (key_is_primitive && key_size == 1)
        {
          return RleLookbackPolicy{
            256, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {LookbackDelayAlgorithm::no_delay, 0, 640}};
        }
        if (key_is_primitive && key_size == 2)
        {
          return RleLookbackPolicy{
            256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {LookbackDelayAlgorithm::no_delay, 0, 900}};
        }
        if (key_is_primitive && key_size == 4)
        {
          return RleLookbackPolicy{
            256,
            13,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::no_delay, 0, 1080}};
        }
        if (key_is_primitive && key_size == 8)
        {
          return RleLookbackPolicy{
            224,
            9,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::no_delay, 0, 1075}};
        }
        if (key_t == type_t::int128 || key_t == type_t::uint128)
        {
          return RleLookbackPolicy{
            128,
            7,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::no_delay, 0, 630}};
        }
      }

      // no tuning, use a default one
      return __make_default_policy(LOAD_DEFAULT);
    }

    // for SM50
    return __make_default_policy(LOAD_LDG);
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto get_lookahead_policy(::cuda::compute_capability cc) const
    -> ::cuda::std::optional<RleLookaheadPolicy>
  {
    // every knob below is a B200 measurement; consumer Blackwell (sm_120) has neither the smem opt-in headroom nor
    // measurements, so only the sm_10x family selects lookahead
    if (cc < ::cuda::compute_capability{10, 0} || cc >= ::cuda::compute_capability{11, 0})
    {
      return ::cuda::std::nullopt;
    }
    if (16 % key_size != 0)
    {
      return ::cuda::std::nullopt;
    }
    const int items_per_thread = (key_size >= 16) ? 8 : (key_size == 8 ? 16 : 32);
    return RleLookaheadPolicy{items_per_thread, 8, 8, 5, 3, 5, 32};
  }

  _CCCL_HOST_DEVICE_API constexpr bool can_use_lookahead(
    [[maybe_unused]] ::cuda::compute_capability cc, [[maybe_unused]] const RleLookaheadPolicy& lookahead_policy) const
  {
    // We need PTX ISA 8.6 and nvcc >= 12.8 for the clusterlaunchcontrol and bulk-copy instructions.
    // The macro `CCCL_DISABLE_WARPSPEED_RLE` will be left in as a kill-switch for users in case they find any bugs
    // after we shipped the implementation. TODO(nanan): remove CCCL_DISABLE_WARPSPEED_RLE in CCCL 4.0
#if __cccl_ptx_isa < 860 || _CCCL_CUDACC_BELOW(12, 8) || defined(CCCL_DISABLE_WARPSPEED_RLE)
    return false;
#else
    if (!input_contiguous || !unique_out_contiguous || !lengths_out_contiguous || !num_runs_out_contiguous
        || !key_is_trivially_copyable || !input_matches_unique_type || !offset_is_i32_or_i64)
    {
      return false;
    }
    if (16 % key_size != 0 || key_align != key_size)
    {
      return false;
    }
    if (lookahead_policy.dyn_smem_bytes(key_size, key_align)
        > ::cuda::arch_traits_for(cc).max_shared_memory_per_block_optin)
    {
      return false;
    }
    return true;
#endif // __cccl_ptx_isa < 860 || _CCCL_CUDACC_BELOW(12, 8) || defined(CCCL_DISABLE_WARPSPEED_RLE)
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> RleEncodePolicy
  {
    // we first try to get the valid lookahead implementation. if we can't run it, fall back to the lookback impl.
    // The lookback policy stays populated either way: the dispatch layer re-checks runtime-only facts (device smem
    // opt-in, tile count, temporary-storage alignment) and may still fall back at launch time.
    if (const auto lookahead_policy = get_lookahead_policy(cc))
    {
      if (can_use_lookahead(cc, *lookahead_policy))
      {
        return RleEncodePolicy{RleAlgorithm::lookahead, get_lookback_policy(cc), *lookahead_policy};
      }
    }
    return RleEncodePolicy{RleAlgorithm::lookback, get_lookback_policy(cc), RleLookaheadPolicy{}};
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(rle_encode_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class LengthT,
          class KeyT,
          class InputIteratorT,
          class UniqueOutputIteratorT,
          class LengthsOutputIteratorT,
          class NumRunsOutputIteratorT,
          class OffsetT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const -> RleEncodePolicy
  {
    constexpr policy_selector selector{
      int{sizeof(LengthT)},
      int{sizeof(KeyT)},
      classify_type<KeyT>,
      is_primitive_v<LengthT>,
      ::cuda::is_trivially_copyable_v<LengthT>,
      is_primitive_v<KeyT>,
      int{alignof(KeyT)},
      ::cuda::is_trivially_copyable_v<KeyT>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<InputIteratorT>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<UniqueOutputIteratorT>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<LengthsOutputIteratorT>,
      THRUST_NS_QUALIFIER::is_contiguous_iterator_v<NumRunsOutputIteratorT>,
      ::cuda::std::is_same_v<it_value_t<InputIteratorT>, KeyT>,
      ::cuda::std::is_signed_v<OffsetT> && (sizeof(OffsetT) == 4 || sizeof(OffsetT) == 8)};
    return selector(cc);
  }
};
} // namespace detail::rle::encode

CUB_NAMESPACE_END
