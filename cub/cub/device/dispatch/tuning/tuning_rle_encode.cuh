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
#include <cub/block/block_store.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce_by_key.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__device/arch_id.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>

#if _CCCL_HAS_CONCEPTS()
#  include <cuda/std/concepts>
#endif // _CCCL_HAS_CONCEPTS()

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::rle::encode
{
// TODO(bgruber): remove in CCCL 4.0 when we drop the reduce dispatchers
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

// TODO(bgruber): remove in CCCL 4.0 when we drop the reduce dispatchers
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
      AgentReduceByKeyPolicy<128,
                             items,
                             BLOCK_LOAD_DIRECT,
                             LoadModifier,
                             BLOCK_SCAN_WARP_SCANS,
                             default_reduce_by_key_delay_constructor_t<LengthT, int>>;
  };

  struct Policy500
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentReduceByKeyPolicy<Tuning::threads,
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

  struct Policy860
      : DefaultPolicy<LOAD_LDG>
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
      -> AgentReduceByKeyPolicy<Tuning::threads,
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

// DeviceRunLengthEncode::Encode delegates to reduce by key
using rle_encode_policy = reduce_by_key::reduce_by_key_policy;

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept rle_encode_policy_selector = reduce_by_key::reduce_by_key_policy_selector<T>;
#endif // _CCCL_HAS_CONCEPTS()

// TODO(bgruber): remove in CCCL 4.0 when we drop the RLE dispatchers
using reduce_by_key::policy_selector_from_hub;

struct policy_selector
{
  length_size length_sz;
  key_size key_sz;
  primitive_length prim_len;
  primitive_key prim_key;
  int max_input_bytes;
  int combined_input_bytes;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> rle_encode_policy
  {
    const bool tuned_prim = (prim_len == primitive_length::yes && prim_key == primitive_key::yes);
    const bool length_4   = (length_sz == length_size::_4);

    if (arch >= ::cuda::arch_id::sm_100 && tuned_prim && length_4)
    {
      if (key_sz == key_size::_1)
      {
        return rle_encode_policy{
          256,
          14,
          BLOCK_LOAD_DIRECT,
          LOAD_CA,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::exponential_backon, 468, 300}};
      }
      if (key_sz == key_size::_2)
      {
        return rle_encode_policy{
          224,
          14,
          BLOCK_LOAD_DIRECT,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::exponential_backon, 376, 420}};
      }
      if (key_sz == key_size::_4)
      {
        return rle_encode_policy{
          256,
          14,
          BLOCK_LOAD_DIRECT,
          LOAD_CA,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::exponential_backon, 956, 70}};
      }
      if (key_sz == key_size::_8)
      {
        return rle_encode_policy{
          224,
          9,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::exponential_backoff, 188, 765}};
      }
    }
    if (arch >= ::cuda::arch_id::sm_90 && tuned_prim && length_4)
    {
      if (key_sz == key_size::_1)
      {
        return rle_encode_policy{
          256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 620}};
      }
      if (key_sz == key_size::_2)
      {
        return rle_encode_policy{
          128, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 775}};
      }
      if (key_sz == key_size::_4)
      {
        return rle_encode_policy{
          192,
          14,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::fixed_delay, 284, 480}};
      }
      if (key_sz == key_size::_8)
      {
        return rle_encode_policy{
          128,
          19,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::no_delay, 0, 515}};
      }
#if _CCCL_HAS_INT128()
      if (key_sz == key_size::_16)
      {
        return rle_encode_policy{
          128,
          11,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::fixed_delay, 428, 930}};
      }
#endif
    }
    if (arch >= ::cuda::arch_id::sm_80 && tuned_prim && length_4)
    {
      if (key_sz == key_size::_1)
      {
        return rle_encode_policy{
          256, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 640}};
      }
      if (key_sz == key_size::_2)
      {
        return rle_encode_policy{
          256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 900}};
      }
      if (key_sz == key_size::_4)
      {
        return rle_encode_policy{
          256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1080}};
      }
      if (key_sz == key_size::_8)
      {
        return rle_encode_policy{
          224,
          9,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::no_delay, 0, 1075}};
      }
#if _CCCL_HAS_INT128()
      if (key_sz == key_size::_16)
      {
        return rle_encode_policy{
          128,
          7,
          BLOCK_LOAD_WARP_TRANSPOSE,
          LOAD_DEFAULT,
          BLOCK_SCAN_WARP_SCANS,
          {delay_constructor_kind::no_delay, 0, 630}};
      }
#endif
    }
    constexpr int nominal_4B_items_per_thread = 6;
    const int items =
      (max_input_bytes <= 8)
        ? 6
        : ::cuda::std::clamp(
            ::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes), 1, nominal_4B_items_per_thread);
    return rle_encode_policy{
      128,
      items,
      BLOCK_LOAD_DIRECT,
      LOAD_LDG,
      BLOCK_SCAN_WARP_SCANS,
      {delay_constructor_kind::fixed_delay, 350, 450},
    };
  }
};

template <class LengthT, class KeyT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> rle_encode_policy
  {
    constexpr policy_selector selector{
      classify_length_size<LengthT>(),
      classify_key_size<KeyT>(),
      is_primitive_length<LengthT>(),
      is_primitive_key<KeyT>(),
      static_cast<int>((::cuda::std::max) (sizeof(LengthT), sizeof(KeyT))),
      static_cast<int>(sizeof(LengthT) + sizeof(KeyT))};
    return selector(arch);
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(rle_encode_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()
} // namespace detail::rle::encode

CUB_NAMESPACE_END
