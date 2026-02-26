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

#include <cuda/__device/arch_id.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/concepts>

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::rle::non_trivial_runs
{
// TODO(bgruber): remove in CCCL 4.0 when we drop all CUB dispatchers
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
  static constexpr int threads                       = 192;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<630>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<1015>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<915>;
};

template <class LengthT, class KeyT>
struct sm80_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<1065>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<1050>;
};

template <class LengthT>
struct sm80_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm80_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

// TODO(bgruber): remove in CCCL 4.0 when we drop all CUB dispatchers
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
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<385>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<675>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<695>;
};

template <class LengthT, class KeyT>
struct sm90_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::no_delay_constructor_t<840>;
};

#if _CCCL_HAS_INT128()
template <class LengthT>
struct sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  using delay_constructor                            = detail::fixed_delay_constructor_t<484, 1150>;
};

template <class LengthT>
struct sm90_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
    : sm90_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
{};
#endif

// TODO(bgruber): remove in CCCL 4.0 when we drop all CUB dispatchers
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
  // ipt_20.tpb_224.trp_1.ts_0.ld_1.ns_64.dcid_2.l2w_315 1.119878  1.003690  1.130067  1.338983
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<64, 315>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_2>
{
  // ipt_20.tpb_224.trp_1.ts_0.ld_0.ns_116.dcid_7.l2w_340 1.146528  1.072769  1.152390  1.333333
  static constexpr int threads                       = 224;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backon_constructor_t<116, 340>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_4>
{
  // ipt_13.tpb_224.trp_0.ts_0.ld_0.ns_252.dcid_2.l2w_470 1.113202  1.003690  1.133114  1.349296
  static constexpr int threads                       = 224;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<252, 470>;
};

template <class LengthT, class KeyT>
struct sm100_tuning<LengthT, KeyT, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{
  // ipt_15.tpb_256.trp_1.ts_0.ld_0.ns_28.dcid_2.l2w_520 1.114944  1.033189  1.122360  1.252083
  static constexpr int threads                       = 256;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr bool store_with_time_slicing      = false;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<28, 520>;
};
// Fall back to Policy900 for double, because that one performs better than the above tuning (same key_size)
// TODO(bgruber): in C++20 put a requires(!::cuda::std::is_same_v<KeyT, double>) onto the above tuning and delete this
// one
template <class LengthT>
struct sm100_tuning<LengthT, double, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
    : sm90_tuning<LengthT, double, primitive_length::yes, primitive_key::yes, length_size::_4, key_size::_8>
{};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class LengthT>
// struct sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {
//   static constexpr int threads = 288;
//   static constexpr int items = 9;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr bool store_with_time_slicing = false;
//   using delay_constructor = detail::fixed_delay_constructor_t<484, 1150>;
// };

// template <class LengthT>
// struct sm100_tuning<LengthT, __uint128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
//     : sm100_tuning<LengthT, __int128_t, primitive_length::yes, primitive_key::no, length_size::_4, key_size::_16>
// {};
#endif

// TODO(bgruber): remove in CCCL 4.0 when we drop all CUB dispatchers
template <class LengthT, class KeyT>
struct policy_hub
{
  template <BlockLoadAlgorithm BlockLoad, typename DelayConstructorKey, CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 15;
    static constexpr int ITEMS_PER_THREAD =
      ::cuda::std::clamp(nominal_4B_items_per_thread * 4 / int{sizeof(KeyT)}, 1, nominal_4B_items_per_thread);
    using RleSweepPolicyT =
      AgentRlePolicy<96,
                     ITEMS_PER_THREAD,
                     BlockLoad,
                     LoadModifier,
                     true,
                     BLOCK_SCAN_WARP_SCANS,
                     default_reduce_by_key_delay_constructor_t<DelayConstructorKey, int>>;
  };

  struct Policy500
      : DefaultPolicy<BLOCK_LOAD_DIRECT, int, LOAD_LDG> // TODO(bgruber): I think we want `LengthT` instead of `int`
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentRlePolicy<Tuning::threads,
                      Tuning::items,
                      Tuning::load_algorithm,
                      LOAD_DEFAULT,
                      Tuning::store_with_time_slicing,
                      BLOCK_SCAN_WARP_SCANS,
                      typename Tuning::delay_constructor>;
  template <typename Tuning>
  static auto select_agent_policy(long) ->
    typename DefaultPolicy<BLOCK_LOAD_WARP_TRANSPOSE, LengthT, LOAD_DEFAULT>::RleSweepPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy500>
  {
    using RleSweepPolicyT = decltype(select_agent_policy<sm80_tuning<LengthT, KeyT>>(0));
  };

  struct Policy860
      : DefaultPolicy<BLOCK_LOAD_DIRECT, int, LOAD_LDG> // TODO(bgruber): I think we want `LengthT` instead of `int`
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using RleSweepPolicyT = decltype(select_agent_policy<sm90_tuning<LengthT, KeyT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> AgentRlePolicy<Tuning::threads,
                        Tuning::items,
                        Tuning::load_algorithm,
                        Tuning::load_modifier,
                        Tuning::store_with_time_slicing,
                        BLOCK_SCAN_WARP_SCANS,
                        typename Tuning::delay_constructor>;
    template <typename Tuning>
    static auto select_agent_policy100(long) -> typename Policy900::RleSweepPolicyT;

    using RleSweepPolicyT = decltype(select_agent_policy100<sm100_tuning<LengthT, KeyT>>(0));
  };

  using MaxPolicy = Policy1000;
};

struct rle_non_trivial_runs_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  bool store_with_time_slicing;
  BlockScanAlgorithm scan_algorithm;
  delay_constructor_policy delay_constructor = {delay_constructor_kind::fixed_delay, 350, 450};

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const rle_non_trivial_runs_policy& lhs, const rle_non_trivial_runs_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_with_time_slicing == rhs.store_with_time_slicing && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.delay_constructor == rhs.delay_constructor;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const rle_non_trivial_runs_policy& lhs, const rle_non_trivial_runs_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const rle_non_trivial_runs_policy& p)
  {
    return os
        << "rle_non_trivial_runs_policy { .block_threads = " << p.block_threads
        << ", .items_per_thread = " << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm
        << ", .load_modifier = " << p.load_modifier << ", .store_with_time_slicing = " << p.store_with_time_slicing
        << ", .scan_algorithm = " << p.scan_algorithm << ", .delay_constructor = " << p.delay_constructor << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept rle_non_trivial_runs_policy_selector = detail::policy_selector<T, rle_non_trivial_runs_policy>;
#endif // _CCCL_HAS_CONCEPTS()

struct policy_selector
{
  // TODO(bgruber): refactor the key information
  int length_size;
  int key_size;
  type_t key_type;
  bool length_is_primitive;
  bool key_is_primitive; // TODO(bgruber): can probably be derived from key_type

  _CCCL_API constexpr auto
  make_default_policy(BlockLoadAlgorithm block_load_alg, int delay_ctor_key_size, CacheLoadModifier load_mod) const
  {
    const int nominal_4B_items_per_thread = 15;
    const int items_per_thread =
      ::cuda::std::clamp(nominal_4B_items_per_thread * 4 / key_size, 1, nominal_4B_items_per_thread);
    return rle_non_trivial_runs_policy{
      96,
      items_per_thread,
      block_load_alg,
      load_mod,
      true,
      BLOCK_SCAN_WARP_SCANS,
      default_reduce_by_key_delay_constructor_policy(delay_ctor_key_size, sizeof(int), true)};
  }

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> rle_non_trivial_runs_policy
  {
    if (arch >= ::cuda::arch_id::sm_100)
    {
      if (length_is_primitive && key_is_primitive && length_size == 4)
      {
        if (key_size == 1)
        {
          // ipt_20.tpb_224.trp_1.ts_0.ld_1.ns_64.dcid_2.l2w_315 1.119878  1.003690  1.130067  1.338983
          return rle_non_trivial_runs_policy{
            224,
            20,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_CA,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::exponential_backoff, 64, 315}};
        }
        if (key_size == 2)
        {
          // ipt_20.tpb_224.trp_1.ts_0.ld_0.ns_116.dcid_7.l2w_340 1.146528  1.072769  1.152390  1.333333
          return rle_non_trivial_runs_policy{
            224,
            20,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::exponential_backon, 116, 340}};
        }
        if (key_size == 4)
        {
          // ipt_13.tpb_224.trp_0.ts_0.ld_0.ns_252.dcid_2.l2w_470 1.113202  1.003690  1.133114  1.349296
          return rle_non_trivial_runs_policy{
            224,
            13,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::exponential_backoff, 252, 470}};
        }
        if (key_size == 8 && key_type != type_t::float64) // fall back to SM90 for double
        {
          // ipt_15.tpb_256.trp_1.ts_0.ld_0.ns_28.dcid_2.l2w_520 1.114944  1.033189  1.122360  1.252083
          return rle_non_trivial_runs_policy{
            256,
            15,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::exponential_backoff, 28, 520}};
        }
      }

      // no tuning for SM100, fall-through to SM90
    }

    if (arch >= ::cuda::arch_id::sm_90)
    {
      if (length_is_primitive && length_size == 4)
      {
        if (key_is_primitive && key_size == 1)
        {
          return rle_non_trivial_runs_policy{
            256,
            18,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 385}};
        }
        if (key_is_primitive && key_size == 2)
        {
          return rle_non_trivial_runs_policy{
            224,
            20,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 675}};
        }
        if (key_is_primitive && key_size == 4)
        {
          return rle_non_trivial_runs_policy{
            256,
            18,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 695}};
        }
        if (key_is_primitive && key_size == 8)
        {
          return rle_non_trivial_runs_policy{
            224,
            14,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 840}};
        }
        if (key_type == type_t::int128 || key_type == type_t::uint128)
        {
          return rle_non_trivial_runs_policy{
            288,
            9,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::fixed_delay, 484, 1150}};
        }
      }

      // no tuning for SM90, use a default policy
      return make_default_policy(BLOCK_LOAD_WARP_TRANSPOSE, length_size, LOAD_DEFAULT);
    }

    if (arch >= ::cuda::arch_id::sm_86)
    {
      // TODO(bgruber): I think we want `LengthT` instead of `int`
      return make_default_policy(BLOCK_LOAD_DIRECT, sizeof(int), LOAD_LDG);
    }

    if (arch >= ::cuda::arch_id::sm_80)
    {
      if (length_is_primitive && length_size == 4)
      {
        if (key_is_primitive && key_size == 1)
        {
          return rle_non_trivial_runs_policy{
            192,
            20,
            BLOCK_LOAD_DIRECT,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 630}};
        }
        if (key_is_primitive && key_size == 2)
        {
          return rle_non_trivial_runs_policy{
            192,
            20,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 1015}};
        }
        if (key_is_primitive && key_size == 4)
        {
          return rle_non_trivial_runs_policy{
            224,
            15,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 915}};
        }
        if (key_is_primitive && key_size == 8)
        {
          return rle_non_trivial_runs_policy{
            256,
            13,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 1065}};
        }
        if (key_type == type_t::int128 || key_type == type_t::uint128)
        {
          return rle_non_trivial_runs_policy{
            192,
            13,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            false,
            BLOCK_SCAN_WARP_SCANS,
            {delay_constructor_kind::no_delay, 0, 1050}};
        }
      }
      // no tuning for SM80, use a default policy
      return make_default_policy(BLOCK_LOAD_WARP_TRANSPOSE, length_size, LOAD_DEFAULT);
    }

    // default is from SM50
    return make_default_policy(BLOCK_LOAD_DIRECT, int{sizeof(int)}, LOAD_LDG);
  }
};

#if _CCCL_HAS_CONCEPTS()
static_assert(rle_non_trivial_runs_policy_selector<policy_selector>);
#endif // _CCCL_HAS_CONCEPTS()

template <class LengthT, class KeyT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> rle_non_trivial_runs_policy
  {
    constexpr policy_selector selector{
      sizeof(LengthT), int{sizeof(KeyT)}, classify_type<KeyT>, is_primitive_v<LengthT>, is_primitive_v<KeyT>};
    return selector(arch);
  }
};
} // namespace detail::rle::non_trivial_runs

CUB_NAMESPACE_END
