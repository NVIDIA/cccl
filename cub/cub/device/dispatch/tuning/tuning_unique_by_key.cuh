// SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/optional>

CUB_NAMESPACE_BEGIN

namespace detail::unique_by_key
{
struct unique_by_key_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  delay_constructor_policy delay_constructor;

  _CCCL_API constexpr friend bool operator==(const unique_by_key_policy& lhs, const unique_by_key_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.scan_algorithm == rhs.scan_algorithm && lhs.delay_constructor == rhs.delay_constructor;
  }

  _CCCL_API constexpr friend bool operator!=(const unique_by_key_policy& lhs, const unique_by_key_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const unique_by_key_policy& p)
  {
    return os
        << "unique_by_key_policy { .block_threads = " << p.block_threads << ", .items_per_thread = "
        << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .scan_algorithm = " << p.scan_algorithm << ", .delay_constructor = " << p.delay_constructor << " }";
  }
#endif // _CCCL_HOSTED()
};

enum class primitive_key
{
  no,
  yes
};
enum class primitive_val
{
  no,
  yes
};
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class val_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class T>
_CCCL_HOST_DEVICE constexpr primitive_key is_primitive_key()
{
  return is_primitive<T>::value ? primitive_key::yes : primitive_key::no;
}

template <class T>
_CCCL_HOST_DEVICE constexpr primitive_val is_primitive_val()
{
  return is_primitive<T>::value ? primitive_val::yes : primitive_val::no;
}

template <class KeyT>
_CCCL_HOST_DEVICE constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class ValueT>
_CCCL_HOST_DEVICE constexpr val_size classify_val_size()
{
  return sizeof(ValueT) == 1 ? val_size::_1
       : sizeof(ValueT) == 2 ? val_size::_2
       : sizeof(ValueT) == 4 ? val_size::_4
       : sizeof(ValueT) == 8 ? val_size::_8
       : sizeof(ValueT) == 16
         ? val_size::_16
         : val_size::unknown;
}

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm80_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<835>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<765>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1155>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1065>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<248, 1200>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 20;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1020>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 22;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<328, 1080>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<535>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1185>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1115>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<320, 1115>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<24, 555>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<324, 1105>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<740, 1105>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<764, 1155>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<992, 1135>;
};

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm90_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<550>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<725>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1130>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1100>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<344, 1165>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<640>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<404, 710>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<525>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1200>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_2, val_size::_16>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<424, 1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<348, 580>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1045>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads                       = 512;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_4, val_size::_16>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1025>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = fixed_delay_constructor_t<964, 1125>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads                       = 640;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1070>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads                       = 448;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1190>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = no_delay_constructor_t<1155>;
};

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm100_tuning;

// 8-bit key
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  // ipt_12.tpb_512.trp_0.ld_0.ns_948.dcid_5.l2w_955 1.121279  1.000000  1.114566  1.43765
  static constexpr int threads                       = 512;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<948, 955>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_1228.dcid_7.l2w_320 1.151229  1.007229  1.151131  1.443520
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1228, 320>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_2016.dcid_7.l2w_620 1.165300  1.095238  1.164478  1.266667
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<2016, 620>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_1728.dcid_5.l2w_980 1.118716  0.997167  1.116537  1.400000
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1728, 980>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
// {
//   static constexpr int threads                       = 288;
//   static constexpr int items                         = 7;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = fixed_delay_constructor_t<344, 1165>;
// };
#endif

// 16-bit key
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_508.dcid_7.l2w_1020 1.171886  0.906530  1.157128  1.457933
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<508, 1020>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  // ipt_12.tpb_384.trp_0.ld_0.ns_928.dcid_7.l2w_605 1.166564  0.997579  1.154805  1.406709
  static constexpr int threads                       = 384;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<928, 605>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  // ipt_11.tpb_384.trp_0.ld_1.ns_1620.dcid_7.l2w_810 1.144483  1.011085  1.152798  1.393750
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
  using delay_constructor                            = exponential_backon_constructor_t<1620, 810>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_1984.dcid_5.l2w_935 1.605554  1.177083  1.564488  1.946224
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1984, 935>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_2, val_size::_16>
// {
//   static constexpr int threads                       = 224;
//   static constexpr int items                         = 9;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = fixed_delay_constructor_t<424, 1055>;
// };
#endif

// 32-bit key
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  // ipt_14.tpb_512.trp_0.ld_0.ns_1136.dcid_7.l2w_605 1.148057  0.848558  1.133064  1.451074
  static constexpr int threads                       = 512;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<1136, 605>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  // ipt_11.tpb_384.trp_0.ld_0.ns_656.dcid_7.l2w_825 1.216312  1.090485  1.211800  1.535714
  static constexpr int threads                       = 384;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_constructor_t<656, 825>;
};

// todo(gonidelis): tuning performs very well for medium input size, regresses for large input sizes.
// find better tuning.
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
    : sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  // // ipt_14.tpb_512.trp_0.ld_0.ns_408.dcid_7.l2w_960 1.136333  0.995833  1.144371  1.448687
  // static constexpr int threads                       = 512;
  // static constexpr int items                         = 14;
  // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  // static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  // using delay_constructor                            = exponential_backon_constructor_t<408, 960>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_1012.dcid_5.l2w_800 1.164713  1.014819  1.174307  1.526042
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1012, 800>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_4, val_size::_16>
// {
//   static constexpr int threads                       = 384;
//   static constexpr int items                         = 7;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = no_delay_constructor_t<1025>;
// };
#endif

// 64-bit key

// todo(gonidelis): tuning regresses for large input sizes. find better tuning.
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
    : sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  // // ipt_9.tpb_384.trp_0.ld_0.ns_1064.dcid_7.l2w_600 1.085831  0.972452  1.080521  1.397089
  // static constexpr int threads                       = 384;
  // static constexpr int items                         = 9;
  // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  // static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  // using delay_constructor                            = exponential_backon_constructor_t<1064, 600>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_864.dcid_5.l2w_1130 1.124095  0.985748  1.120262  1.391304
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<864, 1130>;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  // ipt_10.tpb_384.trp_0.ld_0.ns_772.dcid_5.l2w_665 1.152243  1.019816  1.166636  1.517526
  static constexpr int threads                       = 384;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<772, 665>;
};

// todo(gonidelis): tuning regresses for large input sizes. find better tuning.
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
    : sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  // // ipt_7.tpb_576.trp_0.ld_0.ns_1132.dcid_5.l2w_1115 1.120721  0.977642  1.131594  1.449407
  // static constexpr int threads                       = 576;
  // static constexpr int items                         = 7;
  // static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  // static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
  // using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1132, 1115>;
};

// TODO(gonidelis): Tune for I128.
#if _CCCL_HAS_INT128()
// template <class KeyT, class ValueT>
// struct sm100_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
// {
//   static constexpr int threads                       = 256;
//   static constexpr int items                         = 9;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
//   using delay_constructor                            = no_delay_constructor_t<1155>;
// };
#endif

template <class KeyT, class ValueT>
struct policy_hub
{
  template <int Nominal4bItemsPerThread, int Threads>
  struct DefaultPolicy
  {
    static constexpr int items_per_thread = Nominal4BItemsToItems<KeyT>(Nominal4bItemsPerThread);
    using UniqueByKeyPolicyT =
      AgentUniqueByKeyPolicy<Threads,
                             items_per_thread,
                             BLOCK_LOAD_WARP_TRANSPOSE,
                             LOAD_LDG,
                             BLOCK_SCAN_WARP_SCANS,
                             detail::default_delay_constructor_t<int>>;
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy500
  using DefaultPolicy500 = DefaultPolicy<9, 128>;

  struct Policy500
      : DefaultPolicy500
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static _CCCL_HOST_DEVICE auto select_agent_policy(int)
    -> AgentUniqueByKeyPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              BLOCK_SCAN_WARP_SCANS,
                              typename Tuning::delay_constructor>;
  template <typename Tuning>
  static _CCCL_HOST_DEVICE auto select_agent_policy(long) -> typename DefaultPolicy<11, 64>::UniqueByKeyPolicyT;

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy520
  using DefaultPolicy520 = DefaultPolicy<11, 64>;

  struct Policy520
      : DefaultPolicy520
      , ChainedPolicy<520, Policy520, Policy500>
  {};

  struct Policy800 : ChainedPolicy<800, Policy800, Policy520>
  {
    using UniqueByKeyPolicyT = decltype(select_agent_policy<sm80_tuning<KeyT, ValueT>>(0));
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy860
  using DefaultPolicy860 = DefaultPolicy<11, 64>;

  struct Policy860
      : DefaultPolicy860
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using UniqueByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<KeyT, ValueT>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy100(int)
      -> AgentUniqueByKeyPolicy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                BLOCK_SCAN_WARP_SCANS,
                                typename Tuning::delay_constructor>;
    template <typename Tuning>
    static _CCCL_HOST_DEVICE auto select_agent_policy100(long) -> typename Policy900::UniqueByKeyPolicyT;

    using UniqueByKeyPolicyT = decltype(select_agent_policy100<sm100_tuning<KeyT, ValueT>>(0));
  };

  using MaxPolicy = Policy1000;
};

struct policy_selector
{
  int key_size;
  int value_size;
  bool primitive_key;
  bool primitive_value;

private:
  [[nodiscard]] _CCCL_API constexpr auto default_items_per_thread() const -> int
  {
    return cub::detail::nominal_4B_items_to_items(11, key_size);
  }

  [[nodiscard]] _CCCL_API constexpr auto get_default_policy() const -> unique_by_key_policy
  {
    return {64,
            default_items_per_thread(),
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_LDG,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::fixed_delay, 350, 450}};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_sm100_tuning() const -> ::cuda::std::optional<unique_by_key_policy>
  {
    if (!primitive_key)
    {
      return {};
    }

    if (primitive_value)
    {
      switch (key_size)
      {
        case 1:
          switch (value_size)
          {
            case 1:
              // ipt_12.tpb_512.trp_0.ld_0.ns_948.dcid_5.l2w_955 1.121279  1.000000  1.114566  1.43765
              return unique_by_key_policy{
                512,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 948, 955}};
            case 2:
              // ipt_14.tpb_512.trp_0.ld_0.ns_1228.dcid_7.l2w_320 1.151229  1.007229  1.151131  1.443520
              return unique_by_key_policy{
                512,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 1228, 320}};
            case 4:
              // ipt_14.tpb_512.trp_0.ld_0.ns_2016.dcid_7.l2w_620 1.165300  1.095238  1.164478  1.266667
              return unique_by_key_policy{
                512,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 2016, 620}};
            case 8:
              // ipt_10.tpb_384.trp_0.ld_0.ns_1728.dcid_5.l2w_980 1.118716  0.997167  1.116537  1.400000
              return unique_by_key_policy{
                384,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 1728, 980}};
            default:
              return {};
          }
        case 2:
          switch (value_size)
          {
            case 1:
              // ipt_14.tpb_512.trp_0.ld_0.ns_508.dcid_7.l2w_1020 1.171886  0.906530  1.157128  1.457933
              return unique_by_key_policy{
                512,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 508, 1020}};
            case 2:
              // ipt_12.tpb_384.trp_0.ld_0.ns_928.dcid_7.l2w_605 1.166564  0.997579  1.154805  1.406709
              return unique_by_key_policy{
                384,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 928, 605}};
            case 4:
              // ipt_11.tpb_384.trp_0.ld_1.ns_1620.dcid_7.l2w_810 1.144483  1.011085  1.152798  1.393750
              return unique_by_key_policy{
                384,
                11,
                BLOCK_LOAD_DIRECT,
                LOAD_CA,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 1620, 810}};
            case 8:
              // ipt_10.tpb_384.trp_0.ld_0.ns_1984.dcid_5.l2w_935 1.605554  1.177083  1.564488  1.946224
              return unique_by_key_policy{
                384,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 1984, 935}};
            default:
              return {};
          }
        case 4:
          switch (value_size)
          {
            case 1:
              // ipt_14.tpb_512.trp_0.ld_0.ns_1136.dcid_7.l2w_605 1.148057  0.848558  1.133064  1.451074
              return unique_by_key_policy{
                512,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 1136, 605}};
            case 2:
              // ipt_11.tpb_384.trp_0.ld_0.ns_656.dcid_7.l2w_825 1.216312  1.090485  1.211800  1.535714
              return unique_by_key_policy{
                384,
                11,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon, 656, 825}};
            case 8:
              // ipt_10.tpb_384.trp_0.ld_0.ns_1012.dcid_5.l2w_800 1.164713  1.014819  1.174307  1.526042
              return unique_by_key_policy{
                384,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 1012, 800}};
            default:
              return {};
          }
        case 8:
          switch (value_size)
          {
            case 2:
              // ipt_10.tpb_384.trp_0.ld_0.ns_864.dcid_5.l2w_1130 1.124095  0.985748  1.120262  1.391304
              return unique_by_key_policy{
                384,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 864, 1130}};
            case 4:
              // ipt_10.tpb_384.trp_0.ld_0.ns_772.dcid_5.l2w_665 1.152243  1.019816  1.166636  1.517526
              return unique_by_key_policy{
                384,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, 772, 665}};
            default:
              return {};
          }
        default:
          return {};
      }
    }

    return {};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_sm90_tuning() const -> ::cuda::std::optional<unique_by_key_policy>
  {
    if (!primitive_key)
    {
      return {};
    }

    if (primitive_value)
    {
      switch (key_size)
      {
        case 1:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 550}};
            case 2:
              return unique_by_key_policy{
                448,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 725}};
            case 4:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1130}};
            case 8:
              return unique_by_key_policy{
                512,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1100}};
            default:
              return {};
          }
        case 2:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 640}};
            case 2:
              return unique_by_key_policy{
                288,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 404, 710}};
            case 4:
              return unique_by_key_policy{
                512,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 525}};
            case 8:
              return unique_by_key_policy{
                256,
                23,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1200}};
            default:
              return {};
          }
        case 4:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                448,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 348, 580}};
            case 2:
              return unique_by_key_policy{
                384,
                9,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1060}};
            case 4:
              return unique_by_key_policy{
                512,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1045}};
            case 8:
              return unique_by_key_policy{
                512,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1120}};
            default:
              return {};
          }
        case 8:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                384,
                9,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1060}};
            case 2:
              return unique_by_key_policy{
                384,
                9,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 964, 1125}};
            case 4:
              return unique_by_key_policy{
                640,
                7,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1070}};
            case 8:
              return unique_by_key_policy{
                448,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1190}};
            default:
              return {};
          }
        default:
          return {};
      }
    }

    if (value_size == 16)
    {
      switch (key_size)
      {
        case 1:
          return unique_by_key_policy{
            288,
            7,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::fixed_delay, 344, 1165}};
        case 2:
          return unique_by_key_policy{
            224,
            9,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::fixed_delay, 424, 1055}};
        case 4:
          return unique_by_key_policy{
            384,
            7,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1025}};
        case 8:
          return unique_by_key_policy{
            256,
            9,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1155}};
        default:
          return {};
      }
    }

    return {};
  }

  [[nodiscard]] _CCCL_API constexpr auto get_sm80_tuning() const -> ::cuda::std::optional<unique_by_key_policy>
  {
    if (!primitive_key)
    {
      return {};
    }

    if (primitive_value)
    {
      switch (key_size)
      {
        case 1:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 835}};
            case 2:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 765}};
            case 4:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1155}};
            case 8:
              return unique_by_key_policy{
                224,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1065}};
            default:
              return {};
          }
        case 2:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                320,
                20,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1020}};
            case 2:
              return unique_by_key_policy{
                192,
                22,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 328, 1080}};
            case 4:
              return unique_by_key_policy{
                256,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 535}};
            case 8:
              return unique_by_key_policy{
                256,
                10,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1055}};
            default:
              return {};
          }
        case 4:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                256,
                12,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1120}};
            case 2:
              return unique_by_key_policy{
                256,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1185}};
            case 4:
              return unique_by_key_policy{
                256,
                11,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::no_delay, 0, 1115}};
            case 8:
              return unique_by_key_policy{
                256,
                7,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 320, 1115}};
            default:
              return {};
          }
        case 8:
          switch (value_size)
          {
            case 1:
              return unique_by_key_policy{
                256,
                7,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 24, 555}};
            case 2:
              return unique_by_key_policy{
                256,
                7,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 324, 1105}};
            case 4:
              return unique_by_key_policy{
                256,
                7,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 740, 1105}};
            case 8:
              return unique_by_key_policy{
                192,
                7,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                delay_constructor_policy{delay_constructor_kind::fixed_delay, 764, 1155}};
            default:
              return {};
          }
        default:
          return {};
      }
    }

    if (value_size == 16)
    {
      switch (key_size)
      {
        case 1:
          return unique_by_key_policy{
            128,
            15,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::fixed_delay, 248, 1200}};
        case 8:
          return unique_by_key_policy{
            128,
            7,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_DEFAULT,
            BLOCK_SCAN_WARP_SCANS,
            delay_constructor_policy{delay_constructor_kind::fixed_delay, 992, 1135}};
        default:
          return {};
      }
    }

    return {};
  }

public:
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> unique_by_key_policy
  {
    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (auto tuning = get_sm100_tuning())
      {
        return *tuning;
      }
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (auto tuning = get_sm90_tuning())
      {
        return *tuning;
      }
      return get_default_policy();
    }

    if (cc >= ::cuda::compute_capability{8, 6})
    {
      return get_default_policy();
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      if (auto tuning = get_sm80_tuning())
      {
        return *tuning;
      }
    }

    return get_default_policy();
  }
};

template <typename KeyT, typename ValueT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> unique_by_key_policy
  {
    return policy_selector{
      static_cast<int>(sizeof(KeyT)),
      static_cast<int>(sizeof(ValueT)),
      is_primitive<KeyT>::value && sizeof(KeyT) <= 8,
      is_primitive<ValueT>::value && sizeof(ValueT) <= 8}(cc);
  }
};

template <typename ActivePolicyT>
_CCCL_API constexpr auto convert_policy() -> unique_by_key_policy
{
  using policy_t = typename ActivePolicyT::UniqueByKeyPolicyT;
  return {policy_t::BLOCK_THREADS,
          policy_t::ITEMS_PER_THREAD,
          policy_t::LOAD_ALGORITHM,
          policy_t::LOAD_MODIFIER,
          policy_t::SCAN_ALGORITHM,
          delay_constructor_policy_from_type<typename policy_t::detail::delay_constructor_t>};
}

template <typename PolicyHub>
struct policy_selector_from_hub
{
  [[nodiscard]] _CCCL_DEVICE_API constexpr auto operator()(::cuda::compute_capability) const -> unique_by_key_policy
  {
    using UniqueByKeyPolicyT = typename PolicyHub::MaxPolicy::UniqueByKeyPolicyT;
    return unique_by_key_policy{
      UniqueByKeyPolicyT::BLOCK_THREADS,
      UniqueByKeyPolicyT::ITEMS_PER_THREAD,
      UniqueByKeyPolicyT::LOAD_ALGORITHM,
      UniqueByKeyPolicyT::LOAD_MODIFIER,
      UniqueByKeyPolicyT::SCAN_ALGORITHM,
      delay_constructor_policy_from_type<typename UniqueByKeyPolicyT::detail::delay_constructor_t>};
  }
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept unique_by_key_policy_selector = cub::detail::policy_selector<T, unique_by_key_policy>;
#endif
} // namespace detail::unique_by_key

CUB_NAMESPACE_END
