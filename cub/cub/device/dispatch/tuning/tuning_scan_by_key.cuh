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

#include <cub/agent/agent_scan_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/device/dispatch/tuning/common.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/compute_capability.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__type_traits/is_trivially_copyable.h>

CUB_NAMESPACE_BEGIN

namespace detail::scan_by_key
{
struct scan_by_key_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
  delay_constructor_policy delay_constructor;

  _CCCL_API constexpr friend bool operator==(const scan_by_key_policy& lhs, const scan_by_key_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.store_algorithm == rhs.store_algorithm && lhs.scan_algorithm == rhs.scan_algorithm
        && lhs.delay_constructor == rhs.delay_constructor;
  }

  _CCCL_API constexpr friend bool operator!=(const scan_by_key_policy& lhs, const scan_by_key_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const scan_by_key_policy& p)
  {
    return os
        << "scan_by_key_policy { .block_threads = " << p.block_threads << ", .items_per_thread = " << p.items_per_thread
        << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .store_algorithm = " << p.store_algorithm << ", .scan_algorithm = " << p.scan_algorithm
        << ", .delay_constructor = " << p.delay_constructor << " }";
  }
#endif // _CCCL_HOSTED()
};
enum class primitive_accum
{
  no,
  yes
};
enum class primitive_op
{
  no,
  yes
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
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class AccumT>
constexpr primitive_accum is_primitive_accum()
{
  return is_primitive<AccumT>::value ? primitive_accum::yes : primitive_accum::no;
}

template <class ScanOpT>
constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ScanOpT>::value ? primitive_op::yes : primitive_op::no;
}

template <class ValueT>
constexpr val_size classify_val_size()
{
  return sizeof(ValueT) == 1 ? val_size::_1
       : sizeof(ValueT) == 2 ? val_size::_2
       : sizeof(ValueT) == 4 ? val_size::_4
       : sizeof(ValueT) == 8 ? val_size::_8
       : sizeof(ValueT) == 16
         ? val_size::_16
         : val_size::unknown;
}

template <class KeyT>
constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class KeyT,
          class ValueT,
          primitive_op PrimitiveOp,
          key_size KeySize                     = classify_key_size<KeyT>(),
          val_size ValueSize                   = classify_val_size<ValueT>(),
          primitive_accum PrimitiveAccumulator = is_primitive_accum<ValueT>()>
struct sm80_tuning;

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = no_delay_constructor_t<795>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<825>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<640>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<124, 1040>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_1, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 19;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1095>;
};

template <class KeyT>
struct sm80_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_1, val_size::_16, primitive_accum::no>
    : sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_1, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 8;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = no_delay_constructor_t<1070>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 14;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<625>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1055>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 160;
  static constexpr int items                           = 17;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<160, 695>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_2, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 160;
  static constexpr int items                           = 14;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1105>;
};

template <class KeyT>
struct sm80_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_2, val_size::_16, primitive_accum::no>
    : sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_2, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = no_delay_constructor_t<1130>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1130>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1140>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 9;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<888, 635>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_4, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 17;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1100>;
};

template <class KeyT>
struct sm80_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_4, val_size::_16, primitive_accum::no>
    : sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_4, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 11;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1115>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 13;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<24, 1060>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1160>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_8, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 320;
  static constexpr int items                           = 8;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<220>;
};

template <class KeyT>
struct sm80_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_8, val_size::_16, primitive_accum::no>
    : sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_8, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 7;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<144, 1120>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 7;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<364, 780>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 7;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1170>;
};

template <class KeyT, class ValueT>
struct sm80_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1030>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_16, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1160>;
};

template <class KeyT>
struct sm80_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_16, val_size::_16, primitive_accum::no>
    : sm80_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_16, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT,
          class ValueT,
          primitive_op PrimitiveOp,
          key_size KeySize                     = classify_key_size<KeyT>(),
          val_size ValueSize                   = classify_val_size<ValueT>(),
          primitive_accum PrimitiveAccumulator = is_primitive_accum<ValueT>()>
struct sm90_tuning;

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = no_delay_constructor_t<650>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 16;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<124, 995>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<488, 545>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<488, 1070>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_1, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 23;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<936, 1105>;
};

template <class KeyT>
struct sm90_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_1, val_size::_16, primitive_accum::no>
    : sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_1, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = fixed_delay_constructor_t<136, 785>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 20;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<445>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 22;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<312, 865>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<352, 1170>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_2, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 23;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<504, 1190>;
};

template <class KeyT>
struct sm90_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_2, val_size::_16, primitive_accum::no>
    : sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_2, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = no_delay_constructor_t<850>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 14;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<128, 965>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 288;
  static constexpr int items                           = 14;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<700, 1005>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 14;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<556, 1195>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_4, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 23;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<512, 1030>;
};

template <class KeyT>
struct sm90_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_4, val_size::_16, primitive_accum::no>
    : sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_4, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 12;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_DIRECT;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_DIRECT;
  using delay_constructor                              = fixed_delay_constructor_t<504, 1010>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<420, 970>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<500, 1125>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 11;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<600, 930>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_8, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 15;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<364, 1085>;
};

template <class KeyT>
struct sm90_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_8, val_size::_16, primitive_accum::no>
    : sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_8, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_1, primitive_accum::yes>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 7;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<500, 975>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_2, primitive_accum::yes>
{
  static constexpr int threads                         = 224;
  static constexpr int items                           = 10;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<164, 1075>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_4, primitive_accum::yes>
{
  static constexpr int threads                         = 256;
  static constexpr int items                           = 9;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<268, 1120>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_op::yes, key_size::_16, val_size::_8, primitive_accum::yes>
{
  static constexpr int threads                         = 192;
  static constexpr int items                           = 9;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<320, 1200>;
};

#if _CCCL_HAS_INT128()
template <class KeyT>
struct sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_16, val_size::_16, primitive_accum::no>
{
  static constexpr int threads                         = 128;
  static constexpr int items                           = 23;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<364, 1050>;
};

template <class KeyT>
struct sm90_tuning<KeyT, __uint128_t, primitive_op::yes, key_size::_16, val_size::_16, primitive_accum::no>
    : sm90_tuning<KeyT, __int128_t, primitive_op::yes, key_size::_16, val_size::_16, primitive_accum::no>
{};
#endif

template <class KeyT,
          class ValueT,
          primitive_op PrimitiveOp,
          key_size KeySize                     = classify_key_size<KeyT>(),
          val_size ValueSize                   = classify_val_size<ValueT>(),
          primitive_accum PrimitiveAccumulator = is_primitive_accum<ValueT>()>
struct sm100_tuning;

// key_size = 8 bits
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_1, primitive_accum::yes>
{
  // ipt_13.tpb_288.ns_420.dcid_0.l2w_745.trp_1.ld_0 1.030222  0.998162  1.027506  1.068348
  static constexpr int items                           = 13;
  static constexpr int threads                         = 288;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<745>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_2, primitive_accum::yes>
{
  // ipt_13.tpb_288.ns_388.dcid_1.l2w_570.trp_1.ld_0 1.228612   1.0  1.216841  1.416167
  static constexpr int items                           = 13;
  static constexpr int threads                         = 288;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<388, 570>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_4, primitive_accum::yes>
{
  // ipt_19.tpb_224.ns_1028.dcid_5.l2w_910.trp_1.ld_1 1.163440   1.0  1.146400  1.260684
  static constexpr int items                           = 19;
  static constexpr int threads                         = 224;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_jitter_window_constructor_t<1028, 910>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_1, val_size::_8, primitive_accum::yes>
{
  // ipt_18.tpb_192.ns_432.dcid_1.l2w_1035.trp_1.ld_1 1.177638  0.985417  1.157164  1.296477
  static constexpr int items                           = 18;
  static constexpr int threads                         = 192;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<432, 1035>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

// key_size = 16 bits
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_1, primitive_accum::yes>
{
  // ipt_12.tpb_384.ns_1900.dcid_0.l2w_840.trp_1.ld_0 1.010828  0.985782  1.007993  1.048859
  static constexpr int items                           = 12;
  static constexpr int threads                         = 384;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1900>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_2, primitive_accum::yes>
{
  // ipt_14.tpb_160.ns_1736.dcid_7.l2w_170.trp_1.ld_0 1.095207  1.065061  1.100302  1.142857
  static constexpr int items                           = 14;
  static constexpr int threads                         = 160;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_constructor_t<1736, 170>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_4, primitive_accum::yes>
{
  // ipt_14.tpb_160.ns_336.dcid_1.l2w_805.trp_1.ld_0 1.119313  1.095238  1.122013  1.148681
  static constexpr int items                           = 14;
  static constexpr int threads                         = 160;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = fixed_delay_constructor_t<336, 805>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_2, val_size::_8, primitive_accum::yes>
{
  static constexpr int items                           = 13;
  static constexpr int threads                         = 224;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backoff_constructor_t<348, 735>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

// key_size = 32 bits
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_1, primitive_accum::yes>
{
  // todo(gonidlelis): Significant regression. Search more workloads.
  // ipt_20.tpb_224.ns_1436.dcid_7.l2w_155.trp_1.ld_1 1.135878  0.866667  1.106600  1.339708
  static constexpr int items                           = 20;
  static constexpr int threads                         = 224;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_constructor_t<1436, 155>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_2, primitive_accum::yes>
{
  // ipt_13.tpb_288.ns_620.dcid_7.l2w_925.trp_1.ld_2 1.050929  1.000000  1.047178  1.115809
  static constexpr int items                           = 13;
  static constexpr int threads                         = 288;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_constructor_t<620, 925>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_4, primitive_accum::yes>
{
  // ipt_20.tpb_224.ns_1856.dcid_5.l2w_280.trp_1.ld_1 1.247248  1.000000  1.220196  1.446328
  static constexpr int items                           = 20;
  static constexpr int threads                         = 224;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_jitter_window_constructor_t<1856, 280>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_4, val_size::_8, primitive_accum::yes>
{
  // ipt_14.tpb_224.ns_464.dcid_2.l2w_680.trp_1.ld_1 1.070831  1.002088  1.064736  1.105437
  static constexpr int items                           = 14;
  static constexpr int threads                         = 224;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backoff_constructor_t<464, 860>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

// key_size = 64 bits
template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_1, primitive_accum::yes>
{
  // ipt_12.tpb_160.ns_532.dcid_0.l2w_850.trp_1.ld_0 1.041966  1.000000  1.037010  1.078399
  static constexpr int items                           = 12;
  static constexpr int threads                         = 160;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<532>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_2, primitive_accum::yes>
{
  // todo(gonidlelis): Significant regression. Search more workloads.
  // ipt_15.tpb_288.ns_988.dcid_7.l2w_335.trp_1.ld_0 1.064413  0.866667  1.045946  1.116803
  static constexpr int items                           = 15;
  static constexpr int threads                         = 288;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_constructor_t<988, 335>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_4, primitive_accum::yes>
{
  // ipt_22.tpb_160.ns_1032.dcid_5.l2w_505.trp_1.ld_2 1.184805  1.000000  1.164843  1.338536
  static constexpr int items                           = 22;
  static constexpr int threads                         = 160;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = exponential_backon_jitter_window_constructor_t<1032, 505>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_CA;
};

template <class KeyT, class ValueT>
struct sm100_tuning<KeyT, ValueT, primitive_op::yes, key_size::_8, val_size::_8, primitive_accum::yes>
{
  // ipt_23.tpb_256.ns_1232.dcid_0.l2w_810.trp_1.ld_0 1.067631  1.000000  1.059607  1.135646
  static constexpr int items                           = 23;
  static constexpr int threads                         = 256;
  static constexpr BlockLoadAlgorithm load_algorithm   = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr BlockStoreAlgorithm store_algorithm = BLOCK_STORE_WARP_TRANSPOSE;
  using delay_constructor                              = no_delay_constructor_t<1232>;
  static constexpr CacheLoadModifier load_modifier     = LOAD_DEFAULT;
};

template <typename KeysInputIteratorT, typename AccumT, typename ValueT, typename ScanOpT>
struct policy_hub
{
  using key_t                               = it_value_t<KeysInputIteratorT>;
  static constexpr int max_input_bytes      = static_cast<int>((::cuda::std::max) (sizeof(key_t), sizeof(AccumT)));
  static constexpr int combined_input_bytes = static_cast<int>(sizeof(key_t) + sizeof(AccumT));

  struct Policy500 : ChainedPolicy<500, Policy500, Policy500>
  {
    static constexpr int nominal_4b_items_per_thread = 6;
    static constexpr int items_per_thread =
      max_input_bytes <= 8 ? 6 : Nominal4BItemsToItemsCombined(nominal_4b_items_per_thread, combined_input_bytes);

    using ScanByKeyPolicyT =
      AgentScanByKeyPolicy<128,
                           items_per_thread,
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LOAD_CA,
                           BLOCK_SCAN_WARP_SCANS,
                           BLOCK_STORE_WARP_TRANSPOSE,
                           default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };

  template <CacheLoadModifier LoadModifier, typename DelayConstructurValueT>
  struct DefaultPolicy
  {
    static constexpr int nominal_4b_items_per_thread = 9;
    static constexpr int items_per_thread =
      max_input_bytes <= 8 ? 9 : Nominal4BItemsToItemsCombined(nominal_4b_items_per_thread, combined_input_bytes);

    using ScanByKeyPolicyT =
      AgentScanByKeyPolicy<256,
                           items_per_thread,
                           BLOCK_LOAD_WARP_TRANSPOSE,
                           LoadModifier,
                           BLOCK_SCAN_WARP_SCANS,
                           BLOCK_STORE_WARP_TRANSPOSE,
                           default_reduce_by_key_delay_constructor_t<DelayConstructurValueT, int>>;
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy520
  using DefaultPolicy520 = DefaultPolicy<LOAD_CA, AccumT>;

  struct Policy520
      : DefaultPolicy520
      , ChainedPolicy<520, Policy520, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick the default
  template <typename Tuning>
  static auto select_agent_policy(int)
    -> AgentScanByKeyPolicy<Tuning::threads,
                            Tuning::items,
                            Tuning::load_algorithm,
                            LOAD_DEFAULT,
                            BLOCK_SCAN_WARP_SCANS,
                            Tuning::store_algorithm,
                            typename Tuning::delay_constructor>;

  template <typename Tuning>
  // FIXME(bgruber): should we rather use `AccumT` instead of `ValueT` like the other default policies?
  static auto select_agent_policy(long) -> typename DefaultPolicy<LOAD_DEFAULT, ValueT>::ScanByKeyPolicyT;

  struct Policy800 : ChainedPolicy<800, Policy800, Policy520>
  {
    using ScanByKeyPolicyT = decltype(select_agent_policy<sm80_tuning<key_t, ValueT, is_primitive_op<ScanOpT>()>>(0));
  };

  // nvbug5935129: GCC-11.2 cannot directly use DefaultPolicy inside Policy860
  using DefaultPolicy860 = DefaultPolicy<LOAD_CA, AccumT>;

  struct Policy860
      : DefaultPolicy860
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ScanByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<key_t, ValueT, is_primitive_op<ScanOpT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick Policy900
    template <typename Tuning>
    static auto select_agent_policy100(int)
      -> AgentScanByKeyPolicy<Tuning::threads,
                              Tuning::items,
                              Tuning::load_algorithm,
                              Tuning::load_modifier,
                              BLOCK_SCAN_WARP_SCANS,
                              Tuning::store_algorithm,
                              typename Tuning::delay_constructor>;

    template <typename Tuning>
    // FIXME(bgruber): should we rather use `AccumT` instead of `ValueT` like the other default policies?
    static auto select_agent_policy100(long) -> typename Policy900::ScanByKeyPolicyT;

    using ScanByKeyPolicyT =
      decltype(select_agent_policy100<sm100_tuning<key_t, ValueT, is_primitive_op<ScanOpT>()>>(0));
  };

  using MaxPolicy = Policy1000;
};

// TODO(griwes): remove in CCCL 4.0 when we drop the scan dispatcher after publishing the tuning API
template <typename ActivePolicyT>
_CCCL_API constexpr auto convert_policy() -> scan_by_key_policy
{
  using policy_t = typename ActivePolicyT::ScanByKeyPolicyT;
  return {policy_t::BLOCK_THREADS,
          policy_t::ITEMS_PER_THREAD,
          policy_t::LOAD_ALGORITHM,
          policy_t::LOAD_MODIFIER,
          policy_t::STORE_ALGORITHM,
          policy_t::SCAN_ALGORITHM,
          delay_constructor_policy_from_type<typename policy_t::detail::delay_constructor_t>};
}

// TODO(griwes): remove in CCCL 4.0 when we drop the scan dispatcher after publishing the tuning API
template <typename PolicyHub>
struct policy_selector_from_hub
{
private:
  struct extract_policy_dispatch_t
  {
    scan_by_key_policy& policy;

    template <typename ActivePolicyT>
    _CCCL_API constexpr cudaError_t Invoke()
    {
      policy = convert_policy<ActivePolicyT>();
      return cudaSuccess;
    }
  };

public:
  _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> scan_by_key_policy
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      ({
                        scan_by_key_policy policy{};
                        extract_policy_dispatch_t dispatch{policy};
                        PolicyHub::MaxPolicy::Invoke(cc.get() * 10, dispatch);
                        return policy;
                      }),
                      ({ return convert_policy<typename PolicyHub::MaxPolicy::ActivePolicy>(); }));
  }
};

struct policy_selector
{
  int key_size;
  int value_size;
  int accum_size;
  bool value_is_primitive;
  bool value_is_trivially_copyable;
  type_t key_type;
  type_t value_type;
  type_t accum_type;
  op_kind_t operation_t;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> scan_by_key_policy
  {
    const bool value_is_primitive_or_trivially_copyable = value_is_primitive || value_is_trivially_copyable;
    const bool primitive_accum =
      accum_type != type_t::other && accum_type != type_t::int128 && accum_type != type_t::uint128;
    const bool primitive_value = value_is_primitive && value_type != type_t::int128 && value_type != type_t::uint128;
    const bool primitive_op    = operation_t != op_kind_t::other;
    const int max_input_bytes  = (::cuda::std::max) (key_size, accum_size);
    const int combined_input_bytes = key_size + accum_size;

    const auto default_items =
      max_input_bytes <= 8
        ? 9
        : Nominal4BItemsToItemsCombined(/* nominal_4b_items_per_thread */ 9, combined_input_bytes);

    const auto policy500_items =
      max_input_bytes <= 8
        ? 6
        : Nominal4BItemsToItemsCombined(/* nominal_4b_items_per_thread */ 6, combined_input_bytes);

    if (cc >= ::cuda::compute_capability{10, 0})
    {
      if (primitive_op && primitive_value)
      {
        switch (key_size)
        {
          case 1:
            switch (value_size)
            {
              case 1:
                // ipt_13.tpb_288.ns_420.dcid_0.l2w_745.trp_1.ld_0 1.030222  0.998162  1.027506  1.068348
                return {288,
                        13,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<no_delay_constructor_t<745>>};
              case 2:
                // ipt_13.tpb_288.ns_388.dcid_1.l2w_570.trp_1.ld_0 1.228612   1.0  1.216841  1.416167
                return {288,
                        13,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<fixed_delay_constructor_t<388, 570>>};
              case 4:
                // ipt_19.tpb_224.ns_1028.dcid_5.l2w_910.trp_1.ld_1 1.163440   1.0  1.146400  1.260684
                return {224,
                        19,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_jitter_window_constructor_t<1028, 910>>};
              case 8:
                // ipt_18.tpb_192.ns_432.dcid_1.l2w_1035.trp_1.ld_1 1.177638  0.985417  1.157164  1.296477
                return {192,
                        18,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<fixed_delay_constructor_t<432, 1035>>};
              default:
                break;
            }
            break;
          case 2:
            switch (value_size)
            {
              case 1:
                // ipt_12.tpb_384.ns_1900.dcid_0.l2w_840.trp_1.ld_0 1.010828  0.985782  1.007993  1.048859
                return {384,
                        12,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<no_delay_constructor_t<1900>>};
              case 2:
                // ipt_14.tpb_160.ns_1736.dcid_7.l2w_170.trp_1.ld_0 1.095207  1.065061  1.100302  1.142857
                return {160,
                        14,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_constructor_t<1736, 170>>};
              case 4:
                // ipt_14.tpb_160.ns_336.dcid_1.l2w_805.trp_1.ld_0 1.119313  1.095238  1.122013  1.148681
                return {160,
                        14,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<fixed_delay_constructor_t<336, 805>>};
              case 8:
                return {224,
                        13,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backoff_constructor_t<348, 735>>};
              default:
                break;
            }
            break;
          case 4:
            switch (value_size)
            {
              case 1:
                // todo(gonidlelis): Significant regression. Search more workloads.
                // ipt_20.tpb_224.ns_1436.dcid_7.l2w_155.trp_1.ld_1 1.135878  0.866667  1.106600  1.339708
                return {224,
                        20,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_constructor_t<1436, 155>>};
              case 2:
                // ipt_13.tpb_288.ns_620.dcid_7.l2w_925.trp_1.ld_2 1.050929  1.000000  1.047178  1.115809
                return {288,
                        13,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_constructor_t<620, 925>>};
              case 4:
                // ipt_20.tpb_224.ns_1856.dcid_5.l2w_280.trp_1.ld_1 1.247248  1.000000  1.220196  1.446328
                return {224,
                        20,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_jitter_window_constructor_t<1856, 280>>};
              case 8:
                // ipt_14.tpb_224.ns_464.dcid_2.l2w_680.trp_1.ld_1 1.070831  1.002088  1.064736  1.105437
                return {224,
                        14,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backoff_constructor_t<464, 860>>};
              default:
                break;
            }
            break;
          case 8:
            switch (value_size)
            {
              case 1:
                // ipt_12.tpb_160.ns_532.dcid_0.l2w_850.trp_1.ld_0 1.041966  1.000000  1.037010  1.078399
                return {160,
                        12,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<no_delay_constructor_t<532>>};
              case 2:
                // todo(gonidlelis): Significant regression. Search more workloads.
                // ipt_15.tpb_288.ns_988.dcid_7.l2w_335.trp_1.ld_0 1.064413  0.866667  1.045946  1.116803
                return {288,
                        15,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_constructor_t<988, 335>>};
              case 4:
                // ipt_22.tpb_160.ns_1032.dcid_5.l2w_505.trp_1.ld_2 1.184805  1.000000  1.164843  1.338536
                return {160,
                        22,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_CA,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<exponential_backon_jitter_window_constructor_t<1032, 505>>};
              case 8:
                // ipt_23.tpb_256.ns_1232.dcid_0.l2w_810.trp_1.ld_0 1.067631  1.000000  1.059607  1.135646
                return {256,
                        23,
                        BLOCK_LOAD_WARP_TRANSPOSE,
                        LOAD_DEFAULT,
                        BLOCK_STORE_WARP_TRANSPOSE,
                        BLOCK_SCAN_WARP_SCANS,
                        delay_constructor_policy_from_type<no_delay_constructor_t<1232>>};
              default:
                break;
            }
            break;
          default:
            break;
        }
      }
    }

    if (cc >= ::cuda::compute_capability{9, 0})
    {
      if (primitive_op)
      {
        if (primitive_value)
        {
          switch (key_size)
          {
            case 1:
              switch (value_size)
              {
                case 1:
                  return {128,
                          12,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<650>>};
                case 2:
                  return {256,
                          16,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<124, 995>>};
                case 4:
                  return {128,
                          15,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<488, 545>>};
                case 8:
                  return {224,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<488, 1070>>};
                default:
                  break;
              }
              break;
            case 2:
              switch (value_size)
              {
                case 1:
                  return {128,
                          12,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<136, 785>>};
                case 2:
                  return {128,
                          20,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<445>>};
                case 4:
                  return {128,
                          22,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<312, 865>>};
                case 8:
                  return {224,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<352, 1170>>};
                default:
                  break;
              }
              break;
            case 4:
              switch (value_size)
              {
                case 1:
                  return {128,
                          12,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<850>>};
                case 2:
                  return {256,
                          14,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<128, 965>>};
                case 4:
                  return {288,
                          14,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<700, 1005>>};
                case 8:
                  return {224,
                          14,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<556, 1195>>};
                default:
                  break;
              }
              break;
            case 8:
              switch (value_size)
              {
                case 1:
                  return {128,
                          12,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<504, 1010>>};
                case 2:
                  return {224,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<420, 970>>};
                case 4:
                  return {192,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<500, 1125>>};
                case 8:
                  return {224,
                          11,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<600, 930>>};
                default:
                  break;
              }
              break;
            default:
              break;
          }
        }
#if _CCCL_HAS_INT128()
        else if (accum_type == type_t::int128 || accum_type == type_t::uint128)
        {
          switch (key_size)
          {
            case 1:
              switch (value_size)
              {
                case 16:
                  return {128,
                          23,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<936, 1105>>};
                default:
                  break;
              }
              break;
            case 2:
              switch (value_size)
              {
                case 16:
                  return {128,
                          23,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<504, 1190>>};
                default:
                  break;
              }
              break;
            case 4:
              switch (value_size)
              {
                case 16:
                  return {128,
                          23,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<512, 1030>>};
                default:
                  break;
              }
              break;
            case 8:
              switch (value_size)
              {
                case 16:
                  return {192,
                          15,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<364, 1085>>};
                default:
                  break;
              }
              break;
            default:
              break;
          }
        }
#endif

        if (key_size == 16 && primitive_accum)
        {
          switch (value_size)
          {
            case 1:
              return {192,
                      7,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<500, 975>>};
            case 2:
              return {224,
                      10,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<164, 1075>>};
            case 4:
              return {256,
                      9,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<268, 1120>>};
            case 8:
              return {192,
                      9,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<320, 1200>>};
            default:
              break;
          }
        }
#if _CCCL_HAS_INT128()
        else if (key_size == 16 && (accum_type == type_t::int128 || accum_type == type_t::uint128))
        {
          switch (accum_size)
          {
            case 16:
              return {128,
                      23,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<364, 1050>>};
            default:
              break;
          }
        }
#endif
      }

      return {256,
              default_items,
              BLOCK_LOAD_WARP_TRANSPOSE,
              LOAD_DEFAULT,
              BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS,
              default_reduce_by_key_delay_constructor_policy(
                sizeof(int), value_size, true, value_is_primitive_or_trivially_copyable)};
    }

    if (cc >= ::cuda::compute_capability{8, 6}) // && cc < ::cuda::compute_capability{9, 0}
    {
      return {256,
              default_items,
              BLOCK_LOAD_WARP_TRANSPOSE,
              LOAD_CA,
              BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS,
              default_reduce_by_key_delay_constructor_policy(sizeof(int), accum_size, true, primitive_accum)};
    }

    if (cc >= ::cuda::compute_capability{8, 0})
    {
      if (primitive_op)
      {
        if (primitive_value)
        {
          switch (key_size)
          {
            case 1:
              switch (value_size)
              {
                case 1:
                  return {128,
                          12,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<795>>};
                case 2:
                  return {288,
                          12,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<825>>};
                case 4:
                  return {256,
                          15,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<640>>};
                case 8:
                  return {192,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<124, 1040>>};
                default:
                  break;
              }
              break;
            case 2:
              switch (value_size)
              {
                case 1:
                  return {256,
                          8,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1070>>};
                case 2:
                  return {320,
                          14,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<625>>};
                case 4:
                  return {256,
                          15,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1055>>};
                case 8:
                  return {160,
                          17,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<160, 695>>};
                default:
                  break;
              }
              break;
            case 4:
              switch (value_size)
              {
                case 1:
                  return {128,
                          12,
                          BLOCK_LOAD_DIRECT,
                          LOAD_DEFAULT,
                          BLOCK_STORE_DIRECT,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1130>>};
                case 2:
                  return {256,
                          12,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1130>>};
                case 4:
                  return {256,
                          15,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1140>>};
                case 8:
                  return {256,
                          9,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<888, 635>>};
                default:
                  break;
              }
              break;
            case 8:
              switch (value_size)
              {
                case 1:
                  return {128,
                          11,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1120>>};
                case 2:
                  return {256,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1115>>};
                case 4:
                  return {224,
                          13,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<fixed_delay_constructor_t<24, 1060>>};
                case 8:
                  return {224,
                          10,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1160>>};
                default:
                  break;
              }
              break;
            default:
              break;
          }
        }
#if _CCCL_HAS_INT128()
        else if (accum_type == type_t::int128 || accum_type == type_t::uint128)
        {
          switch (key_size)
          {
            case 1:
              switch (value_size)
              {
                case 16:
                  return {128,
                          19,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1095>>};
                default:
                  break;
              }
              break;
            case 2:
              switch (value_size)
              {
                case 16:
                  return {160,
                          14,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1105>>};
                default:
                  break;
              }
              break;
            case 4:
              switch (value_size)
              {
                case 16:
                  return {128,
                          17,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<1100>>};
                default:
                  break;
              }
              break;
            case 8:
              switch (value_size)
              {
                case 16:
                  return {320,
                          8,
                          BLOCK_LOAD_WARP_TRANSPOSE,
                          LOAD_DEFAULT,
                          BLOCK_STORE_WARP_TRANSPOSE,
                          BLOCK_SCAN_WARP_SCANS,
                          delay_constructor_policy_from_type<no_delay_constructor_t<220>>};
                default:
                  break;
              }
              break;
            default:
              break;
          }
        }
#endif

        if (key_size == 16 && primitive_accum)
        {
          switch (value_size)
          {
            case 1:
              return {192,
                      7,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<144, 1120>>};
            case 2:
              return {192,
                      7,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<fixed_delay_constructor_t<364, 780>>};
            case 4:
              return {256,
                      7,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<no_delay_constructor_t<1170>>};
            case 8:
              return {128,
                      15,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<no_delay_constructor_t<1030>>};
            default:
              break;
          }
        }
#if _CCCL_HAS_INT128()
        else if (key_size == 16 && (accum_type == type_t::int128 || accum_type == type_t::uint128))
        {
          switch (accum_size)
          {
            case 16:
              return {128,
                      15,
                      BLOCK_LOAD_WARP_TRANSPOSE,
                      LOAD_DEFAULT,
                      BLOCK_STORE_WARP_TRANSPOSE,
                      BLOCK_SCAN_WARP_SCANS,
                      delay_constructor_policy_from_type<no_delay_constructor_t<1160>>};
            default:
              break;
          }
        }
#endif
      }

      return {256,
              default_items,
              BLOCK_LOAD_WARP_TRANSPOSE,
              LOAD_DEFAULT,
              BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS,
              default_reduce_by_key_delay_constructor_policy(
                sizeof(int), value_size, true, value_is_primitive_or_trivially_copyable)};
    }

    if (cc >= ::cuda::compute_capability{6, 0})
    {
      return {256,
              default_items,
              BLOCK_LOAD_WARP_TRANSPOSE,
              LOAD_CA,
              BLOCK_STORE_WARP_TRANSPOSE,
              BLOCK_SCAN_WARP_SCANS,
              default_reduce_by_key_delay_constructor_policy(sizeof(int), accum_size, true, primitive_accum)};
    }

    return {128,
            policy500_items,
            BLOCK_LOAD_WARP_TRANSPOSE,
            LOAD_CA,
            BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS,
            default_reduce_by_key_delay_constructor_policy(sizeof(int), accum_size, true, primitive_accum)};
  }
};

template <typename KeyT, typename AccumT, typename ValueT, typename ScanOpT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::compute_capability cc) const -> scan_by_key_policy
  {
    return policy_selector{
      static_cast<int>(sizeof(KeyT)),
      static_cast<int>(sizeof(ValueT)),
      static_cast<int>(sizeof(AccumT)),
      is_primitive<ValueT>::value,
      ::cuda::std::is_trivially_copyable_v<ValueT>,
      classify_type<KeyT>,
      classify_type<ValueT>,
      classify_type<AccumT>,
      classify_op<ScanOpT>}(cc);
  }
};
} // namespace detail::scan_by_key

CUB_NAMESPACE_END
