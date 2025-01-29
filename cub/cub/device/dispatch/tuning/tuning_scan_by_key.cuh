/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

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
#include <cub/thread/thread_operators.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace scan_by_key
{
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
enum class offset_size
{
  _4,
  _8,
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
  return Traits<AccumT>::PRIMITIVE ? primitive_accum::yes : primitive_accum::no;
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

template <typename KeysInputIteratorT, typename AccumT, typename ValueT, typename ScanOpT>
struct policy_hub
{
  using key_t                               = value_t<KeysInputIteratorT>;
  static constexpr int max_input_bytes      = static_cast<int>((::cuda::std::max)(sizeof(key_t), sizeof(AccumT)));
  static constexpr int combined_input_bytes = static_cast<int>(sizeof(key_t) + sizeof(AccumT));

  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
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

  struct Policy520
      : DefaultPolicy<LOAD_CA, AccumT>
      , ChainedPolicy<520, Policy520, Policy350>
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

  struct Policy860
      : DefaultPolicy<LOAD_CA, AccumT>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ScanByKeyPolicyT = decltype(select_agent_policy<sm90_tuning<key_t, ValueT, is_primitive_op<ScanOpT>()>>(0));
  };

  using MaxPolicy = Policy900;
};
} // namespace scan_by_key
} // namespace detail

template <typename KeysInputIteratorT, typename AccumT, typename ValueT = AccumT, typename ScanOpT = ::cuda::std::plus<>>
using DeviceScanByKeyPolicy CCCL_DEPRECATED_BECAUSE(
  "This class is considered an implementation detail and it will be "
  "removed.") = detail::scan_by_key::policy_hub<KeysInputIteratorT, AccumT, ValueT, ScanOpT>;

CUB_NAMESPACE_END
