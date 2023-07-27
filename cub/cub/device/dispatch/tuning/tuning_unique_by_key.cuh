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

#include <cub/agent/agent_unique_by_key.cuh>
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/config.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{

namespace unique_by_key
{

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
constexpr primitive_key is_primitive_key()
{
  return Traits<T>::PRIMITIVE ? primitive_key::yes : primitive_key::no;
}

template <class T>
constexpr primitive_val is_primitive_val()
{
  return Traits<T>::PRIMITIVE ? primitive_val::yes : primitive_val::no;
}

template <class KeyT>
constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1    ? key_size::_1
         : sizeof(KeyT) == 2  ? key_size::_2
         : sizeof(KeyT) == 4  ? key_size::_4
         : sizeof(KeyT) == 8  ? key_size::_8
         : sizeof(KeyT) == 16 ? key_size::_16
                              : key_size::unknown;
}

template <class ValueT>
constexpr val_size classify_val_size()
{
  return sizeof(ValueT) == 1    ? val_size::_1
         : sizeof(ValueT) == 2  ? val_size::_2
         : sizeof(ValueT) == 4  ? val_size::_4
         : sizeof(ValueT) == 8  ? val_size::_8
         : sizeof(ValueT) == 16 ? val_size::_16
                                : val_size::unknown;
}

template <class KeyT,
          class ValueT,
          primitive_key PrimitiveKey   = is_primitive_key<KeyT>(),
          primitive_val PrimitiveAccum = is_primitive_val<ValueT>(),
          key_size KeySize             = classify_key_size<KeyT>(),
          val_size AccumSize           = classify_val_size<ValueT>()>
struct sm90_tuning
{
  static constexpr int threads = 64;

  static constexpr int nominal_4b_items_per_thread = 11;

  static constexpr int items = Nominal4BItemsToItems<KeyT>(nominal_4b_items_per_thread);

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_LDG;

  using delay_constructor = detail::default_delay_constructor_t<int>;
};

// 8-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<550>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_2>
{
  static constexpr int threads = 448;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<725>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_4>
{
  static constexpr int threads = 256;

  static constexpr int items = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1130>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_1, val_size::_8>
{
  static constexpr int threads = 512;

  static constexpr int items = 10;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1100>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_1, val_size::_16>
{
  static constexpr int threads = 288;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::fixed_delay_constructor_t<344, 1165>;
};

// 16-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_1>
{
  static constexpr int threads = 256;

  static constexpr int items = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<640>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_2>
{
  static constexpr int threads = 288;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::fixed_delay_constructor_t<404, 710>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_4>
{
  static constexpr int threads = 512;

  static constexpr int items = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<525>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_2, val_size::_8>
{
  static constexpr int threads = 256;

  static constexpr int items = 23;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1200>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_2, val_size::_16>
{
  static constexpr int threads = 224;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::fixed_delay_constructor_t<424, 1055>;
};

// 32-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_1>
{
  static constexpr int threads = 448;

  static constexpr int items = 12;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::fixed_delay_constructor_t<348, 580>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_2>
{
  static constexpr int threads = 384;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_4>
{
  static constexpr int threads = 512;

  static constexpr int items = 14;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1045>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_4, val_size::_8>
{
  static constexpr int threads = 512;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1120>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_4, val_size::_16>
{
  static constexpr int threads = 384;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1025>;
};

// 64-bit key
template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_1>
{
  static constexpr int threads = 384;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1060>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_2>
{
  static constexpr int threads = 384;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::fixed_delay_constructor_t<964, 1125>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_4>
{
  static constexpr int threads = 640;

  static constexpr int items = 7;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1070>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::yes, key_size::_8, val_size::_8>
{
  static constexpr int threads = 448;

  static constexpr int items = 11;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1190>;
};

template <class KeyT, class ValueT>
struct sm90_tuning<KeyT, ValueT, primitive_key::yes, primitive_val::no, key_size::_8, val_size::_16>
{
  static constexpr int threads = 256;

  static constexpr int items = 9;

  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;

  static constexpr CacheLoadModifier load_modifier = LOAD_DEFAULT;

  using delay_constructor = detail::no_delay_constructor_t<1155>;
};

} // namespace unique_by_key
} // namespace detail

template <typename KeyInputIteratorT, typename ValueInputIteratorT = unsigned long long int *>
struct DeviceUniqueByKeyPolicy
{
  using KeyT   = typename std::iterator_traits<KeyInputIteratorT>::value_type;
  using ValueT = typename std::iterator_traits<ValueInputIteratorT>::value_type;

  // SM350
  struct Policy350 : ChainedPolicy<350, Policy350, Policy350>
  {
    const static int INPUT_SIZE = sizeof(KeyT);
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 9,
      ITEMS_PER_THREAD            = Nominal4BItemsToItems<KeyT>(NOMINAL_4B_ITEMS_PER_THREAD),
    };

    using UniqueByKeyPolicyT = AgentUniqueByKeyPolicy<128,
                                                      ITEMS_PER_THREAD,
                                                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                                      cub::LOAD_LDG,
                                                      cub::BLOCK_SCAN_WARP_SCANS,
                                                      detail::default_delay_constructor_t<int>>;
  };

  // SM520
  struct Policy520 : ChainedPolicy<520, Policy520, Policy350>
  {
    const static int INPUT_SIZE = sizeof(KeyT);
    enum
    {
      NOMINAL_4B_ITEMS_PER_THREAD = 11,
      ITEMS_PER_THREAD            = Nominal4BItemsToItems<KeyT>(NOMINAL_4B_ITEMS_PER_THREAD),
    };

    using UniqueByKeyPolicyT = AgentUniqueByKeyPolicy<64,
                                                      ITEMS_PER_THREAD,
                                                      cub::BLOCK_LOAD_WARP_TRANSPOSE,
                                                      cub::LOAD_LDG,
                                                      cub::BLOCK_SCAN_WARP_SCANS,
                                                      detail::default_delay_constructor_t<int>>;
  };

  /// SM90
  struct Policy900 : ChainedPolicy<900, Policy900, Policy520>
  {
    using tuning = detail::unique_by_key::sm90_tuning<KeyT, ValueT>;

    using UniqueByKeyPolicyT = AgentUniqueByKeyPolicy<tuning::threads,
                                                      tuning::items,
                                                      tuning::load_algorithm,
                                                      tuning::load_modifier,
                                                      BLOCK_SCAN_WARP_SCANS,
                                                      typename tuning::delay_constructor>;
  };

  using MaxPolicy = Policy900;
};

CUB_NAMESPACE_END
