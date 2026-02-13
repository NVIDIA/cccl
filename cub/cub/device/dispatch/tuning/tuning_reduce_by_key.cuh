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
#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/detail/delay_constructor.cuh>
#include <cub/util_device.cuh>
#include <cub/util_type.cuh>

#include <cuda/__device/arch_id.h>
#include <cuda/cmath>
#include <cuda/std/__algorithm/clamp.h>

#if _CCCL_HAS_CONCEPTS()
#  include <cuda/std/concepts>
#endif // _CCCL_HAS_CONCEPTS()

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail::reduce_by_key
{
enum class primitive_key
{
  no,
  yes
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
enum class key_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};
enum class accum_size
{
  _1,
  _2,
  _4,
  _8,
  _16,
  unknown
};

template <class T>
_CCCL_API constexpr primitive_key is_primitive_key()
{
  return detail::is_primitive<T>::value ? primitive_key::yes : primitive_key::no;
}

template <class T>
_CCCL_API constexpr primitive_accum is_primitive_accum()
{
  return detail::is_primitive<T>::value ? primitive_accum::yes : primitive_accum::no;
}

template <class ReductionOpT>
_CCCL_API constexpr primitive_op is_primitive_op()
{
  return basic_binary_op_t<ReductionOpT>::value ? primitive_op::yes : primitive_op::no;
}

template <class KeyT>
_CCCL_API constexpr key_size classify_key_size()
{
  return sizeof(KeyT) == 1 ? key_size::_1
       : sizeof(KeyT) == 2 ? key_size::_2
       : sizeof(KeyT) == 4 ? key_size::_4
       : sizeof(KeyT) == 8 ? key_size::_8
       : sizeof(KeyT) == 16
         ? key_size::_16
         : key_size::unknown;
}

template <class AccumT>
_CCCL_API constexpr accum_size classify_accum_size()
{
  return sizeof(AccumT) == 1 ? accum_size::_1
       : sizeof(AccumT) == 2 ? accum_size::_2
       : sizeof(AccumT) == 4 ? accum_size::_4
       : sizeof(AccumT) == 8 ? accum_size::_8
       : sizeof(AccumT) == 16
         ? accum_size::_16
         : accum_size::unknown;
}

_CCCL_API constexpr int size_of(key_size sz)
{
  return sz == key_size::_1 ? 1
       : sz == key_size::_2 ? 2
       : sz == key_size::_4 ? 4
       : sz == key_size::_8 ? 8
       : sz == key_size::_16
         ? 16
         : 4;
}

_CCCL_API constexpr int size_of(accum_size sz)
{
  return sz == accum_size::_1 ? 1
       : sz == accum_size::_2 ? 2
       : sz == accum_size::_4 ? 4
       : sz == accum_size::_8 ? 8
       : sz == accum_size::_16
         ? 16
         : 4;
}

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm80_tuning;

// 8-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<975>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 12;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<840>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<760>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1070>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1175>;
};

// 16-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<620>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<640>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<905>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<810>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2, accum_size::_16>
{
  static constexpr int threads                       = 160;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1115>;
};

// 32-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  static constexpr int threads                       = 288;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1110>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1200>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 15;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1110>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1165>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4, accum_size::_16>
{
  static constexpr int threads                       = 160;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1100>;
};

// 64-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1175>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  static constexpr int threads                       = 224;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1075>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  static constexpr int threads                       = 384;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1040>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1080>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<430>;
};

// 128-bit key
template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_1>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1105>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_2>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<755>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<535>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_8>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 7;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1035>;
};

template <class KeyT, class AccumT>
struct sm80_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::no, key_size::_16, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1090>;
};

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm90_tuning;

// 8-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<720>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  static constexpr int threads                       = 320;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<865>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  static constexpr int threads                       = 192;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<735>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<580>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1100>;
};

// 16-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<985>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<276, 650>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 14;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<240, 765>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 19;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1190>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1175>;
};

// 32-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<404, 645>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1160>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 18;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1170>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1055>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1195>;
};

// 64-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 10;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::no_delay_constructor_t<1170>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  static constexpr int threads                       = 256;
  static constexpr int items                         = 9;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = detail::fixed_delay_constructor_t<236, 1030>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 13;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<152, 560>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 23;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1030>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1125>;
};

// 128-bit key
template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_1>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1080>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_2>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<320, 1005>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_4>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::fixed_delay_constructor_t<232, 1100>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::yes, key_size::_16, accum_size::_8>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1195>;
};

template <class KeyT, class AccumT>
struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::no, primitive_accum::no, key_size::_16, accum_size::_16>
{
  static constexpr int threads                       = 128;
  static constexpr int items                         = 11;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::no_delay_constructor_t<1150>;
};

template <class KeyT,
          class AccumT,
          primitive_op PrimitiveOp,
          primitive_key PrimitiveKey     = is_primitive_key<KeyT>(),
          primitive_accum PrimitiveAccum = is_primitive_accum<AccumT>(),
          key_size KeySize               = classify_key_size<KeyT>(),
          accum_size AccumSize           = classify_accum_size<AccumT>()>
struct sm100_tuning;

// 8-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_1>
{
  // ipt_13.tpb_576.trp_0.ld_1.ns_2044.dcid_5.l2w_240 1.161888  0.848558  1.134941  1.299109
  static constexpr int items                         = 13;
  static constexpr int threads                       = 576;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<2044, 240>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_2>
{
  // ipt_10.tpb_224.trp_0.ld_0.ns_244.dcid_4.l2w_390 1.313932  1.260540  1.319588  1.427374
  static constexpr int items                         = 10;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_jitter_window_constructor_t<224, 390>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_4>
{
  // ipt_14.tpb_128.trp_0.ld_0.ns_248.dcid_2.l2w_285  1.118109  1.051534  1.134336  1.326788
  static constexpr int items                         = 14;
  static constexpr int threads                       = 128;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_constructor_t<248, 285>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_1, accum_size::_8>
{
  // ipt_19.tpb_128.trp_1.ld_0.ns_132.dcid_1.l2w_540 1.113820  1.002404  1.105014  1.202296
  static constexpr int items                         = 19;
  static constexpr int threads                       = 128;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<132, 540>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_1,
// accum_size::_16>
// {
// static constexpr int threads                       = 128;
// static constexpr int items                         = 11;
// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
// using delay_constructor                            = detail::no_delay_constructor_t<1100>;
// };

// 16-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_1>
{
  // ipt_14.tpb_128.trp_1.ld_0.ns_164.dcid_2.l2w_290 1.239579  1.119705  1.239111  1.313112
  static constexpr int items                         = 14;
  static constexpr int threads                       = 128;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = detail::exponential_backoff_constructor_t<164, 290>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_2>
{
  // ipt_14.tpb_256.trp_1.ld_0.ns_180.dcid_2.l2w_975 1.145635  1.012658  1.139956  1.251546
  static constexpr int items                         = 14;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<180, 975>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{
  // ipt_11.tpb_256.trp_0.ld_0.ns_224.dcid_2.l2w_550 1.066293  1.000109  1.073092  1.181818
  static constexpr int items                         = 11;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_constructor_t<224, 550>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_8>
{
  // ipt_10.tpb_160.trp_1.ld_0.ns_156.dcid_1.l2w_725 1.045007  1.002105  1.049690  1.141827
  static constexpr int items                         = 10;
  static constexpr int threads                       = 160;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = fixed_delay_constructor_t<156, 725>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// I16, F32, I32 regresses, default it back.
template <class KeyT>
struct sm100_tuning<KeyT, float, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_2, accum_size::_4>
{};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_2,
// accum_size::_16>
// {
// static constexpr int threads                       = 128;
// static constexpr int items                         = 11;
// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
// using delay_constructor                            = detail::no_delay_constructor_t<1100>;
// };

// 32-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_1>
{
  // ipt_10.tpb_224.trp_0.ld_0.ns_324.dcid_2.l2w_285 1.157217  1.073724  1.166510  1.356940
  static constexpr int items                         = 10;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backoff_constructor_t<324, 285>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_2>
{
  // ipt_11.tpb_256.trp_0.ld_0.ns_1984.dcid_5.l2w_115 1.214155  1.128842  1.214093  1.364476
  static constexpr int items                         = 11;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_DIRECT;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1984, 115>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_4>
{
  // ipt_14.tpb_224.trp_1.ld_0.ns_476.dcid_5.l2w_1005 1.187378  1.119705  1.185397  1.258420

  static constexpr int items                         = 14;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<476, 1005>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_4, accum_size::_8>
{
  // ipt_10.tpb_256.trp_1.ld_0.ns_1868.dcid_7.l2w_145 1.142915  1.020581  1.137459  1.237913
  static constexpr int items                         = 10;
  static constexpr int threads                       = 256;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_constructor_t<1868, 145>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_4,
// accum_size::_16>
// {
// static constexpr int threads                       = 128;
// static constexpr int items                         = 11;
// static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
// using delay_constructor                            = detail::no_delay_constructor_t<1100>;
// };

// 64-bit key
template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_1>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_1940.dcid_5.l2w_460 1.157294  1.075650  1.153566  1.250729
  static constexpr int items                         = 9;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backon_jitter_window_constructor_t<1940, 460>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_2>
{
  // ipt_11.tpb_224.trp_1.ld_1.ns_392.dcid_2.l2w_550 1.104034  1.007212  1.099543  1.220401
  static constexpr int items                         = 11;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<392, 550>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_CA;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_4>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_244.dcid_2.l2w_475 1.130098  1.000000  1.130661  1.215722
  static constexpr int items                         = 9;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<244, 475>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

template <class KeyT, class AccumT>
struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::yes, key_size::_8, accum_size::_8>
{
  // ipt_9.tpb_224.trp_1.ld_0.ns_196.dcid_2.l2w_340 1.272056  1.142857  1.262499  1.352941
  static constexpr int items                         = 9;
  static constexpr int threads                       = 224;
  static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
  using delay_constructor                            = exponential_backoff_constructor_t<196, 340>;
  static constexpr CacheLoadModifier load_modifier   = LOAD_DEFAULT;
};

// todo(gonidelis): Add tunings for I128.
// template <class KeyT, class AccumT>
// struct sm100_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_8,
// accum_size::_16>
// {
//   static constexpr int threads                       = 128;
//   static constexpr int items                         = 11;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = detail::no_delay_constructor_t<1125>;
// };

// todo(gonidelis): Add tunings for 128-bit key.
// 128-bit key
// template <class KeyT, class AccumT>
// struct sm90_tuning<KeyT, AccumT, primitive_op::yes, primitive_key::yes, primitive_accum::no, key_size::_16,
// accum_size::_1>
// {
//   static constexpr int threads                       = 128;
//   static constexpr int items                         = 11;
//   static constexpr BlockLoadAlgorithm load_algorithm = BLOCK_LOAD_WARP_TRANSPOSE;
//   using delay_constructor                            = detail::no_delay_constructor_t<1125>;
// };

template <class ReductionOpT, class AccumT, class KeyT>
struct policy_hub
{
  static constexpr int max_input_bytes      = static_cast<int>((::cuda::std::max) (sizeof(KeyT), sizeof(AccumT)));
  static constexpr int combined_input_bytes = sizeof(KeyT) + sizeof(AccumT);

  template <CacheLoadModifier LoadModifier>
  struct DefaultPolicy
  {
    static constexpr int nominal_4B_items_per_thread = 6;
    static constexpr int items_per_thread =
      (max_input_bytes <= 8)
        ? 6
        : ::cuda::std::clamp(
            ::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes), 1, nominal_4B_items_per_thread);

    using ReduceByKeyPolicyT =
      AgentReduceByKeyPolicy<128,
                             items_per_thread,
                             BLOCK_LOAD_DIRECT,
                             LoadModifier,
                             BLOCK_SCAN_WARP_SCANS,
                             default_reduce_by_key_delay_constructor_t<AccumT, int>>;
  };

  struct Policy500
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<500, Policy500, Policy500>
  {};

  // Use values from tuning if a specialization exists, otherwise pick DefaultPolicy
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
    using ReduceByKeyPolicyT =
      decltype(select_agent_policy<sm80_tuning<KeyT, AccumT, is_primitive_op<ReductionOpT>()>>(0));
  };

  struct Policy860
      : DefaultPolicy<LOAD_LDG>
      , ChainedPolicy<860, Policy860, Policy800>
  {};

  struct Policy900 : ChainedPolicy<900, Policy900, Policy860>
  {
    using ReduceByKeyPolicyT =
      decltype(select_agent_policy<sm90_tuning<KeyT, AccumT, is_primitive_op<ReductionOpT>()>>(0));
  };

  struct Policy1000 : ChainedPolicy<1000, Policy1000, Policy900>
  {
    // Use values from tuning if a specialization exists, otherwise pick the default
    template <typename Tuning>
    static auto select_agent_policy(int)
      -> AgentReduceByKeyPolicy<Tuning::threads,
                                Tuning::items,
                                Tuning::load_algorithm,
                                Tuning::load_modifier,
                                BLOCK_SCAN_WARP_SCANS,
                                typename Tuning::delay_constructor>;

    template <typename Tuning>
    static auto select_agent_policy(long) -> typename Policy900::ReduceByKeyPolicyT;

    using ReduceByKeyPolicyT =
      decltype(select_agent_policy<sm100_tuning<KeyT, AccumT, is_primitive_op<ReductionOpT>()>>(0));
  };
  using MaxPolicy = Policy1000;
};

struct reduce_by_key_policy
{
  int block_threads;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  delay_constructor_policy delay_constructor;

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator==(const reduce_by_key_policy& lhs, const reduce_by_key_policy& rhs)
  {
    return lhs.block_threads == rhs.block_threads && lhs.items_per_thread == rhs.items_per_thread
        && lhs.load_algorithm == rhs.load_algorithm && lhs.load_modifier == rhs.load_modifier
        && lhs.scan_algorithm == rhs.scan_algorithm && lhs.delay_constructor == rhs.delay_constructor;
  }

  [[nodiscard]] _CCCL_API constexpr friend bool
  operator!=(const reduce_by_key_policy& lhs, const reduce_by_key_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const reduce_by_key_policy& p)
  {
    return os
        << "reduce_by_key_policy { .block_threads = " << p.block_threads << ", .items_per_thread = "
        << p.items_per_thread << ", .load_algorithm = " << p.load_algorithm << ", .load_modifier = " << p.load_modifier
        << ", .scan_algorithm = " << p.scan_algorithm << ", .delay_constructor = " << p.delay_constructor << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

_CCCL_HOST_DEVICE constexpr reduce_by_key_policy
make_default_reduce_by_key_policy(int combined_input_bytes, int max_input_bytes, CacheLoadModifier load_mod)
{
  constexpr int nominal_4B_items_per_thread = 6;
  const int items_per_thread =
    (max_input_bytes <= 8)
      ? 6
      : ::cuda::std::clamp(static_cast<int>(::cuda::ceil_div(nominal_4B_items_per_thread * 8, combined_input_bytes)),
                           1,
                           nominal_4B_items_per_thread);
  return reduce_by_key_policy{
    128,
    items_per_thread,
    BLOCK_LOAD_DIRECT,
    load_mod,
    BLOCK_SCAN_WARP_SCANS,
    {delay_constructor_kind::fixed_delay, 350, 450}};
}

struct policy_selector
{
  int key_size;
  int accum_size;
  bool is_primitive_key_t;
  bool is_primitive_accum_t;
  bool is_primitive_op;

  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> reduce_by_key_policy
  {
    const int combined_input_bytes = key_size + accum_size;
    const int max_input_bytes      = (::cuda::std::max) (key_size, accum_size);
    const auto default_ldg         = [&] {
      return make_default_reduce_by_key_policy(combined_input_bytes, max_input_bytes, LOAD_LDG);
    };
    const auto default_load_default = [&] {
      return make_default_reduce_by_key_policy(combined_input_bytes, max_input_bytes, LOAD_DEFAULT);
    };

    const bool tuned_prim = (is_primitive_key_t && is_primitive_accum_t);

    if (!is_primitive_op)
    {
      return default_ldg();
    }

    if (arch >= ::cuda::arch_id::sm_100 && tuned_prim)
    {
      if (key_size == 1 && accum_size == 1)
      {
        return {576,
                13,
                BLOCK_LOAD_DIRECT,
                LOAD_CA,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backon_jitter_window, 2044, 240}};
      }
      if (key_size == 1 && accum_size == 2)
      {
        return {224,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff_jitter_window, 224, 390}};
      }
      if (key_size == 1 && accum_size == 4)
      {
        return {128,
                14,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 248, 285}};
      }
      if (key_size == 1 && accum_size == 8)
      {
        return {128,
                19,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 132, 540}};
      }
      if (key_size == 2 && accum_size == 1)
      {
        return {128,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 164, 290}};
      }
      if (key_size == 2 && accum_size == 2)
      {
        return {256,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 180, 975}};
      }
      if (key_size == 2 && accum_size == 4)
      {
        return {256,
                11,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 224, 550}};
      }
      if (key_size == 2 && accum_size == 8)
      {
        return {160,
                10,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 156, 725}};
      }
      if (key_size == 4 && accum_size == 1)
      {
        return {224,
                10,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 324, 285}};
      }
      if (key_size == 4 && accum_size == 2)
      {
        return {256,
                11,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backon_jitter_window, 1984, 115}};
      }
      if (key_size == 4 && accum_size == 4)
      {
        return {224,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backon_jitter_window, 476, 1005}};
      }
      if (key_size == 4 && accum_size == 8)
      {
        return {256,
                10,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backon, 1868, 145}};
      }
      if (key_size == 8 && accum_size == 1)
      {
        return {224,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backon_jitter_window, 1940, 460}};
      }
      if (key_size == 8 && accum_size == 2)
      {
        return {224,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_CA,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 392, 550}};
      }
      if (key_size == 8 && accum_size == 4)
      {
        return {224,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 244, 475}};
      }
      if (key_size == 8 && accum_size == 8)
      {
        return {224,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::exponential_backoff, 196, 340}};
      }
    }

    if (arch >= ::cuda::arch_id::sm_90 && tuned_prim)
    {
      if (key_size == 1 && accum_size == 1)
      {
        return {
          256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 720}};
      }
      if (key_size == 1 && accum_size == 2)
      {
        return {
          320, 23, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 865}};
      }
      if (key_size == 1 && accum_size == 4)
      {
        return {192,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 735}};
      }
      if (key_size == 1 && accum_size == 8)
      {
        return {128,
                13,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 580}};
      }
      if (key_size == 1 && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1100}};
      }
      if (key_size == 2 && accum_size == 1)
      {
        return {
          128, 23, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 985}};
      }
      if (key_size == 2 && accum_size == 2)
      {
        return {256,
                11,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 276, 650}};
      }
      if (key_size == 2 && accum_size == 4)
      {
        return {256,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 240, 765}};
      }
      if (key_size == 2 && accum_size == 8)
      {
        return {128,
                19,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1190}};
      }
      if (key_size == 2 && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1175}};
      }
      if (key_size == 4 && accum_size == 1)
      {
        return {256,
                13,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 404, 645}};
      }
      if (key_size == 4 && accum_size == 2)
      {
        return {256,
                18,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1160}};
      }
      if (key_size == 4 && accum_size == 4)
      {
        return {256,
                18,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1170}};
      }
      if (key_size == 4 && accum_size == 8)
      {
        return {128,
                13,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1055}};
      }
      if (key_size == 4 && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1195}};
      }
      if (key_size == 8 && accum_size == 1)
      {
        return {
          256, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1170}};
      }
      if (key_size == 8 && accum_size == 2)
      {
        return {256,
                9,
                BLOCK_LOAD_DIRECT,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 236, 1030}};
      }
      if (key_size == 8 && accum_size == 4)
      {
        return {128,
                13,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 152, 560}};
      }
      if (key_size == 8 && accum_size == 8)
      {
        return {128,
                23,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1030}};
      }
      if (key_size == 8 && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1125}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 1)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1080}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 2)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 320, 1005}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 4)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::fixed_delay, 232, 1100}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 8)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1195}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1150}};
      }
      return default_load_default();
    }

    if (arch >= ::cuda::arch_id::sm_86)
    {
      return default_ldg();
    }

    if (arch >= ::cuda::arch_id::sm_80 && tuned_prim)
    {
      if (key_size == 1 && accum_size == 1)
      {
        return {
          256, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 975}};
      }
      if (key_size == 1 && accum_size == 2)
      {
        return {
          224, 12, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 840}};
      }
      if (key_size == 1 && accum_size == 4)
      {
        return {256,
                15,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 760}};
      }
      if (key_size == 1 && accum_size == 8)
      {
        return {
          224, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1070}};
      }
      if (key_size == 1 && accum_size == 16)
      {
        return {128,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1175}};
      }
      if (key_size == 2 && accum_size == 1)
      {
        return {
          256, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 620}};
      }
      if (key_size == 2 && accum_size == 2)
      {
        return {224,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 640}};
      }
      if (key_size == 2 && accum_size == 4)
      {
        return {256,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 905}};
      }
      if (key_size == 2 && accum_size == 8)
      {
        return {224,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 810}};
      }
      if (key_size == 2 && accum_size == 16)
      {
        return {160,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1115}};
      }
      if (key_size == 4 && accum_size == 1)
      {
        return {
          288, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1110}};
      }
      if (key_size == 4 && accum_size == 2)
      {
        return {192,
                15,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1200}};
      }
      if (key_size == 4 && accum_size == 4)
      {
        return {
          256, 15, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1110}};
      }
      if (key_size == 4 && accum_size == 8)
      {
        return {224,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1165}};
      }
      if (key_size == 4 && accum_size == 16)
      {
        return {160,
                9,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1100}};
      }
      if (key_size == 8 && accum_size == 1)
      {
        return {192,
                10,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1175}};
      }
      if (key_size == 8 && accum_size == 2)
      {
        return {
          224, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1075}};
      }
      if (key_size == 8 && accum_size == 4)
      {
        return {
          384, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1040}};
      }
      if (key_size == 8 && accum_size == 8)
      {
        return {128,
                14,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1080}};
      }
      if (key_size == 8 && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 430}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 1)
      {
        return {
          192, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1105}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 2)
      {
        return {192,
                7,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 755}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 4)
      {
        return {192,
                7,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 535}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 8)
      {
        return {
          192, 7, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS, {delay_constructor_kind::no_delay, 0, 1035}};
      }
      if (key_size == 16 && !is_primitive_key_t && accum_size == 16)
      {
        return {128,
                11,
                BLOCK_LOAD_WARP_TRANSPOSE,
                LOAD_DEFAULT,
                BLOCK_SCAN_WARP_SCANS,
                {delay_constructor_kind::no_delay, 0, 1090}};
      }
      return default_load_default();
    }

    return default_ldg();
  }
};

#if _CCCL_HAS_CONCEPTS()
template <typename T>
concept reduce_by_key_policy_selector = policy_selector<T, reduce_by_key_policy>;
#endif // _CCCL_HAS_CONCEPTS()

// TODO(bgruber): remove in CCCL 4.0 when we drop the reduce-by-key dispatchers
template <typename PolicyHub>
struct policy_selector_from_hub
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id /*arch*/) const -> reduce_by_key_policy
  {
    using ReduceByKeyPolicyT = typename PolicyHub::MaxPolicy::ReduceByKeyPolicyT;
    return reduce_by_key_policy{
      ReduceByKeyPolicyT::BLOCK_THREADS,
      ReduceByKeyPolicyT::ITEMS_PER_THREAD,
      ReduceByKeyPolicyT::LOAD_ALGORITHM,
      ReduceByKeyPolicyT::LOAD_MODIFIER,
      ReduceByKeyPolicyT::SCAN_ALGORITHM,
      delay_constructor_policy_from_type<typename ReduceByKeyPolicyT::detail::delay_constructor_t>,
    };
  }
};

template <class ReductionOpT, class AccumT, class KeyT>
struct policy_selector_from_types
{
  [[nodiscard]] _CCCL_API constexpr auto operator()(::cuda::arch_id arch) const -> reduce_by_key_policy
  {
    return policy_selector{
      int{sizeof(KeyT)},
      int{sizeof(AccumT)},
      (is_primitive_key<KeyT>() == primitive_key::yes),
      (is_primitive_accum<AccumT>() == primitive_accum::yes),
      (is_primitive_op<ReductionOpT>() == primitive_op::yes)}(arch);
  }
};
} // namespace detail::reduce_by_key

CUB_NAMESPACE_END
