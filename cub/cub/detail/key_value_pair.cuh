// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * \brief A key identifier paired with a corresponding value
 */
template <typename KeyT, typename ValueT>
struct KeyValuePair
{
  using Key   = KeyT; ///< Key data type
  using Value = ValueT; ///< Value data type

  Key key; ///< Item key
  Value value; ///< Item value

  /// Constructor
  _CCCL_FORCEINLINE KeyValuePair() = default;

  /// Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair(Key const& key, Value const& value)
      : key(key)
      , value(value)
  {}

  /// Equality operator
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator==(const KeyValuePair& b) const
  {
    return (value == b.value) && (key == b.key);
  }

  /// Inequality operator
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE bool operator!=(const KeyValuePair& b) const
  {
    return (value != b.value) || (key != b.key);
  }

#  if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const KeyValuePair& pair)
  {
    return os << '(' << pair.key << ',' << pair.value << ')';
  }
#  endif // _CCCL_HOSTED()
};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
