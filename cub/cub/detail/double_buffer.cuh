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

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * \brief Double-buffer storage wrapper for multi-pass stream transformations that require more than one storage array
 * for streaming intermediate results back and forth.
 *
 * Many multi-pass computations require a pair of "ping-pong" storage
 * buffers (e.g., one for reading from and the other for writing to, and then
 * vice-versa for the subsequent pass).  This structure wraps a set of device
 * buffers and a "selector" member to track which is "current".
 */
template <typename T>
struct DoubleBuffer
{
  /// Pair of device buffer pointers
  T* d_buffers[2]{};

  ///  Selector into \p d_buffers (i.e., the active/valid buffer)
  int selector = 0;

  /// \brief Constructor
  DoubleBuffer() = default;

  /// \brief Constructor
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE DoubleBuffer(T* d_current, ///< The currently valid buffer
                                                   T* d_alternate) ///< Alternate storage buffer of the same size as \p
                                                                   ///< d_current
      : d_buffers{d_current, d_alternate}
  {}

  /// \brief Return pointer to the currently valid buffer
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T* Current()
  {
    return d_buffers[selector];
  }

  /// \brief Return pointer to the currently invalid buffer
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T* Alternate()
  {
    return d_buffers[selector ^ 1];
  }
};
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
