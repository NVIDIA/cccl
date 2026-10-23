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

#include <cub/detail/it_traits.cuh>

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/**
 * \brief Allows algorithms that take a value as input to take a future value that is not computed yet at launch time.
 *
 * Note that it is user's responsibility to ensure that the result will be ready before use via external synchronization
 * or stream-ordering dependencies.
 *
 * \code
 * int *d_intermediate_result;
 * allocator.DeviceAllocate((void **)&d_intermediate_result, sizeof(int));
 * compute_intermediate_result<<<blocks, threads>>>(
 *     d_intermediate_result,  // output
 *     arg1,                   // input
 *     arg2);                  // input
 * cub::FutureValue<int> init_value(d_intermediate_result);
 * cub::DeviceScan::ExclusiveScan(
 *     d_temp_storage,
 *     temp_storage_bytes,
 *     d_in,
 *     d_out,
 *     cuda::std::plus<>{},
 *     init_value,
 *     num_items);
 * allocator.DeviceFree(d_intermediate_result);
 * \endcode
 */
template <typename T, typename IterT = T*>
struct FutureValue
{
  using value_type    = T;
  using iterator_type = IterT;

  explicit _CCCL_HOST_DEVICE _CCCL_FORCEINLINE FutureValue(IterT iter)
      : m_iter(iter)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE operator T() const noexcept
  {
    return *m_iter;
  }

private:
  IterT m_iter;
};

template <typename IterT>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES FutureValue(IterT) -> FutureValue<detail::it_value_t<IterT>, IterT>;
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
