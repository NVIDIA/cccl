// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2020, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
/**
 * @brief Helper class template that allows overwriting the `BLOCK_THREAD` and `ITEMS_PER_THREAD`
 * configurations of a given policy.
 */
// TODO(bgruber): this should be called something like "override_policy"
template <typename PolicyT, int BLOCK_THREADS_, int ITEMS_PER_THREAD_ = PolicyT::ITEMS_PER_THREAD>
struct policy_wrapper_t : PolicyT
{
  static constexpr int ITEMS_PER_THREAD = ITEMS_PER_THREAD_;
  static constexpr int BLOCK_THREADS    = BLOCK_THREADS_;
  static constexpr int ITEMS_PER_TILE   = BLOCK_THREADS * ITEMS_PER_THREAD;
};
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
