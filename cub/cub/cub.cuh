// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2018, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * CUB umbrella include file
 */

#pragma once

// Static configuration
#include <cub/config.cuh>

#ifndef CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK
#  if _CCCL_COMPILER(NVRTC)
#    error \
      "Including <cub/cub.cuh> is not supported when compiling with NVRTC. Include the specific device header instead (e.g. <cub/block/block_reduce.cuh>). You can define CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK to disable this warning."
#  endif // _CCCL_COMPILER(NVRTC)
#endif // CCCL_DISABLE_CUB_NVRTC_COMPATIBILITY_CHECK

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// Block
#include <cub/block/block_adjacent_difference.cuh>
#include <cub/block/block_discontinuity.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_histogram.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_radix_rank.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
// #include <cub/block/block_shift.cuh>

// Device
#include <cub/device/device_adjacent_difference.cuh>
#include <cub/device/device_copy.cuh>
#include <cub/device/device_for.cuh>
#include <cub/device/device_histogram.cuh>
#include <cub/device/device_memcpy.cuh>
#include <cub/device/device_merge.cuh>
#include <cub/device/device_merge_sort.cuh>
#include <cub/device/device_partition.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_reduce.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_select.cuh>
#include <cub/device/device_topk.cuh>
#include <cub/device/device_transform.cuh>

// Grid
#include <cub/grid/grid_even_share.cuh>
#include <cub/grid/grid_mapping.cuh>
#include <cub/grid/grid_queue.cuh>

// Thread
#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_reduce.cuh>
#include <cub/thread/thread_scan.cuh>
#include <cub/thread/thread_store.cuh>

// Warp
#include <cub/warp/warp_exchange.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_merge_sort.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_store.cuh>

// Iterator
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/iterator/cache_modified_output_iterator.cuh>
#include <cub/iterator/tex_obj_input_iterator.cuh>

// Util
#include <cub/util_allocator.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>
