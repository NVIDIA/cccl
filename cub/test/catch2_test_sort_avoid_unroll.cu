// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Defining this makes all merge/sort primitives default to not unrolling their loops. This is used by RAPIDS cuDF to
// avoid compile time issues. The macro must be defined before including any CUB header.
#define CCCL_AVOID_SORT_UNROLL

#include <cub/block/block_merge_sort.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/device/dispatch/tuning/tuning_merge.cuh>
#include <cub/device/dispatch/tuning/tuning_merge_sort.cuh>
#include <cub/thread/thread_sort.cuh>
#include <cub/warp/warp_merge_sort.cuh>

#include <cuda/std/type_traits>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

// the default for the Unroll template parameter must honor the macro
static_assert(cuda::std::is_same_v<cub::BlockMergeSort<int, 128, 4>, //
                                   cub::BlockMergeSort<int, 128, 4, cub::NullType, 1, 1, /*Unroll=*/false>>);

// WarpMergeSort has no Unroll parameter and inherits it from its base class
static_assert(cuda::std::is_base_of_v<
              cub::BlockMergeSortStrategy<int, cub::NullType, 32, 4, cub::WarpMergeSort<int, 4>, /*Unroll=*/false>,
              cub::WarpMergeSort<int, 4>>);

CUB_TEST_CASE("CCCL_AVOID_SORT_UNROLL is honored by default tuning policies", "[merge_sort][unroll]", CUB_SMALL)
{
  CHECK(!cub::MergeSortPolicy{}.unroll);
  CHECK(!cub::MergePolicy{}.unroll);
}
