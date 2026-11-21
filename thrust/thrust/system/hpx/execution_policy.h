// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/*! \file thrust/system/hpx/execution_policy.h
 *  \brief Execution policies for Thrust's Standard C++ system.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// get the execution policies definitions first
#include <thrust/system/hpx/detail/execution_policy.h>

// get the definition of par, par_unseq, seq, unseq
#include <thrust/system/hpx/detail/par.h>
#include <thrust/system/hpx/detail/par_unseq.h>
#include <thrust/system/hpx/detail/seq.h>
#include <thrust/system/hpx/detail/unseq.h>

// now get all the algorithm definitions

#include <thrust/system/hpx/detail/adjacent_difference.h>
#include <thrust/system/hpx/detail/assign_value.h>
#include <thrust/system/hpx/detail/binary_search.h>
#include <thrust/system/hpx/detail/copy.h>
#include <thrust/system/hpx/detail/copy_if.h>
#include <thrust/system/hpx/detail/count.h>
#include <thrust/system/hpx/detail/equal.h>
#include <thrust/system/hpx/detail/extrema.h>
#include <thrust/system/hpx/detail/fill.h>
#include <thrust/system/hpx/detail/find.h>
#include <thrust/system/hpx/detail/for_each.h>
#include <thrust/system/hpx/detail/gather.h>
#include <thrust/system/hpx/detail/generate.h>
#include <thrust/system/hpx/detail/get_value.h>
#include <thrust/system/hpx/detail/inner_product.h>
#include <thrust/system/hpx/detail/iter_swap.h>
#include <thrust/system/hpx/detail/logical.h>
#include <thrust/system/hpx/detail/malloc_and_free.h>
#include <thrust/system/hpx/detail/merge.h>
#include <thrust/system/hpx/detail/mismatch.h>
#include <thrust/system/hpx/detail/partition.h>
#include <thrust/system/hpx/detail/reduce.h>
#include <thrust/system/hpx/detail/reduce_by_key.h>
#include <thrust/system/hpx/detail/remove.h>
#include <thrust/system/hpx/detail/replace.h>
#include <thrust/system/hpx/detail/reverse.h>
#include <thrust/system/hpx/detail/scan.h>
#include <thrust/system/hpx/detail/scan_by_key.h>
#include <thrust/system/hpx/detail/scatter.h>
#include <thrust/system/hpx/detail/sequence.h>
#include <thrust/system/hpx/detail/set_operations.h>
#include <thrust/system/hpx/detail/sort.h>
#include <thrust/system/hpx/detail/swap_ranges.h>
#include <thrust/system/hpx/detail/tabulate.h>
#include <thrust/system/hpx/detail/transform.h>
#include <thrust/system/hpx/detail/transform_reduce.h>
#include <thrust/system/hpx/detail/transform_scan.h>
#include <thrust/system/hpx/detail/uninitialized_copy.h>
#include <thrust/system/hpx/detail/uninitialized_fill.h>
#include <thrust/system/hpx/detail/unique.h>
#include <thrust/system/hpx/detail/unique_by_key.h>
