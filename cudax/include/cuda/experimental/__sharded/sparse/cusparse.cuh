//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Shared cuSPARSE plumbing of the sparse tier: `cusparse_safe_call`,
 *        the element-type mapping, `cusparse_handles` (one place-bound
 *        handle per place of a group) and the output-shard matching helper
 *        the SpMV / SpMM plans share.
 *
 * Opt-in vendor tier: requires the cuSPARSE headers; not part of the
 * `cuda/experimental/sharded.cuh` umbrella.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !__has_include(<cusparse.h>)
#  error "<cuda/experimental/__sharded/sparse/cusparse.cuh> requires the cuSPARSE headers (cusparse.h) to be installed"
#endif // !__has_include(<cusparse.h>)

#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/source_location>

#include <cuda/experimental/__places/place_group.cuh>
#include <cuda/experimental/__places/places.cuh>
#include <cuda/experimental/__sharded/container/csr.cuh>
#include <cuda/experimental/__sharded/container/sharded_array.cuh>

#include <cstdint>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

namespace cuda::experimental::sharded
{
/**
 * @brief Throw `std::runtime_error` when a cuSPARSE call does not return
 * `CUSPARSE_STATUS_SUCCESS` (the sharded-scope counterpart of
 * `cuda_safe_call`, following the same optional-vendor-status precedent).
 */
inline void cusparse_safe_call(cusparseStatus_t status,
                               const ::cuda::std::source_location loc = ::cuda::std::source_location::current())
{
  if (status != CUSPARSE_STATUS_SUCCESS)
  {
    _CCCL_THROW(::std::runtime_error,
                ::std::string(loc.file_name()) + "(" + ::std::to_string(loc.line())
                  + "): cuSPARSE error: " + cusparseGetErrorString(status));
  }
}

namespace reserved
{
/// @brief Maps element types to cuSPARSE data types (FP64 is the primary
/// target; FP32 is provided for completeness).
template <typename _Tp>
struct cusparse_data_type;

template <>
struct cusparse_data_type<double>
{
  static constexpr cudaDataType value = CUDA_R_64F;
};

template <>
struct cusparse_data_type<float>
{
  static constexpr cudaDataType value = CUDA_R_32F;
};

/**
 * @brief Match an output array against a matrix's row partition.
 *
 * `sharded_array` allocation skips zero-size shards, so an output for a
 * matrix with row-less shards has fewer shards than the matrix. Walk both
 * sides, pair every rows>0 matrix shard with the next output shard, and
 * validate sizes; also refuse nnz==0 shards, which cuSPARSE descriptors do
 * not support.
 *
 * @return (matrix shard index, output pointer) for every participating shard
 */
template <typename _Tp>
::std::vector<::std::pair<size_t, _Tp*>>
matched_output_shards(const sharded_csr<_Tp>& A, sharded_array<_Tp>& out, ::std::int64_t n_cols, const char* what)
{
  ::std::vector<::std::pair<size_t, _Tp*>> pairs;
  size_t out_idx = 0;
  for (size_t i = 0; i < A.num_shards(); i++)
  {
    const auto& sh = A.shard(i);
    if (sh.rows == 0)
    {
      continue;
    }
    if (sh.nnz == 0)
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": shard " + ::std::to_string(i)
                    + " has rows but no nonzeros; adjust the row boundaries");
    }
    if (out_idx >= out.num_shards()
        || out.shard(out_idx).size != static_cast<size_t>(sh.rows) * static_cast<size_t>(n_cols))
    {
      _CCCL_THROW(::std::invalid_argument,
                  ::std::string(what) + ": output is not row-partitioned like the matrix "
                    + "(use sharded_csr::make_row_partitioned)");
    }
    pairs.emplace_back(i, out.shard(out_idx).data);
    out_idx++;
  }
  if (out_idx != out.num_shards())
  {
    _CCCL_THROW(::std::invalid_argument,
                ::std::string(what) + ": output has more shards than the matrix has row ranges");
  }
  return pairs;
}
} // namespace reserved

/**
 * @brief PLACE-BOUND cuSPARSE state: one handle per place of a group, created
 * lazily under the place's exec scope on first use, shared by every matrix
 * and plan built over the group.
 *
 * Create ONE of these next to the group and pass it to the plans; it must
 * outlive them. The stream is never bound here — plans rebind it per call,
 * since one handle serves every matrix at its place. Lazy creation is
 * mutex-guarded (thread-safe); teardown is best-effort (no throwing).
 */
class cusparse_handles
{
public:
  explicit cusparse_handles(places::place_group& group)
      : group_(&group)
      , handles_(group.size(), nullptr)
  {}

  cusparse_handles(const cusparse_handles&)            = delete;
  cusparse_handles& operator=(const cusparse_handles&) = delete;

  /// @brief The handle of the idx-th place, created on first use under the
  /// place's exec scope (so lazy library state lands in the confined context
  /// that runs the calls).
  cusparseHandle_t get(size_t place_idx)
  {
    _CCCL_ASSERT(place_idx < handles_.size(), "cusparse_handles: place index out of range");
    ::std::lock_guard<::std::mutex> lock(mutex_);
    if (!handles_[place_idx])
    {
      places::exec_place_scope scope(group_->place(place_idx));
      cusparse_safe_call(cusparseCreate(&handles_[place_idx]));
    }
    return handles_[place_idx];
  }

  places::place_group& group() const
  {
    return *group_;
  }

  ~cusparse_handles()
  {
    for (cusparseHandle_t h : handles_)
    {
      if (h)
      {
        cusparseDestroy(h); // best-effort
      }
    }
  }

private:
  places::place_group* group_;
  ::std::mutex mutex_;
  ::std::vector<cusparseHandle_t> handles_;
};
} // namespace cuda::experimental::sharded
