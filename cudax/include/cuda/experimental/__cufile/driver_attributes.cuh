//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cufile.h>

namespace cuda::experimental::cufile_driver_attributes
{
template <class _ParamEnum>
[[nodiscard]] _CCCL_CONSTEVAL auto __attr_from_param_type() noexcept
{
  if constexpr (::cuda::std::is_same_v<_ParamEnum, ::CUFileSizeTConfigParameter_t>)
  {
    return ::cuda::std::size_t{};
  }
  else if constexpr (::cuda::std::is_same_v<_ParamEnum, ::CUFileBoolConfigParameter_t>)
  {
    return bool{};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_ParamEnum>, "Unsupported parameter type");
  }
}

template <auto _Param, bool _CanBeSetWhenOpened = false>
struct __attr_from_param
{
  using __enum_type                              = decltype(_Param);
  static constexpr auto __enum_value             = _Param;
  static constexpr auto __can_be_get_when_closed = true;
  static constexpr auto __can_be_set_when_closed = true;
  static constexpr auto __can_be_set_when_opened = _CanBeSetWhenOpened;
  static constexpr auto __has_queryable_range    = ::cuda::std::is_same_v<__enum_type, ::CUFileSizeTConfigParameter_t>;

  using type = decltype(__attr_from_param_type<__enum_type>());
};

template <::CUfileDriverStatusFlags_t _Status>
struct __attr_from_status
{
  using __enum_type                              = ::CUfileDriverStatusFlags_t;
  static constexpr auto __enum_value             = _Status;
  static constexpr auto __can_be_get_when_closed = false;
  static constexpr auto __can_be_set_when_closed = false;
  static constexpr auto __can_be_set_when_opened = false;
  static constexpr auto __has_queryable_range    = false;

  using type = bool;
};

template <::CUfileFeatureFlags_t _Feature>
struct __attr_from_feature
{
  using __enum_type                              = ::CUfileFeatureFlags_t;
  static constexpr auto __enum_value             = _Feature;
  static constexpr auto __can_be_get_when_closed = false;
  static constexpr auto __can_be_set_when_closed = false;
  static constexpr auto __can_be_set_when_opened = false;
  static constexpr auto __has_queryable_range    = false;

  using type = bool;
};

using max_io_queue_depth_t       = __attr_from_param<::CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH>;
using max_io_threads_t           = __attr_from_param<::CUFILE_PARAM_EXECUTION_MAX_IO_THREADS>;
using min_io_threshold_size_kb_t = __attr_from_param<::CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB>;
using max_request_parallelism_t  = __attr_from_param<::CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM>;
using max_direct_io_size_kb_t    = __attr_from_param<::CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB, true>;
using max_device_cache_size_kb_t = __attr_from_param<::CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB, true>;
using per_buffer_cache_size_kb_t = __attr_from_param<::CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB>;
using max_device_pinned_mem_size_kb_t =
  __attr_from_param<::CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB, true>;
using io_batchsize_t                 = __attr_from_param<::CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE>;
using pollthreshold_size_kb_t        = __attr_from_param<::CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB, true>;
using batch_io_timeout_ms_t          = __attr_from_param<::CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS>;
using use_poll_mode_t                = __attr_from_param<::CUFILE_PARAM_PROPERTIES_USE_POLL_MODE, true>;
using allow_compat_mode_t            = __attr_from_param<::CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE>;
using force_compat_mode_t            = __attr_from_param<::CUFILE_PARAM_FORCE_COMPAT_MODE>;
using fs_misc_api_check_aggressive_t = __attr_from_param<::CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE>;
using parallel_io_t                  = __attr_from_param<::CUFILE_PARAM_EXECUTION_PARALLEL_IO>;
using profile_nvtx_t                 = __attr_from_param<::CUFILE_PARAM_PROFILE_NVTX>;
using allow_system_memory_t          = __attr_from_param<::CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY>;
using use_pcip2pdma_t                = __attr_from_param<::CUFILE_PARAM_USE_PCIP2PDMA>;
using prefer_io_uring_t              = __attr_from_param<::CUFILE_PARAM_PREFER_IO_URING>;
using force_odirect_mode_t           = __attr_from_param<::CUFILE_PARAM_FORCE_ODIRECT_MODE>;
using skip_topology_detection_t      = __attr_from_param<::CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION>;
using stream_memops_bypass_t         = __attr_from_param<::CUFILE_PARAM_STREAM_MEMOPS_BYPASS>;
using has_luster_support_t           = __attr_from_status<::CU_FILE_LUSTRE_SUPPORTED>;
using has_wekafs_support_t           = __attr_from_status<::CU_FILE_WEKAFS_SUPPORTED>;
using has_nfs_support_t              = __attr_from_status<::CU_FILE_NFS_SUPPORTED>;
using has_gpfs_support_t             = __attr_from_status<::CU_FILE_GPFS_SUPPORTED>;
using has_nvme_support_t             = __attr_from_status<::CU_FILE_NVME_SUPPORTED>;
using has_nvmeof_support_t           = __attr_from_status<::CU_FILE_NVMEOF_SUPPORTED>;
using has_scsi_support_t             = __attr_from_status<::CU_FILE_SCSI_SUPPORTED>;
using has_scaleflux_csd_support_t    = __attr_from_status<::CU_FILE_SCALEFLUX_CSD_SUPPORTED>;
using has_nvmesh_support_t           = __attr_from_status<::CU_FILE_NVMESH_SUPPORTED>;
using has_beegfs_support_t           = __attr_from_status<::CU_FILE_BEEGFS_SUPPORTED>;
using has_nvme_p2p_support_t         = __attr_from_status<::CU_FILE_NVME_P2P_SUPPORTED>;
using has_scatefs_support_t          = __attr_from_status<::CU_FILE_SCATEFS_SUPPORTED>;
using has_dynamic_routing_support_t  = __attr_from_feature<::CU_FILE_DYN_ROUTING_SUPPORTED>;
using has_batch_io_support_t         = __attr_from_feature<::CU_FILE_BATCH_IO_SUPPORTED>;
using has_streams_support_t          = __attr_from_feature<::CU_FILE_STREAMS_SUPPORTED>;
using has_parallel_io_support_t      = __attr_from_feature<::CU_FILE_PARALLEL_IO_SUPPORTED>;

// todo: add documentation of each attribute
//   1. type
//   2. whether it is read-only or can be set
//   3. if it can be set/read when driver is open/closed
//   4. default value, constraints

inline constexpr max_io_queue_depth_t max_io_queue_depth{};
inline constexpr max_io_threads_t max_io_threads{};
inline constexpr min_io_threshold_size_kb_t min_io_threshold_size_kb{};
inline constexpr max_request_parallelism_t max_request_parallelism{};
inline constexpr max_direct_io_size_kb_t max_direct_io_size_kb{};
inline constexpr max_device_cache_size_kb_t max_device_cache_size_kb{};
inline constexpr per_buffer_cache_size_kb_t per_buffer_cache_size_kb{};
inline constexpr max_device_pinned_mem_size_kb_t max_device_pinned_mem_size_kb{};
inline constexpr io_batchsize_t io_batchsize{};
inline constexpr pollthreshold_size_kb_t pollthreshold_size_kb{};
inline constexpr batch_io_timeout_ms_t batch_io_timeout_ms{};
inline constexpr use_poll_mode_t use_poll_mode{};
inline constexpr allow_compat_mode_t allow_compat_mode{};
inline constexpr force_compat_mode_t force_compat_mode{};
inline constexpr fs_misc_api_check_aggressive_t fs_misc_api_check_aggressive{};
inline constexpr parallel_io_t parallel_io{};
inline constexpr profile_nvtx_t profile_nvtx{};
inline constexpr allow_system_memory_t allow_system_memory{};
inline constexpr use_pcip2pdma_t use_pcip2pdma{};
inline constexpr prefer_io_uring_t prefer_io_uring{};
inline constexpr force_odirect_mode_t force_odirect_mode{};
inline constexpr skip_topology_detection_t skip_topology_detection{};
inline constexpr stream_memops_bypass_t stream_memops_bypass{};
inline constexpr has_luster_support_t has_luster_support{};
inline constexpr has_wekafs_support_t has_wekafs_support{};
inline constexpr has_nfs_support_t has_nfs_support{};
inline constexpr has_gpfs_support_t has_gpfs_support{};
inline constexpr has_nvme_support_t has_nvme_support{};
inline constexpr has_nvmeof_support_t has_nvmeof_support{};
inline constexpr has_scsi_support_t has_scsi_support{};
inline constexpr has_scaleflux_csd_support_t has_scaleflux_csd_support{};
inline constexpr has_nvmesh_support_t has_nvmesh_support{};
inline constexpr has_beegfs_support_t has_beegfs_support{};
inline constexpr has_nvme_p2p_support_t has_nvme_p2p_support{};
inline constexpr has_scatefs_support_t has_scatefs_support{};
inline constexpr has_dynamic_routing_support_t has_dynamic_routing_support{};
inline constexpr has_batch_io_support_t has_batch_io_support{};
inline constexpr has_streams_support_t has_streams_support{};
inline constexpr has_parallel_io_support_t has_parallel_io_support{};
} // namespace cuda::experimental::cufile_driver_attributes
