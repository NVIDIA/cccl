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

#include <cuda/std/type_traits>

#include <cuda/experimental/__cufile/cufile.hpp>
#include <cuda/experimental/__cufile/detail/enums.hpp>

#include <string>

#include <cufile.h>

namespace cuda::experimental::io
{

namespace __detail
{

// Base template for driver parameter attributes
template <auto _Param, typename _Type>
struct __driver_param_impl
{
  using type = _Type;

  [[nodiscard]] constexpr auto operator()() const
  {
    return static_cast<type>(_get_parameter_value());
  }

private:
  [[nodiscard]] auto _get_parameter_value() const
  {
    if constexpr (::cuda::std::is_same_v<type, size_t>)
    {
      size_t value;
      CUfileError_t error = cuFileGetParameterSizeT(static_cast<CUFileSizeTConfigParameter_t>(_Param), &value);
      check_cufile_result(error, "cuFileGetParameterSizeT");
      return value;
    }
    else if constexpr (::cuda::std::is_same_v<type, bool>)
    {
      bool value;
      CUfileError_t error = cuFileGetParameterBool(static_cast<CUFileBoolConfigParameter_t>(_Param), &value);
      check_cufile_result(error, "cuFileGetParameterBool");
      return value;
    }
    else
    {
      static_assert(::cuda::std::is_same_v<type, void>, "Unsupported parameter type");
    }
  }
};

// Template for size_t parameters
template <auto _Param>
struct __driver_param_size_t : __driver_param_impl<_Param, size_t>
{};

// Template for bool parameters
template <auto _Param>
struct __driver_param_bool : __driver_param_impl<_Param, bool>
{};

} // namespace __detail

// ================================================================================================
// Attribute-based API
// ================================================================================================

// (removed driver_parameters; unified below)

namespace driver_attributes
{
// ----------------------
// Parameters (SizeT)
// ----------------------
using profile_stats_t = __detail::__driver_param_size_t<CUFILE_PARAM_PROFILE_STATS>;
static constexpr profile_stats_t profile_stats{};
using execution_max_io_queue_depth_t = __detail::__driver_param_size_t<CUFILE_PARAM_EXECUTION_MAX_IO_QUEUE_DEPTH>;
static constexpr execution_max_io_queue_depth_t execution_max_io_queue_depth{};
using execution_max_io_threads_t = __detail::__driver_param_size_t<CUFILE_PARAM_EXECUTION_MAX_IO_THREADS>;
static constexpr execution_max_io_threads_t execution_max_io_threads{};
using execution_min_io_threshold_size_kb_t =
  __detail::__driver_param_size_t<CUFILE_PARAM_EXECUTION_MIN_IO_THRESHOLD_SIZE_KB>;
static constexpr execution_min_io_threshold_size_kb_t execution_min_io_threshold_size_kb{};
using execution_max_request_parallelism_t =
  __detail::__driver_param_size_t<CUFILE_PARAM_EXECUTION_MAX_REQUEST_PARALLELISM>;
static constexpr execution_max_request_parallelism_t execution_max_request_parallelism{};
using properties_max_direct_io_size_kb_t =
  __detail::__driver_param_size_t<CUFILE_PARAM_PROPERTIES_MAX_DIRECT_IO_SIZE_KB>;
static constexpr properties_max_direct_io_size_kb_t properties_max_direct_io_size_kb{};
using properties_max_device_cache_size_kb_t =
  __detail::__driver_param_size_t<CUFILE_PARAM_PROPERTIES_MAX_DEVICE_CACHE_SIZE_KB>;
static constexpr properties_max_device_cache_size_kb_t properties_max_device_cache_size_kb{};
using properties_per_buffer_cache_size_kb_t =
  __detail::__driver_param_size_t<CUFILE_PARAM_PROPERTIES_PER_BUFFER_CACHE_SIZE_KB>;
static constexpr properties_per_buffer_cache_size_kb_t properties_per_buffer_cache_size_kb{};
using properties_max_device_pinned_mem_size_kb_t =
  __detail::__driver_param_size_t<CUFILE_PARAM_PROPERTIES_MAX_DEVICE_PINNED_MEM_SIZE_KB>;
static constexpr properties_max_device_pinned_mem_size_kb_t properties_max_device_pinned_mem_size_kb{};
using properties_io_batchsize_t = __detail::__driver_param_size_t<CUFILE_PARAM_PROPERTIES_IO_BATCHSIZE>;
static constexpr properties_io_batchsize_t properties_io_batchsize{};
using pollthreshold_size_kb_t = __detail::__driver_param_size_t<CUFILE_PARAM_POLLTHRESHOLD_SIZE_KB>;
static constexpr pollthreshold_size_kb_t pollthreshold_size_kb{};
using properties_batch_io_timeout_ms_t = __detail::__driver_param_size_t<CUFILE_PARAM_PROPERTIES_BATCH_IO_TIMEOUT_MS>;
static constexpr properties_batch_io_timeout_ms_t properties_batch_io_timeout_ms{};

// ----------------------
// Parameters (Bool)
// ----------------------
using properties_use_poll_mode_t = __detail::__driver_param_bool<CUFILE_PARAM_PROPERTIES_USE_POLL_MODE>;
static constexpr properties_use_poll_mode_t properties_use_poll_mode{};
using properties_allow_compat_mode_t = __detail::__driver_param_bool<CUFILE_PARAM_PROPERTIES_ALLOW_COMPAT_MODE>;
static constexpr properties_allow_compat_mode_t properties_allow_compat_mode{};
using force_compat_mode_t = __detail::__driver_param_bool<CUFILE_PARAM_FORCE_COMPAT_MODE>;
static constexpr force_compat_mode_t force_compat_mode{};
using fs_misc_api_check_aggressive_t = __detail::__driver_param_bool<CUFILE_PARAM_FS_MISC_API_CHECK_AGGRESSIVE>;
static constexpr fs_misc_api_check_aggressive_t fs_misc_api_check_aggressive{};
using execution_parallel_io_t = __detail::__driver_param_bool<CUFILE_PARAM_EXECUTION_PARALLEL_IO>;
static constexpr execution_parallel_io_t execution_parallel_io{};
using profile_nvtx_t = __detail::__driver_param_bool<CUFILE_PARAM_PROFILE_NVTX>;
static constexpr profile_nvtx_t profile_nvtx{};
using properties_allow_system_memory_t = __detail::__driver_param_bool<CUFILE_PARAM_PROPERTIES_ALLOW_SYSTEM_MEMORY>;
static constexpr properties_allow_system_memory_t properties_allow_system_memory{};
using use_pcip2pdma_t = __detail::__driver_param_bool<CUFILE_PARAM_USE_PCIP2PDMA>;
static constexpr use_pcip2pdma_t use_pcip2pdma{};
using prefer_io_uring_t = __detail::__driver_param_bool<CUFILE_PARAM_PREFER_IO_URING>;
static constexpr prefer_io_uring_t prefer_io_uring{};
using force_odirect_mode_t = __detail::__driver_param_bool<CUFILE_PARAM_FORCE_ODIRECT_MODE>;
static constexpr force_odirect_mode_t force_odirect_mode{};
using skip_topology_detection_t = __detail::__driver_param_bool<CUFILE_PARAM_SKIP_TOPOLOGY_DETECTION>;
static constexpr skip_topology_detection_t skip_topology_detection{};
using stream_memops_bypass_t = __detail::__driver_param_bool<CUFILE_PARAM_STREAM_MEMOPS_BYPASS>;
static constexpr stream_memops_bypass_t stream_memops_bypass{};

// ----------------------
// Properties via cuFileDriverGetProperties
// ----------------------
// Fetch properties from cuFileDriverGetProperties and extract specific fields
struct nvfs_major_version_t
{
  using type = unsigned int;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return props.nvfs.major_version;
  }
};
static constexpr nvfs_major_version_t nvfs_major_version{};

struct nvfs_minor_version_t
{
  using type = unsigned int;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return props.nvfs.minor_version;
  }
};
static constexpr nvfs_minor_version_t nvfs_minor_version{};

struct feature_flags_t
{
  using type = unsigned int;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return props.fflags;
  }
};
static constexpr feature_flags_t feature_flags{};

// Filesystem support (from dstatusflags)
struct lustre_supported_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.nvfs.dstatusflags & (1 << CU_FILE_LUSTRE_SUPPORTED)) != 0;
  }
};
static constexpr lustre_supported_t lustre_supported{};

struct nfs_supported_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.nvfs.dstatusflags & (1 << CU_FILE_NFS_SUPPORTED)) != 0;
  }
};
static constexpr nfs_supported_t nfs_supported{};

struct nvme_supported_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.nvfs.dstatusflags & (1 << CU_FILE_NVME_SUPPORTED)) != 0;
  }
};
static constexpr nvme_supported_t nvme_supported{};

// Feature flags (fflags)
struct dynamic_routing_supported_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.fflags & (1 << CU_FILE_DYN_ROUTING_SUPPORTED)) != 0;
  }
};
static constexpr dynamic_routing_supported_t dynamic_routing_supported{};

struct batch_io_supported_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.fflags & (1 << CU_FILE_BATCH_IO_SUPPORTED)) != 0;
  }
};
static constexpr batch_io_supported_t batch_io_supported{};

struct streams_supported_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.fflags & (1 << CU_FILE_STREAMS_SUPPORTED)) != 0;
  }
};
static constexpr streams_supported_t streams_supported{};

// Control flags (dcontrolflags)
struct use_poll_mode_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.nvfs.dcontrolflags & (1 << CU_FILE_USE_POLL_MODE)) != 0;
  }
};
static constexpr use_poll_mode_t use_poll_mode{};

struct allow_compat_mode_t
{
  using type = bool;
  type operator()() const
  {
    CUfileDrvProps_t props{};
    CUfileError_t error = cuFileDriverGetProperties(&props);
    check_cufile_result(error, "cuFileDriverGetProperties");
    return (props.nvfs.dcontrolflags & (1 << CU_FILE_ALLOW_COMPAT_MODE)) != 0;
  }
};
static constexpr allow_compat_mode_t allow_compat_mode{};
} // namespace driver_attributes

inline void driver_open()
{
  CUfileError_t error = cuFileDriverOpen();
  check_cufile_result(error, "cuFileDriverOpen");
}
inline void driver_close()
{
  CUfileError_t error = cuFileDriverClose();
  check_cufile_result(error, "cuFileDriverClose");
}
inline long driver_use_count()
{
  return cuFileUseCount();
}
// No get_driver_properties(): use driver_attributes or driver_parameters
inline int get_version()
{
  int version         = 0;
  CUfileError_t error = cuFileGetVersion(&version);
  check_cufile_result(error, "cuFileGetVersion");
  return version;
}

inline void set_poll_mode(bool enabled, size_t threshold_kb)
{
  CUfileError_t error = cuFileDriverSetPollMode(enabled, threshold_kb);
  check_cufile_result(error, "cuFileDriverSetPollMode");
}
inline void set_max_direct_io_size(size_t size_kb)
{
  CUfileError_t error = cuFileDriverSetMaxDirectIOSize(size_kb);
  check_cufile_result(error, "cuFileDriverSetMaxDirectIOSize");
}
inline void set_max_cache_size(size_t size_kb)
{
  CUfileError_t error = cuFileDriverSetMaxCacheSize(size_kb);
  check_cufile_result(error, "cuFileDriverSetMaxCacheSize");
}
inline void set_max_pinned_memory_size(size_t size_kb)
{
  CUfileError_t error = cuFileDriverSetMaxPinnedMemSize(size_kb);
  check_cufile_result(error, "cuFileDriverSetMaxPinnedMemSize");
}

// Parameter setting functions (kept for backward compatibility)
inline void set_parameter_size_t(CUFileSizeTConfigParameter_t param, size_t value)
{
  CUfileError_t error = cuFileSetParameterSizeT(param, value);
  check_cufile_result(error, "cuFileSetParameterSizeT");
}
inline void set_parameter_bool(CUFileBoolConfigParameter_t param, bool value)
{
  CUfileError_t error = cuFileSetParameterBool(param, value);
  check_cufile_result(error, "cuFileSetParameterBool");
}

#ifdef CUfileStatsLevel1_t
inline void set_stats_level(int level)
{
  CUfileError_t error = cuFileSetStatsLevel(level);
  check_cufile_result(error, "cuFileSetStatsLevel");
}
inline int get_stats_level()
{
  int level;
  CUfileError_t error = cuFileGetStatsLevel(&level);
  check_cufile_result(error, "cuFileGetStatsLevel");
  return level;
}
inline void stats_start()
{
  CUfileError_t error = cuFileStatsStart();
  check_cufile_result(error, "cuFileStatsStart");
}
inline void stats_stop()
{
  CUfileError_t error = cuFileStatsStop();
  check_cufile_result(error, "cuFileStatsStop");
}
inline void stats_reset()
{
  CUfileError_t error = cuFileStatsReset();
  check_cufile_result(error, "cuFileStatsReset");
}
inline CUfileStatsLevel1_t get_stats_l1()
{
  CUfileStatsLevel1_t stats;
  CUfileError_t error = cuFileGetStatsL1(&stats);
  check_cufile_result(error, "cuFileGetStatsL1");
  return stats;
}
inline CUfileStatsLevel2_t get_stats_l2()
{
  CUfileStatsLevel2_t stats;
  CUfileError_t error = cuFileGetStatsL2(&stats);
  check_cufile_result(error, "cuFileGetStatsL2");
  return stats;
}
inline CUfileStatsLevel3_t get_stats_l3()
{
  CUfileStatsLevel3_t stats;
  CUfileError_t error = cuFileGetStatsL3(&stats);
  check_cufile_result(error, "cuFileGetStatsL3");
  return stats;
}
#endif

inline bool is_cufile_library_available() noexcept
{
  int version         = 0;
  CUfileError_t error = cuFileGetVersion(&version);
  return (error.err == to_c_enum(cu_file_error::success));
}
inline bool is_cufile_available() noexcept
{
  CUfileDrvProps_t props;
  CUfileError_t error = cuFileDriverGetProperties(&props);
  return (error.err == to_c_enum(cu_file_error::success));
}
inline bool is_batch_api_available() noexcept
{
  CUfileDrvProps_t props = {};
  CUfileError_t error    = cuFileDriverGetProperties(&props);
  if (error.err != to_c_enum(cu_file_error::success))
  {
    return false;
  }
  return (props.fflags & (1 << CU_FILE_BATCH_IO_SUPPORTED)) != 0;
}
inline bool is_stream_api_available() noexcept
{
  CUfileDrvProps_t props = {};
  CUfileError_t error    = cuFileDriverGetProperties(&props);
  if (error.err != to_c_enum(cu_file_error::success))
  {
    return false;
  }
  return (props.fflags & (1 << CU_FILE_STREAMS_SUPPORTED)) != 0;
}

class driver_handle
{
public:
  driver_handle()
  {
    cuda::experimental::io::driver_open();
  }
  ~driver_handle() noexcept
  {
    cuda::experimental::io::driver_close();
  }

  driver_handle(const driver_handle&)            = delete;
  driver_handle& operator=(const driver_handle&) = delete;
  driver_handle(driver_handle&&)                 = default;
  driver_handle& operator=(driver_handle&&)      = default;
};

#ifdef cuFileGetBARSizeInKB
inline size_t get_bar_size_kb(int gpu_index)
{
  size_t bar_size;
  CUfileError_t error = cuFileGetBARSizeInKB(gpu_index, &bar_size);
  check_cufile_result(error, "cuFileGetBARSizeInKB");
  return bar_size;
}
#endif
} // namespace cuda::experimental::io
