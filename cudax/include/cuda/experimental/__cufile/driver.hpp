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

#include <cuda/experimental/__cufile/cufile.hpp>
#include <cuda/experimental/__cufile/detail/enums.hpp>

#include <string>

#include <cufile.h>

namespace cuda::experimental::io
{

// no additional declarations

//! C++ wrapper for CUfileDrvProps_t with convenient accessor methods
class driver_properties
{
private:
  CUfileDrvProps_t props_;

public:
  //! Initialize and get driver properties
  driver_properties()
  {
    CUfileError_t error = cuFileDriverGetProperties(&props_);
    check_cufile_result(error, "cuFileDriverGetProperties");
  }

  // NVFS Properties
  unsigned int get_nvfs_major_version() const noexcept
  {
    return props_.nvfs.major_version;
  }
  unsigned int get_nvfs_minor_version() const noexcept
  {
    return props_.nvfs.minor_version;
  }
  size_t get_poll_threshold_size() const noexcept
  {
    return props_.nvfs.poll_thresh_size;
  }
  size_t get_max_direct_io_size() const noexcept
  {
    return props_.nvfs.max_direct_io_size;
  }
  unsigned int get_status_flags() const noexcept
  {
    return props_.nvfs.dstatusflags;
  }
  unsigned int get_control_flags() const noexcept
  {
    return props_.nvfs.dcontrolflags;
  }

  // Driver Properties
  unsigned int get_feature_flags() const noexcept
  {
    return props_.fflags;
  }
  unsigned int get_max_device_cache_size() const noexcept
  {
    return props_.max_device_cache_size;
  }
  unsigned int get_per_buffer_cache_size() const noexcept
  {
    return props_.per_buffer_cache_size;
  }
  unsigned int get_max_pinned_memory_size() const noexcept
  {
    return props_.max_device_pinned_mem_size;
  }
  unsigned int get_max_batch_io_size() const noexcept
  {
    return props_.max_batch_io_size;
  }
  unsigned int get_max_batch_io_timeout_msecs() const noexcept
  {
    return props_.max_batch_io_timeout_msecs;
  }

  // Filesystem Support
  bool lustre_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_LUSTRE_SUPPORTED)) != 0;
  }
  bool wekafs_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_WEKAFS_SUPPORTED)) != 0;
  }
  bool nfs_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_NFS_SUPPORTED)) != 0;
  }
  bool gpfs_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_GPFS_SUPPORTED)) != 0;
  }
  bool nvme_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_NVME_SUPPORTED)) != 0;
  }
  bool nvmeof_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_NVMEOF_SUPPORTED)) != 0;
  }
  bool scsi_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_SCSI_SUPPORTED)) != 0;
  }
  bool beegfs_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_BEEGFS_SUPPORTED)) != 0;
  }
  bool scatefs_supported() const noexcept
  {
    return (props_.nvfs.dstatusflags & (1 << CU_FILE_SCATEFS_SUPPORTED)) != 0;
  }

  // Driver Features
  bool has_poll_mode() const noexcept
  {
    return (props_.nvfs.dcontrolflags & (1 << CU_FILE_USE_POLL_MODE)) != 0;
  }
  bool has_compat_mode() const noexcept
  {
    return (props_.nvfs.dcontrolflags & (1 << CU_FILE_ALLOW_COMPAT_MODE)) != 0;
  }
  bool dynamic_routing_supported() const noexcept
  {
    return (props_.fflags & (1 << CU_FILE_DYN_ROUTING_SUPPORTED)) != 0;
  }
  bool batch_io_supported() const noexcept
  {
    return (props_.fflags & (1 << CU_FILE_BATCH_IO_SUPPORTED)) != 0;
  }
  bool streams_supported() const noexcept
  {
    return (props_.fflags & (1 << CU_FILE_STREAMS_SUPPORTED)) != 0;
  }

  //! Get direct access to the underlying C struct
  const CUfileDrvProps_t& get_raw_properties() const noexcept;
};

// Driver Management
void driver_open()
{
  CUfileError_t error = cuFileDriverOpen();
  check_cufile_result(error, "cuFileDriverOpen");
}
void driver_close()
{
  CUfileError_t error = cuFileDriverClose();
  check_cufile_result(error, "cuFileDriverClose");
}
long driver_use_count()
{
  return cuFileUseCount();
}
driver_properties get_driver_properties()
{
  return driver_properties{};
}
int get_version()
{
  int version         = 0;
  CUfileError_t error = cuFileGetVersion(&version);
  check_cufile_result(error, "cuFileGetVersion");
  return version;
}

// Driver Configuration
void set_poll_mode(bool enabled, size_t threshold_kb)
{
  CUfileError_t error = cuFileDriverSetPollMode(enabled, threshold_kb);
  check_cufile_result(error, "cuFileDriverSetPollMode");
}
void set_max_direct_io_size(size_t size_kb)
{
  CUfileError_t error = cuFileDriverSetMaxDirectIOSize(size_kb);
  check_cufile_result(error, "cuFileDriverSetMaxDirectIOSize");
}
void set_max_cache_size(size_t size_kb)
{
  CUfileError_t error = cuFileDriverSetMaxCacheSize(size_kb);
  check_cufile_result(error, "cuFileDriverSetMaxCacheSize");
}
void set_max_pinned_memory_size(size_t size_kb)
{
  CUfileError_t error = cuFileDriverSetMaxPinnedMemSize(size_kb);
  check_cufile_result(error, "cuFileDriverSetMaxPinnedMemSize");
}

// Parameter Management
size_t get_parameter_size_t(CUFileSizeTConfigParameter_t param)
{
  size_t value;
  CUfileError_t error = cuFileGetParameterSizeT(param, &value);
  check_cufile_result(error, "cuFileGetParameterSizeT");
  return value;
}
bool get_parameter_bool(CUFileBoolConfigParameter_t param)
{
  bool value;
  CUfileError_t error = cuFileGetParameterBool(param, &value);
  check_cufile_result(error, "cuFileGetParameterBool");
  return value;
}
::std::string get_parameter_string(CUFileStringConfigParameter_t param)
{
  char buffer[1024]; // Reasonable buffer size
  CUfileError_t error = cuFileGetParameterString(param, buffer, sizeof(buffer));
  check_cufile_result(error, "cuFileGetParameterString");
  return ::std::string(buffer);
}

void set_parameter_size_t(CUFileSizeTConfigParameter_t param, size_t value)
{
  CUfileError_t error = cuFileSetParameterSizeT(param, value);
  check_cufile_result(error, "cuFileSetParameterSizeT");
}
void set_parameter_bool(CUFileBoolConfigParameter_t param, bool value)
{
  CUfileError_t error = cuFileSetParameterBool(param, value);
  check_cufile_result(error, "cuFileSetParameterBool");
}
void set_parameter_string(CUFileStringConfigParameter_t param, const ::std::string& value)
{
  CUfileError_t error = cuFileSetParameterString(param, value.c_str());
  check_cufile_result(error, "cuFileSetParameterString");
}

// Statistics Management
#ifdef CUfileStatsLevel1_t
void set_stats_level(int level)
{
  CUfileError_t error = cuFileSetStatsLevel(level);
  check_cufile_result(error, "cuFileSetStatsLevel");
}
int get_stats_level()
{
  int level;
  CUfileError_t error = cuFileGetStatsLevel(&level);
  check_cufile_result(error, "cuFileGetStatsLevel");
  return level;
}
void stats_start()
{
  CUfileError_t error = cuFileStatsStart();
  check_cufile_result(error, "cuFileStatsStart");
}
void stats_stop()
{
  CUfileError_t error = cuFileStatsStop();
  check_cufile_result(error, "cuFileStatsStop");
}
void stats_reset()
{
  CUfileError_t error = cuFileStatsReset();
  check_cufile_result(error, "cuFileStatsReset");
}
CUfileStatsLevel1_t get_stats_l1()
{
  CUfileStatsLevel1_t stats;
  CUfileError_t error = cuFileGetStatsL1(&stats);
  check_cufile_result(error, "cuFileGetStatsL1");
  return stats;
}
CUfileStatsLevel2_t get_stats_l2()
{
  CUfileStatsLevel2_t stats;
  CUfileError_t error = cuFileGetStatsL2(&stats);
  check_cufile_result(error, "cuFileGetStatsL2");
  return stats;
}
CUfileStatsLevel3_t get_stats_l3()
{
  CUfileStatsLevel3_t stats;
  CUfileError_t error = cuFileGetStatsL3(&stats);
  check_cufile_result(error, "cuFileGetStatsL3");
  return stats;
}
#endif

#ifdef cuFileGetBARSizeInKB
size_t get_bar_size_kb(int gpu_index)
{
  size_t bar_size;
  CUfileError_t error = cuFileGetBARSizeInKB(gpu_index, &bar_size);
  check_cufile_result(error, "cuFileGetBARSizeInKB");
  return bar_size;
}
#endif

// Capability Checking
bool is_cufile_library_available() noexcept
{
  int version         = 0;
  CUfileError_t error = cuFileGetVersion(&version);
  return (error.err == to_c_enum(cu_file_error::success));
}
bool is_cufile_available() noexcept
{
  CUfileDrvProps_t props;
  CUfileError_t error = cuFileDriverGetProperties(&props);
  return (error.err == to_c_enum(cu_file_error::success));
}
bool is_batch_api_available() noexcept
{
  CUfileDrvProps_t props = {};
  CUfileError_t error    = cuFileDriverGetProperties(&props);
  if (error.err != to_c_enum(cu_file_error::success))
  {
    return false;
  }
  return (props.fflags & (1 << CU_FILE_BATCH_IO_SUPPORTED)) != 0;
}
bool is_stream_api_available() noexcept
{
  CUfileDrvProps_t props = {};
  CUfileError_t error    = cuFileDriverGetProperties(&props);
  if (error.err != to_c_enum(cu_file_error::success))
  {
    return false;
  }
  return (props.fflags & (1 << CU_FILE_STREAMS_SUPPORTED)) != 0;
}

//! RAII wrapper for automatic driver management
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

// ===================== Inline implementations =====================

inline const CUfileDrvProps_t& driver_properties::get_raw_properties() const noexcept
{
  return props_;
}

// Statistics Management
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
