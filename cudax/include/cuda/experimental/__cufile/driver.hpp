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

#include <memory>
#include <string>
#include <vector>

#include <cufile.h>

#include <cuda/experimental/__cufile/detail/error_handling.hpp>

namespace cuda::experimental::cufile
{

/**
 * @brief C++ wrapper for CUfileDrvProps_t with convenient accessor methods
 */
class driver_properties
{
private:
  CUfileDrvProps_t props_;

public:
  /**
   * @brief Initialize and get driver properties
   */
  driver_properties();

  // NVFS Properties
  unsigned int get_nvfs_major_version() const noexcept;
  unsigned int get_nvfs_minor_version() const noexcept;
  size_t get_poll_threshold_size() const noexcept;
  size_t get_max_direct_io_size() const noexcept;
  unsigned int get_status_flags() const noexcept;
  unsigned int get_control_flags() const noexcept;

  // Driver Properties
  unsigned int get_feature_flags() const noexcept;
  unsigned int get_max_device_cache_size() const noexcept;
  unsigned int get_per_buffer_cache_size() const noexcept;
  unsigned int get_max_pinned_memory_size() const noexcept;
  unsigned int get_max_batch_io_size() const noexcept;
  unsigned int get_max_batch_io_timeout_msecs() const noexcept;

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

  /**
   * @brief Get direct access to the underlying C struct
   */
  const CUfileDrvProps_t& get_raw_properties() const noexcept;
};

// Driver Management
void driver_open();
void driver_close();
long driver_use_count();
driver_properties get_driver_properties();
int get_version();

// Driver Configuration
void set_poll_mode(bool enabled, size_t threshold_kb);
void set_max_direct_io_size(size_t size_kb);
void set_max_cache_size(size_t size_kb);
void set_max_pinned_memory_size(size_t size_kb);

// Parameter Management
size_t get_parameter_size_t(CUFileSizeTConfigParameter_t param);
bool get_parameter_bool(CUFileBoolConfigParameter_t param);
::std::string get_parameter_string(CUFileStringConfigParameter_t param);

void set_parameter_size_t(CUFileSizeTConfigParameter_t param, size_t value);
void set_parameter_bool(CUFileBoolConfigParameter_t param, bool value);
void set_parameter_string(CUFileStringConfigParameter_t param, const ::std::string& value);

// Statistics Management
#ifdef CUfileStatsLevel1_t
void set_stats_level(int level);
int get_stats_level();
void stats_start();
void stats_stop();
void stats_reset();
CUfileStatsLevel1_t get_stats_l1();
CUfileStatsLevel2_t get_stats_l2();
CUfileStatsLevel3_t get_stats_l3();
#endif

#ifdef cuFileGetBARSizeInKB
size_t get_bar_size_kb(int gpu_index);
#endif

// Capability Checking
bool is_cufile_library_available() noexcept;
bool is_cufile_available() noexcept;
bool is_batch_api_available() noexcept;
bool is_stream_api_available() noexcept;

/**
 * @brief RAII wrapper for automatic driver management
 */
class driver_handle
{
public:
  driver_handle()
  {
    driver_open();
  }
  ~driver_handle() noexcept
  {
    try
    {
      driver_close();
    }
    catch (...)
    {}
  }

  driver_handle(const driver_handle&)            = delete;
  driver_handle& operator=(const driver_handle&) = delete;
  driver_handle(driver_handle&&)                 = default;
  driver_handle& operator=(driver_handle&&)      = default;
};

} // namespace cuda::experimental::cufile

#include <cuda/experimental/__cufile/detail/driver_impl.hpp>
