//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__DEVICE_ATTRIBUTES_
#define _CUDAX__DEVICE_ATTRIBUTES_

#include <cuda_runtime_api.h>
// cuda_runtime_api needs to come first

#include <cuda/std/__cuda/api_wrapper.h>

#include "cuda/experimental/__device/device.cuh"
#include "cuda/std/__cccl/attributes.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental
{

struct device::attrs
{
private:
  template <::cudaDeviceAttr Attribute, class Derived>
  struct __crtp
  {
    _CCCL_NODISCARD auto operator()(device __device) const
    {
      int value = 0;
      _CCCL_TRY_CUDA_API(cudaDeviceGetAttribute, "failed to get device attribute", &value, Attribute, __device.get());
      return static_cast<typename Derived::type>(value);
    }
  };

  template <::cudaDeviceAttr Attribute, class Type = int>
  struct __attr : __crtp<Attribute, __attr<Attribute, Type>>
  {
    using type = Type;
  };

public:
  // Maximum number of threads per block
  struct max_threads_per_block_t : __attr<cudaDevAttrMaxThreadsPerBlock>
  {};

  static constexpr max_threads_per_block_t max_threads_per_block{};

  // Maximum x-dimension of a block
  struct max_block_dim_x_t : __attr<cudaDevAttrMaxBlockDimX>
  {};

  static constexpr max_block_dim_x_t max_block_dim_x{};

  // Maximum y-dimension of a block
  struct max_block_dim_y_t : __attr<cudaDevAttrMaxBlockDimY>
  {};

  static constexpr max_block_dim_y_t max_block_dim_y{};

  // Maximum z-dimension of a block
  struct max_block_dim_z_t : __attr<cudaDevAttrMaxBlockDimZ>
  {};

  static constexpr max_block_dim_z_t max_block_dim_z{};

  // Maximum x-dimension of a grid
  struct max_grid_dim_x_t : __attr<cudaDevAttrMaxGridDimX>
  {};

  static constexpr max_grid_dim_x_t max_grid_dim_x{};

  // Maximum y-dimension of a grid
  struct max_grid_dim_y_t : __attr<cudaDevAttrMaxGridDimY>
  {};

  static constexpr max_grid_dim_y_t max_grid_dim_y{};

  // Maximum z-dimension of a grid
  struct max_grid_dim_z_t : __attr<cudaDevAttrMaxGridDimZ>
  {};

  static constexpr max_grid_dim_z_t max_grid_dim_z{};

  // Maximum amount of shared memory available to a thread block in bytes
  struct max_shared_memory_per_block_t : __attr<cudaDevAttrMaxSharedMemoryPerBlock>
  {};

  static constexpr max_shared_memory_per_block_t max_shared_memory_per_block{};

  // Memory available on device for __constant__ variables in a CUDA C kernel in bytes
  struct total_constant_memory_t : __attr<cudaDevAttrTotalConstantMemory>
  {};

  static constexpr total_constant_memory_t total_constant_memory{};

  // Warp size in threads
  struct warp_size_t : __attr<cudaDevAttrWarpSize>
  {};

  static constexpr warp_size_t warp_size{};

  // Maximum pitch in bytes allowed by the memory copy functions that involve
  // memory regions allocated through cudaMallocPitch()
  struct max_pitch_t : __attr<cudaDevAttrMaxPitch>
  {};

  static constexpr max_pitch_t max_pitch{};

  // Maximum 1D texture width
  struct max_texture_1d_width_t : __attr<cudaDevAttrMaxTexture1DWidth>
  {};

  static constexpr max_texture_1d_width_t max_texture_1d_width{};

  // Maximum width for a 1D texture bound to linear memory
  struct max_texture_1d_linear_width_t : __attr<cudaDevAttrMaxTexture1DLinearWidth>
  {};

  static constexpr max_texture_1d_linear_width_t max_texture_1d_linear_width{};

  // Maximum mipmapped 1D texture width
  struct max_texture_1d_mipmapped_width_t : __attr<cudaDevAttrMaxTexture1DMipmappedWidth>
  {};

  static constexpr max_texture_1d_mipmapped_width_t max_texture_1d_mipmapped_width{};

  // Maximum 2D texture width
  struct max_texture_2d_width_t : __attr<cudaDevAttrMaxTexture2DWidth>
  {};

  static constexpr max_texture_2d_width_t max_texture_2d_width{};

  // Maximum 2D texture height
  struct max_texture_2d_height_t : __attr<cudaDevAttrMaxTexture2DHeight>
  {};

  static constexpr max_texture_2d_height_t max_texture_2d_height{};

  // Maximum width for a 2D texture bound to linear memory
  struct max_texture_2d_linear_width_t : __attr<cudaDevAttrMaxTexture2DLinearWidth>
  {};

  static constexpr max_texture_2d_linear_width_t max_texture_2d_linear_width{};

  // Maximum height for a 2D texture bound to linear memory
  struct max_texture_2d_linear_height_t : __attr<cudaDevAttrMaxTexture2DLinearHeight>
  {};

  static constexpr max_texture_2d_linear_height_t max_texture_2d_linear_height{};

  // Maximum pitch in bytes for a 2D texture bound to linear memory
  struct max_texture_2d_linear_pitch_t : __attr<cudaDevAttrMaxTexture2DLinearPitch>
  {};

  static constexpr max_texture_2d_linear_pitch_t max_texture_2d_linear_pitch{};

  // Maximum mipmapped 2D texture width
  struct max_texture_2d_mipmapped_width_t : __attr<cudaDevAttrMaxTexture2DMipmappedWidth>
  {};

  static constexpr max_texture_2d_mipmapped_width_t max_texture_2d_mipmapped_width{};

  // Maximum mipmapped 2D texture height
  struct max_texture_2d_mipmapped_height_t : __attr<cudaDevAttrMaxTexture2DMipmappedHeight>
  {};

  static constexpr max_texture_2d_mipmapped_height_t max_texture_2d_mipmapped_height{};

  // Maximum 3D texture width
  struct max_texture_3d_width_t : __attr<cudaDevAttrMaxTexture3DWidth>
  {};

  static constexpr max_texture_3d_width_t max_texture_3d_width{};

  // Maximum 3D texture height
  struct max_texture_3d_height_t : __attr<cudaDevAttrMaxTexture3DHeight>
  {};

  static constexpr max_texture_3d_height_t max_texture_3d_height{};

  // Maximum 3D texture depth
  struct max_texture_3d_depth_t : __attr<cudaDevAttrMaxTexture3DDepth>
  {};

  static constexpr max_texture_3d_depth_t max_texture_3d_depth{};

  // Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported
  struct max_texture_3d_width_alt_t : __attr<cudaDevAttrMaxTexture3DWidthAlt>
  {};

  static constexpr max_texture_3d_width_alt_t max_texture_3d_width_alt{};

  // Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported
  struct max_texture_3d_height_alt_t : __attr<cudaDevAttrMaxTexture3DHeightAlt>
  {};

  static constexpr max_texture_3d_height_alt_t max_texture_3d_height_alt{};

  // Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported
  struct max_texture_3d_depth_alt_t : __attr<cudaDevAttrMaxTexture3DDepthAlt>
  {};

  static constexpr max_texture_3d_depth_alt_t max_texture_3d_depth_alt{};

  // Maximum cubemap texture width or height
  struct max_texture_cubemap_width_t : __attr<cudaDevAttrMaxTextureCubemapWidth>
  {};

  static constexpr max_texture_cubemap_width_t max_texture_cubemap_width{};

  // Maximum 1D layered texture width
  struct max_texture_1d_layered_width_t : __attr<cudaDevAttrMaxTexture1DLayeredWidth>
  {};

  static constexpr max_texture_1d_layered_width_t max_texture_1d_layered_width{};

  // Maximum layers in a 1D layered texture
  struct max_texture_1d_layered_layers_t : __attr<cudaDevAttrMaxTexture1DLayeredLayers>
  {};

  static constexpr max_texture_1d_layered_layers_t max_texture_1d_layered_layers{};

  // Maximum 2D layered texture width
  struct max_texture_2d_layered_width_t : __attr<cudaDevAttrMaxTexture2DLayeredWidth>
  {};

  static constexpr max_texture_2d_layered_width_t max_texture_2d_layered_width{};

  // Maximum 2D layered texture height
  struct max_texture_2d_layered_height_t : __attr<cudaDevAttrMaxTexture2DLayeredHeight>
  {};

  static constexpr max_texture_2d_layered_height_t max_texture_2d_layered_height{};

  // Maximum layers in a 2D layered texture
  struct max_texture_2d_layered_layers_t : __attr<cudaDevAttrMaxTexture2DLayeredLayers>
  {};

  static constexpr max_texture_2d_layered_layers_t max_texture_2d_layered_layers{};

  // Maximum cubemap layered texture width or height
  struct max_texture_cubemap_layered_width_t : __attr<cudaDevAttrMaxTextureCubemapLayeredWidth>
  {};

  static constexpr max_texture_cubemap_layered_width_t max_texture_cubemap_layered_width{};

  // Maximum layers in a cubemap layered texture
  struct max_texture_cubemap_layered_layers_t : __attr<cudaDevAttrMaxTextureCubemapLayeredLayers>
  {};

  static constexpr max_texture_cubemap_layered_layers_t max_texture_cubemap_layered_layers{};

  // Maximum 1D surface width
  struct max_surface_1d_width_t : __attr<cudaDevAttrMaxSurface1DWidth>
  {};

  static constexpr max_surface_1d_width_t max_surface_1d_width{};

  // Maximum 2D surface width
  struct max_surface_2d_width_t : __attr<cudaDevAttrMaxSurface2DWidth>
  {};

  static constexpr max_surface_2d_width_t max_surface_2d_width{};

  // Maximum 2D surface height
  struct max_surface_2d_height_t : __attr<cudaDevAttrMaxSurface2DHeight>
  {};

  static constexpr max_surface_2d_height_t max_surface_2d_height{};

  // Maximum 3D surface width
  struct max_surface_3d_width_t : __attr<cudaDevAttrMaxSurface3DWidth>
  {};

  static constexpr max_surface_3d_width_t max_surface_3d_width{};

  // Maximum 3D surface height
  struct max_surface_3d_height_t : __attr<cudaDevAttrMaxSurface3DHeight>
  {};

  static constexpr max_surface_3d_height_t max_surface_3d_height{};

  // Maximum 3D surface depth
  struct max_surface_3d_depth_t : __attr<cudaDevAttrMaxSurface3DDepth>
  {};

  static constexpr max_surface_3d_depth_t max_surface_3d_depth{};

  // Maximum 1D layered surface width
  struct max_surface_1d_layered_width_t : __attr<cudaDevAttrMaxSurface1DLayeredWidth>
  {};

  static constexpr max_surface_1d_layered_width_t max_surface_1d_layered_width{};

  // Maximum layers in a 1D layered surface
  struct max_surface_1d_layered_layers_t : __attr<cudaDevAttrMaxSurface1DLayeredLayers>
  {};

  static constexpr max_surface_1d_layered_layers_t max_surface_1d_layered_layers{};

  // Maximum 2D layered surface width
  struct max_surface_2d_layered_width_t : __attr<cudaDevAttrMaxSurface2DLayeredWidth>
  {};

  static constexpr max_surface_2d_layered_width_t max_surface_2d_layered_width{};

  // Maximum 2D layered surface height
  struct max_surface_2d_layered_height_t : __attr<cudaDevAttrMaxSurface2DLayeredHeight>
  {};

  static constexpr max_surface_2d_layered_height_t max_surface_2d_layered_height{};

  // Maximum layers in a 2D layered surface
  struct max_surface_2d_layered_layers_t : __attr<cudaDevAttrMaxSurface2DLayeredLayers>
  {};

  static constexpr max_surface_2d_layered_layers_t max_surface_2d_layered_layers{};

  // Maximum cubemap surface width
  struct max_surface_cubemap_width_t : __attr<cudaDevAttrMaxSurfaceCubemapWidth>
  {};

  static constexpr max_surface_cubemap_width_t max_surface_cubemap_width{};

  // Maximum cubemap layered surface width
  struct max_surface_cubemap_layered_width_t : __attr<cudaDevAttrMaxSurfaceCubemapLayeredWidth>
  {};

  static constexpr max_surface_cubemap_layered_width_t max_surface_cubemap_layered_width{};

  // Maximum layers in a cubemap layered surface
  struct max_surface_cubemap_layered_layers_t : __attr<cudaDevAttrMaxSurfaceCubemapLayeredLayers>
  {};

  static constexpr max_surface_cubemap_layered_layers_t max_surface_cubemap_layered_layers{};

  // Maximum number of 32-bit registers available to a thread block
  struct max_registers_per_block_t : __attr<cudaDevAttrMaxRegistersPerBlock>
  {};

  static constexpr max_registers_per_block_t max_registers_per_block{};

  // Peak clock frequency in kilohertz
  struct clock_rate_t : __attr<cudaDevAttrClockRate> // TODO: maybe a strong type for kilohertz?
  {};

  static constexpr clock_rate_t clock_rate{};

  // Alignment requirement; texture base addresses aligned to textureAlign bytes
  // do not need an offset applied to texture fetches
  struct texture_alignment_t : __attr<cudaDevAttrTextureAlignment>
  {};

  static constexpr texture_alignment_t texture_alignment{};

  // Pitch alignment requirement for 2D texture references bound to pitched memory
  struct texture_pitch_alignment_t : __attr<cudaDevAttrTexturePitchAlignment>
  {};

  static constexpr texture_pitch_alignment_t texture_pitch_alignment{};

  // true if the device can concurrently copy memory between host and device
  // while executing a kernel, or false if not
  struct gpu_overlap_t : __attr<cudaDevAttrGpuOverlap, bool>
  {};

  static constexpr gpu_overlap_t gpu_overlap{};

  // Number of multiprocessors on the device
  struct multi_processor_count_t : __attr<cudaDevAttrMultiProcessorCount>
  {};

  static constexpr multi_processor_count_t multi_processor_count{};

  // true if there is a run time limit for kernels executed on the device, or false if not
  struct kernel_exec_timeout_t : __attr<cudaDevAttrKernelExecTimeout, bool>
  {};

  static constexpr kernel_exec_timeout_t kernel_exec_timeout{};

  // true if the device is integrated with the memory subsystem, or false if not
  struct integrated_t : __attr<cudaDevAttrIntegrated, bool>
  {};

  static constexpr integrated_t integrated{};

  // true if the d
  struct can_map_host_memory_t : __attr<cudaDevAttrCanMapHostMemory, bool>
  {};

  static constexpr can_map_host_memory_t can_map_host_memory{};

  // Compute mode is the compute mode that the device is currently in.
  struct compute_mode_t : __crtp<cudaDevAttrComputeMode, compute_mode_t>
  {
    using type = enum class mode {
      _default          = cudaComputeModeDefault,
      prohibited        = cudaComputeModeProhibited,
      exclusive_process = cudaComputeModeExclusiveProcess
    };

    static constexpr mode _default          = mode::_default;
    static constexpr mode prohibited        = mode::prohibited;
    static constexpr mode exclusive_process = mode::exclusive_process;
  };

  static constexpr compute_mode_t compute_mode{};

  // true if the device supports executing multiple kernels within the same context simultaneously, or false if not.
  // It is not guaranteed that multiple kernels will be resident on the device concurrently so this feature should
  // not be relied upon for correctness.
  struct concurrent_kernels_t : __attr<cudaDevAttrConcurrentKernels, bool>
  {};

  static constexpr concurrent_kernels_t concurrent_kernels{};

  // true if error correction is enabled on the device, 0 if error correction is disabled or not supported by the
  // device
  struct ecc_enabled_t : __attr<cudaDevAttrEccEnabled, bool>
  {};

  static constexpr ecc_enabled_t ecc_enabled{};

  // PCI bus identifier of the device
  struct pci_bus_id_t : __attr<cudaDevAttrPciBusId>
  {};

  static constexpr pci_bus_id_t pci_bus_id{};

  // PCI device (also known as slot) identifier of the device
  struct pci_device_id_t : __attr<cudaDevAttrPciDeviceId>
  {};

  static constexpr pci_device_id_t pci_device_id{};

  // true if the device is using a TCC driver. TCC is only available on Tesla hardware running Windows Vista or
  struct tcc_driver_t : __attr<cudaDevAttrTccDriver, bool>
  {};

  static constexpr tcc_driver_t tcc_driver{};
  // later.

  // Peak memory clock frequency in kilohertz
  struct memory_clock_rate_t : __attr<cudaDevAttrMemoryClockRate> // TODO: maybe a strong type for kilohertz?
  {};

  static constexpr memory_clock_rate_t memory_clock_rate{};

  // Global memory bus width in bits
  struct global_memory_bus_width_t : __attr<cudaDevAttrGlobalMemoryBusWidth> // TODO: maybe a strong type for bits?
  {};

  static constexpr global_memory_bus_width_t global_memory_bus_width{};

  // Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
  struct l2_cache_size_t : __attr<cudaDevAttrL2CacheSize> // TODO: maybe a strong type for bytes?
  {};

  static constexpr l2_cache_size_t l2_cache_size{};

  // Maximum resident threads per multiprocessor
  struct max_threads_per_multi_processor_t : __attr<cudaDevAttrMaxThreadsPerMultiProcessor>
  {};

  static constexpr max_threads_per_multi_processor_t max_threads_per_multi_processor{};

  // true if the device shares a unified address space with the host, or false if not
  struct unified_addressing_t : __attr<cudaDevAttrUnifiedAddressing, bool>
  {};

  static constexpr unified_addressing_t unified_addressing{};

  // Major compute capability version number
  struct compute_capability_major_t : __attr<cudaDevAttrComputeCapabilityMajor>
  {};

  static constexpr compute_capability_major_t compute_capability_major{};

  // Minor compute capability version number
  struct compute_capability_minor_t : __attr<cudaDevAttrComputeCapabilityMinor>
  {};

  static constexpr compute_capability_minor_t compute_capability_minor{};

  // true if the device supports stream priorities, or false if not
  struct stream_priorities_supported_t : __attr<cudaDevAttrStreamPrioritiesSupported, bool>
  {};

  static constexpr stream_priorities_supported_t stream_priorities_supported{};

  // true if device supports caching globals in L1 cache, false if not
  struct global_l1_cache_supported_t : __attr<cudaDevAttrGlobalL1CacheSupported, bool>
  {};

  static constexpr global_l1_cache_supported_t global_l1_cache_supported{};

  // true if device supports caching locals in L1 cache, false if not
  struct local_l1_cache_supported_t : __attr<cudaDevAttrLocalL1CacheSupported, bool>
  {};

  static constexpr local_l1_cache_supported_t local_l1_cache_supported{};

  // Maximum amount of shared memory available to a multiprocessor in bytes; this amount is shared by all thread
  // blocks simultaneously resident on a multiprocessor
  struct max_shared_memory_per_multiprocessor_t : __attr<cudaDevAttrMaxSharedMemoryPerMultiprocessor>
  {};

  static constexpr max_shared_memory_per_multiprocessor_t max_shared_memory_per_multiprocessor{};

  // Maximum number of 32-bit registers available to a multiprocessor; this number is shared by all thread blocks
  // simultaneously resident on a multiprocessor
  struct max_registers_per_multiprocessor_t : __attr<cudaDevAttrMaxRegistersPerMultiprocessor>
  {};

  static constexpr max_registers_per_multiprocessor_t max_registers_per_multiprocessor{};

  // true if device supports allocating managed memory, false if not
  struct managed_memory_t : __attr<cudaDevAttrManagedMemory, bool>
  {};

  static constexpr managed_memory_t managed_memory{};

  // true if device is on a multi-GPU board, false if not
  struct is_multi_gpu_board_t : __attr<cudaDevAttrIsMultiGpuBoard, bool>
  {};

  static constexpr is_multi_gpu_board_t is_multi_gpu_board{};

  // Unique identifier for a group of devices on the same multi-GPU board
  struct multi_gpu_board_group_id_t : __attr<cudaDevAttrMultiGpuBoardGroupID>
  {};

  static constexpr multi_gpu_board_group_id_t multi_gpu_board_group_id{};

  // true if the link between the device and the host supports native atomic operations
  struct host_native_atomic_supported_t : __attr<cudaDevAttrHostNativeAtomicSupported, bool>
  {};

  static constexpr host_native_atomic_supported_t host_native_atomic_supported{};

  // Ratio of single precision performance (in floating-point operations per
  // second) to double precision performance
  struct single_to_double_precision_perf_ratio_t : __attr<cudaDevAttrSingleToDoublePrecisionPerfRatio>
  {};

  static constexpr single_to_double_precision_perf_ratio_t single_to_double_precision_perf_ratio{};

  // true if the device supports coherently accessing pageable memory without
  // calling cudaHostRegister on it, and false otherwise
  struct pageable_memory_access_t : __attr<cudaDevAttrPageableMemoryAccess, bool>
  {};

  static constexpr pageable_memory_access_t pageable_memory_access{};

  // true if the device can coherently access managed memory concurrently with
  // the CPU, and false otherwise
  struct concurrent_managed_access_t : __attr<cudaDevAttrConcurrentManagedAccess, bool>
  {};

  static constexpr concurrent_managed_access_t concurrent_managed_access{};

  // true if the device supports Compute Preemption, false if not
  struct compute_preemption_supported_t : __attr<cudaDevAttrComputePreemptionSupported, bool>
  {};

  static constexpr compute_preemption_supported_t compute_preemption_supported{};

  // true if the device can access host registered memory at the same virtual
  // address as the CPU, and false otherwise
  struct can_use_host_pointer_for_registered_mem_t : __attr<cudaDevAttrCanUseHostPointerForRegisteredMem, bool>
  {};

  static constexpr can_use_host_pointer_for_registered_mem_t can_use_host_pointer_for_registered_mem{};

  // true if the device supports launching cooperative kernels via
  // cudaLaunchCooperativeKernel, and false otherwise
  struct cooperative_launch_t : __attr<cudaDevAttrCooperativeLaunch, bool>
  {};

  static constexpr cooperative_launch_t cooperative_launch{};

  // true if the device supports launching cooperative kernels via
  // cudaLaunchCooperativeKernelMultiDevice, and false otherwise
  struct cooperative_multi_device_launch_t : __attr<cudaDevAttrCooperativeMultiDeviceLaunch, bool>
  {};

  static constexpr cooperative_multi_device_launch_t cooperative_multi_device_launch{};

  // true if the device supports flushing of outstanding remote writes, and
  // false otherwise
  struct can_flush_remote_writes_t : __attr<cudaDevAttrCanFlushRemoteWrites, bool>
  {};

  static constexpr can_flush_remote_writes_t can_flush_remote_writes{};

  // true if the device supports host memory registration via cudaHostRegister,
  // and false otherwise
  struct host_register_supported_t : __attr<cudaDevAttrHostRegisterSupported, bool>
  {};

  static constexpr host_register_supported_t host_register_supported{};

  // true if the device accesses pageable memory via the host's page tables, and
  // false otherwise
  struct pageable_memory_access_uses_host_page_tables_t
      : __attr<cudaDevAttrPageableMemoryAccessUsesHostPageTables, bool>
  {};

  static constexpr pageable_memory_access_uses_host_page_tables_t pageable_memory_access_uses_host_page_tables{};

  // true if the host can directly access managed memory on the device without
  // migration, and false otherwise
  struct direct_managed_mem_access_from_host_t : __attr<cudaDevAttrDirectManagedMemAccessFromHost, bool>
  {};

  static constexpr direct_managed_mem_access_from_host_t direct_managed_mem_access_from_host{};

  // Maximum per block shared memory size on the device. This value can be opted
  // into when using cudaFuncSetAttribute
  struct max_shared_memory_per_block_optin_t : __attr<cudaDevAttrMaxSharedMemoryPerBlockOptin>
  {};

  static constexpr max_shared_memory_per_block_optin_t max_shared_memory_per_block_optin{};

  // Maximum number of thread blocks that can reside on a multiprocessor
  struct max_blocks_per_multiprocessor_t : __attr<cudaDevAttrMaxBlocksPerMultiprocessor>
  {};

  static constexpr max_blocks_per_multiprocessor_t max_blocks_per_multiprocessor{};

  // Maximum L2 persisting lines capacity setting in bytes
  struct max_persisting_l2_cache_size_t : __attr<cudaDevAttrMaxPersistingL2CacheSize>
  {};

  static constexpr max_persisting_l2_cache_size_t max_persisting_l2_cache_size{};

  // Maximum value of cudaAccessPolicyWindow::num_bytes
  struct max_access_policy_window_size_t : __attr<cudaDevAttrMaxAccessPolicyWindowSize>
  {};

  static constexpr max_access_policy_window_size_t max_access_policy_window_size{};

  // Shared memory reserved by CUDA driver per block in bytes
  struct reserved_shared_memory_per_block_t : __attr<cudaDevAttrReservedSharedMemoryPerBlock>
  {};

  static constexpr reserved_shared_memory_per_block_t reserved_shared_memory_per_block{};

  // true if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
  struct sparse_cuda_array_supported_t : __attr<cudaDevAttrSparseCudaArraySupported, bool>
  {};

  static constexpr sparse_cuda_array_supported_t sparse_cuda_array_supported{};

  // Device supports using the cudaHostRegister flag cudaHostRegisterReadOnly to
  // register memory that must be mapped as read-only to the GPU
  struct host_register_read_only_supported_t : __attr<cudaDevAttrHostRegisterReadOnlySupported, bool>
  {};

  static constexpr host_register_read_only_supported_t host_register_read_only_supported{};

  // true if the device supports using the cudaMallocAsync and cudaMemPool
  // family of APIs, and false otherwise
  struct memory_pools_supported_t : __attr<cudaDevAttrMemoryPoolsSupported, bool>
  {};

  static constexpr memory_pools_supported_t memory_pools_supported{};

  // true if the device supports GPUDirect RDMA APIs, and false otherwise
  struct gpu_direct_rdma_supported_t : __attr<cudaDevAttrGPUDirectRDMASupported, bool>
  {};

  static constexpr gpu_direct_rdma_supported_t gpu_direct_rdma_supported{};

  // bitmask to be interpreted according to the
  // cudaFlushGPUDirectRDMAWritesOptions enum
  struct gpu_direct_rdma_flush_writes_options_t
      : __crtp<cudaDevAttrGPUDirectRDMAFlushWritesOptions, gpu_direct_rdma_flush_writes_options_t>
  {
    using type = enum class options : unsigned int {
      host    = cudaFlushGPUDirectRDMAWritesOptionHost,
      mem_ops = cudaFlushGPUDirectRDMAWritesOptionMemOps
    };

    friend constexpr options operator|(options lhs, options rhs) noexcept
    {
      return static_cast<options>(static_cast<unsigned int>(lhs) | static_cast<unsigned int>(rhs));
    }

    static constexpr options host    = options::host;
    static constexpr options mem_ops = options::mem_ops;
  };

  static constexpr gpu_direct_rdma_flush_writes_options_t gpu_direct_rdma_flush_writes_options{};

  // see the cudaGPUDirectRDMAWritesOrdering enum for numerical values
  struct gpu_direct_rdma_writes_ordering_t
      : __crtp<cudaDevAttrGPUDirectRDMAWritesOrdering, gpu_direct_rdma_writes_ordering_t>
  {
    using type = enum class ordering {
      none        = cudaGPUDirectRDMAWritesOrderingNone,
      owner       = cudaGPUDirectRDMAWritesOrderingOwner,
      all_devices = cudaGPUDirectRDMAWritesOrderingAllDevices
    };

    static constexpr ordering none        = ordering::none;
    static constexpr ordering owner       = ordering::owner;
    static constexpr ordering all_devices = ordering::all_devices;
  };

  static constexpr gpu_direct_rdma_writes_ordering_t gpu_direct_rdma_writes_ordering{};

  // Bitmask of handle types supported with mempool based IPC
  struct memory_pool_supported_handle_types_t : __attr<cudaDevAttrMemoryPoolSupportedHandleTypes, unsigned int> // TODO
  {};

  static constexpr memory_pool_supported_handle_types_t memory_pool_supported_handle_types{};

  // true if the device supports deferred mapping CUDA arrays and CUDA mipmapped
  // arrays.
  struct deferred_mapping_cuda_array_supported_t : __attr<cudaDevAttrDeferredMappingCudaArraySupported, bool>
  {};

  static constexpr deferred_mapping_cuda_array_supported_t deferred_mapping_cuda_array_supported{};

  // true if the device supports IPC Events.
  struct ipc_event_support_t : __attr<cudaDevAttrIpcEventSupport, bool>
  {};

  static constexpr ipc_event_support_t ipc_event_support{};

  // NUMA configuration of a device: value is of type cudaDeviceNumaConfig enum
  struct numa_config_t : __crtp<cudaDevAttrNumaConfig, numa_config_t>
  {
    using type = enum class config { //
      none      = cudaDeviceNumaConfigNone,
      numa_node = cudaDeviceNumaConfigNumaNode
    };

    static constexpr config none      = config::none;
    static constexpr config numa_node = config::numa_node;
  };

  static constexpr numa_config_t numa_config{};

  // NUMA node ID of the GPU memory
  struct numa_id_t : __attr<cudaDevAttrNumaId>
  {};

  static constexpr numa_id_t numa_id{};
};
} // namespace cuda::experimental

#endif // _CUDAX__DEVICE_ATTRIBUTES_
