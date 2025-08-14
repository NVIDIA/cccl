//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DEVICE_ATTRIBUTES_H
#define _CUDA___DEVICE_ATTRIBUTES_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__device/device_ref.h>
#  include <cuda/std/__cccl/attributes.h>
#  include <cuda/std/__cuda/api_wrapper.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

namespace __detail
{

template <::cudaDeviceAttr _Attr, typename _Type>
struct __dev_attr_impl
{
  using type = _Type;

  [[nodiscard]] constexpr operator ::cudaDeviceAttr() const noexcept
  {
    return _Attr;
  }

  [[nodiscard]] type operator()(device_ref __dev_id) const
  {
    int __value = 0;
    _CCCL_TRY_CUDA_API(::cudaDeviceGetAttribute, "failed to get device attribute", &__value, _Attr, __dev_id.get());
    return static_cast<type>(__value);
  }
};

template <::cudaDeviceAttr _Attr>
struct __dev_attr : __dev_attr_impl<_Attr, int>
{};

// TODO: give this a strong type for kilohertz
template <>
struct __dev_attr<::cudaDevAttrClockRate> //
    : __dev_attr_impl<::cudaDevAttrClockRate, int>
{};
template <>
struct __dev_attr<::cudaDevAttrGpuOverlap> //
    : __dev_attr_impl<::cudaDevAttrGpuOverlap, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrKernelExecTimeout> //
    : __dev_attr_impl<::cudaDevAttrKernelExecTimeout, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrIntegrated> //
    : __dev_attr_impl<::cudaDevAttrIntegrated, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrCanMapHostMemory> //
    : __dev_attr_impl<::cudaDevAttrCanMapHostMemory, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrComputeMode> //
    : __dev_attr_impl<::cudaDevAttrComputeMode, ::cudaComputeMode>
{
  static constexpr type default_mode           = cudaComputeModeDefault;
  static constexpr type prohibited_mode        = cudaComputeModeProhibited;
  static constexpr type exclusive_process_mode = cudaComputeModeExclusiveProcess;
};
template <>
struct __dev_attr<::cudaDevAttrConcurrentKernels> //
    : __dev_attr_impl<::cudaDevAttrConcurrentKernels, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrEccEnabled> //
    : __dev_attr_impl<::cudaDevAttrEccEnabled, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrTccDriver> //
    : __dev_attr_impl<::cudaDevAttrTccDriver, bool>
{};
// TODO: give this a strong type for kilohertz
template <>
struct __dev_attr<::cudaDevAttrMemoryClockRate> //
    : __dev_attr_impl<::cudaDevAttrMemoryClockRate, int>
{};
// TODO: give this a strong type for bits
template <>
struct __dev_attr<::cudaDevAttrGlobalMemoryBusWidth> //
    : __dev_attr_impl<::cudaDevAttrGlobalMemoryBusWidth, int>
{};
// TODO: give this a strong type for bytes
template <>
struct __dev_attr<::cudaDevAttrL2CacheSize> //
    : __dev_attr_impl<::cudaDevAttrL2CacheSize, int>
{};
template <>
struct __dev_attr<::cudaDevAttrUnifiedAddressing> //
    : __dev_attr_impl<::cudaDevAttrUnifiedAddressing, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrStreamPrioritiesSupported> //
    : __dev_attr_impl<::cudaDevAttrStreamPrioritiesSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrGlobalL1CacheSupported> //
    : __dev_attr_impl<::cudaDevAttrGlobalL1CacheSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrLocalL1CacheSupported> //
    : __dev_attr_impl<::cudaDevAttrLocalL1CacheSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrManagedMemory> //
    : __dev_attr_impl<::cudaDevAttrManagedMemory, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrIsMultiGpuBoard> //
    : __dev_attr_impl<::cudaDevAttrIsMultiGpuBoard, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrHostNativeAtomicSupported> //
    : __dev_attr_impl<::cudaDevAttrHostNativeAtomicSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrPageableMemoryAccess> //
    : __dev_attr_impl<::cudaDevAttrPageableMemoryAccess, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrConcurrentManagedAccess> //
    : __dev_attr_impl<::cudaDevAttrConcurrentManagedAccess, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrComputePreemptionSupported> //
    : __dev_attr_impl<::cudaDevAttrComputePreemptionSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrCanUseHostPointerForRegisteredMem> //
    : __dev_attr_impl<::cudaDevAttrCanUseHostPointerForRegisteredMem, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrCooperativeLaunch> //
    : __dev_attr_impl<::cudaDevAttrCooperativeLaunch, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrCanFlushRemoteWrites> //
    : __dev_attr_impl<::cudaDevAttrCanFlushRemoteWrites, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrHostRegisterSupported> //
    : __dev_attr_impl<::cudaDevAttrHostRegisterSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrDirectManagedMemAccessFromHost> //
    : __dev_attr_impl<::cudaDevAttrDirectManagedMemAccessFromHost, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrSparseCudaArraySupported> //
    : __dev_attr_impl<::cudaDevAttrSparseCudaArraySupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrMemoryPoolsSupported> //
    : __dev_attr_impl<::cudaDevAttrMemoryPoolsSupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrGPUDirectRDMASupported> //
    : __dev_attr_impl<::cudaDevAttrGPUDirectRDMASupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrDeferredMappingCudaArraySupported> //
    : __dev_attr_impl<::cudaDevAttrDeferredMappingCudaArraySupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrIpcEventSupport> //
    : __dev_attr_impl<::cudaDevAttrIpcEventSupport, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrPageableMemoryAccessUsesHostPageTables>
    : __dev_attr_impl<::cudaDevAttrPageableMemoryAccessUsesHostPageTables, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrHostRegisterReadOnlySupported> //
    : __dev_attr_impl<::cudaDevAttrHostRegisterReadOnlySupported, bool>
{};
template <>
struct __dev_attr<::cudaDevAttrGPUDirectRDMAFlushWritesOptions> //
    : __dev_attr_impl<::cudaDevAttrGPUDirectRDMAFlushWritesOptions, ::cudaFlushGPUDirectRDMAWritesOptions>
{
  static constexpr type host    = ::cudaFlushGPUDirectRDMAWritesOptionHost;
  static constexpr type mem_ops = ::cudaFlushGPUDirectRDMAWritesOptionMemOps;
};
template <>
struct __dev_attr<::cudaDevAttrGPUDirectRDMAWritesOrdering> //
    : __dev_attr_impl<::cudaDevAttrGPUDirectRDMAWritesOrdering, ::cudaGPUDirectRDMAWritesOrdering>
{
  static constexpr type none        = ::cudaGPUDirectRDMAWritesOrderingNone;
  static constexpr type owner       = ::cudaGPUDirectRDMAWritesOrderingOwner;
  static constexpr type all_devices = ::cudaGPUDirectRDMAWritesOrderingAllDevices;
};
template <>
struct __dev_attr<::cudaDevAttrMemoryPoolSupportedHandleTypes> //
    : __dev_attr_impl<::cudaDevAttrMemoryPoolSupportedHandleTypes, ::cudaMemAllocationHandleType>
{
  static constexpr type none                  = ::cudaMemHandleTypeNone;
  static constexpr type posix_file_descriptor = ::cudaMemHandleTypePosixFileDescriptor;
  static constexpr type win32                 = ::cudaMemHandleTypeWin32;
  static constexpr type win32_kmt             = ::cudaMemHandleTypeWin32Kmt;
#  if _CCCL_CTK_AT_LEAST(12, 4)
  static constexpr type fabric = ::cudaMemHandleTypeFabric;
#  else // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^ / vvv _CCCL_CTK_BELOW(12, 4) vvv
  static inline const type fabric = static_cast<::cudaMemAllocationHandleType>(0x8);
#  endif // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^
};
#  if _CCCL_CTK_AT_LEAST(12, 2)
template <>
struct __dev_attr<::cudaDevAttrNumaConfig> //
    : __dev_attr_impl<::cudaDevAttrNumaConfig, ::cudaDeviceNumaConfig>
{
  static constexpr type none      = ::cudaDeviceNumaConfigNone;
  static constexpr type numa_node = ::cudaDeviceNumaConfigNumaNode;
};
#  endif // _CCCL_CTK_AT_LEAST(12, 2)

} // namespace __detail

namespace device_attributes
{
// Maximum number of threads per block
using max_threads_per_block_t = __detail::__dev_attr<::cudaDevAttrMaxThreadsPerBlock>;
static constexpr max_threads_per_block_t max_threads_per_block{};

// Maximum x-dimension of a block
using max_block_dim_x_t = __detail::__dev_attr<::cudaDevAttrMaxBlockDimX>;
static constexpr max_block_dim_x_t max_block_dim_x{};

// Maximum y-dimension of a block
using max_block_dim_y_t = __detail::__dev_attr<::cudaDevAttrMaxBlockDimY>;
static constexpr max_block_dim_y_t max_block_dim_y{};

// Maximum z-dimension of a block
using max_block_dim_z_t = __detail::__dev_attr<::cudaDevAttrMaxBlockDimZ>;
static constexpr max_block_dim_z_t max_block_dim_z{};

// Maximum x-dimension of a grid
using max_grid_dim_x_t = __detail::__dev_attr<::cudaDevAttrMaxGridDimX>;
static constexpr max_grid_dim_x_t max_grid_dim_x{};

// Maximum y-dimension of a grid
using max_grid_dim_y_t = __detail::__dev_attr<::cudaDevAttrMaxGridDimY>;
static constexpr max_grid_dim_y_t max_grid_dim_y{};

// Maximum z-dimension of a grid
using max_grid_dim_z_t = __detail::__dev_attr<::cudaDevAttrMaxGridDimZ>;
static constexpr max_grid_dim_z_t max_grid_dim_z{};

// Maximum amount of shared memory available to a thread block in bytes
using max_shared_memory_per_block_t = __detail::__dev_attr<::cudaDevAttrMaxSharedMemoryPerBlock>;
static constexpr max_shared_memory_per_block_t max_shared_memory_per_block{};

// Memory available on device for __constant__ variables in a CUDA C kernel in bytes
using total_constant_memory_t = __detail::__dev_attr<::cudaDevAttrTotalConstantMemory>;
static constexpr total_constant_memory_t total_constant_memory{};

// Warp size in threads
using warp_size_t = __detail::__dev_attr<::cudaDevAttrWarpSize>;
static constexpr warp_size_t warp_size{};

// Maximum pitch in bytes allowed by the memory copy functions that involve
// memory regions allocated through cudaMallocPitch()
using max_pitch_t = __detail::__dev_attr<::cudaDevAttrMaxPitch>;
static constexpr max_pitch_t max_pitch{};

// Maximum 1D texture width
using max_texture_1d_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture1DWidth>;
static constexpr max_texture_1d_width_t max_texture_1d_width{};

// Maximum width for a 1D texture bound to linear memory
using max_texture_1d_linear_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture1DLinearWidth>;
static constexpr max_texture_1d_linear_width_t max_texture_1d_linear_width{};

// Maximum mipmapped 1D texture width
using max_texture_1d_mipmapped_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture1DMipmappedWidth>;
static constexpr max_texture_1d_mipmapped_width_t max_texture_1d_mipmapped_width{};

// Maximum 2D texture width
using max_texture_2d_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DWidth>;
static constexpr max_texture_2d_width_t max_texture_2d_width{};

// Maximum 2D texture height
using max_texture_2d_height_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DHeight>;
static constexpr max_texture_2d_height_t max_texture_2d_height{};

// Maximum width for a 2D texture bound to linear memory
using max_texture_2d_linear_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DLinearWidth>;
static constexpr max_texture_2d_linear_width_t max_texture_2d_linear_width{};

// Maximum height for a 2D texture bound to linear memory
using max_texture_2d_linear_height_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DLinearHeight>;
static constexpr max_texture_2d_linear_height_t max_texture_2d_linear_height{};

// Maximum pitch in bytes for a 2D texture bound to linear memory
using max_texture_2d_linear_pitch_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DLinearPitch>;
static constexpr max_texture_2d_linear_pitch_t max_texture_2d_linear_pitch{};

// Maximum mipmapped 2D texture width
using max_texture_2d_mipmapped_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DMipmappedWidth>;
static constexpr max_texture_2d_mipmapped_width_t max_texture_2d_mipmapped_width{};

// Maximum mipmapped 2D texture height
using max_texture_2d_mipmapped_height_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DMipmappedHeight>;
static constexpr max_texture_2d_mipmapped_height_t max_texture_2d_mipmapped_height{};

// Maximum 3D texture width
using max_texture_3d_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture3DWidth>;
static constexpr max_texture_3d_width_t max_texture_3d_width{};

// Maximum 3D texture height
using max_texture_3d_height_t = __detail::__dev_attr<::cudaDevAttrMaxTexture3DHeight>;
static constexpr max_texture_3d_height_t max_texture_3d_height{};

// Maximum 3D texture depth
using max_texture_3d_depth_t = __detail::__dev_attr<::cudaDevAttrMaxTexture3DDepth>;
static constexpr max_texture_3d_depth_t max_texture_3d_depth{};

// Alternate maximum 3D texture width, 0 if no alternate maximum 3D texture size is supported
using max_texture_3d_width_alt_t = __detail::__dev_attr<::cudaDevAttrMaxTexture3DWidthAlt>;
static constexpr max_texture_3d_width_alt_t max_texture_3d_width_alt{};

// Alternate maximum 3D texture height, 0 if no alternate maximum 3D texture size is supported
using max_texture_3d_height_alt_t = __detail::__dev_attr<::cudaDevAttrMaxTexture3DHeightAlt>;
static constexpr max_texture_3d_height_alt_t max_texture_3d_height_alt{};

// Alternate maximum 3D texture depth, 0 if no alternate maximum 3D texture size is supported
using max_texture_3d_depth_alt_t = __detail::__dev_attr<::cudaDevAttrMaxTexture3DDepthAlt>;
static constexpr max_texture_3d_depth_alt_t max_texture_3d_depth_alt{};

// Maximum cubemap texture width or height
using max_texture_cubemap_width_t = __detail::__dev_attr<::cudaDevAttrMaxTextureCubemapWidth>;
static constexpr max_texture_cubemap_width_t max_texture_cubemap_width{};

// Maximum 1D layered texture width
using max_texture_1d_layered_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture1DLayeredWidth>;
static constexpr max_texture_1d_layered_width_t max_texture_1d_layered_width{};

// Maximum layers in a 1D layered texture
using max_texture_1d_layered_layers_t = __detail::__dev_attr<::cudaDevAttrMaxTexture1DLayeredLayers>;
static constexpr max_texture_1d_layered_layers_t max_texture_1d_layered_layers{};

// Maximum 2D layered texture width
using max_texture_2d_layered_width_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DLayeredWidth>;
static constexpr max_texture_2d_layered_width_t max_texture_2d_layered_width{};

// Maximum 2D layered texture height
using max_texture_2d_layered_height_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DLayeredHeight>;
static constexpr max_texture_2d_layered_height_t max_texture_2d_layered_height{};

// Maximum layers in a 2D layered texture
using max_texture_2d_layered_layers_t = __detail::__dev_attr<::cudaDevAttrMaxTexture2DLayeredLayers>;
static constexpr max_texture_2d_layered_layers_t max_texture_2d_layered_layers{};

// Maximum cubemap layered texture width or height
using max_texture_cubemap_layered_width_t = __detail::__dev_attr<::cudaDevAttrMaxTextureCubemapLayeredWidth>;
static constexpr max_texture_cubemap_layered_width_t max_texture_cubemap_layered_width{};

// Maximum layers in a cubemap layered texture
using max_texture_cubemap_layered_layers_t = __detail::__dev_attr<::cudaDevAttrMaxTextureCubemapLayeredLayers>;
static constexpr max_texture_cubemap_layered_layers_t max_texture_cubemap_layered_layers{};

// Maximum 1D surface width
using max_surface_1d_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurface1DWidth>;
static constexpr max_surface_1d_width_t max_surface_1d_width{};

// Maximum 2D surface width
using max_surface_2d_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurface2DWidth>;
static constexpr max_surface_2d_width_t max_surface_2d_width{};

// Maximum 2D surface height
using max_surface_2d_height_t = __detail::__dev_attr<::cudaDevAttrMaxSurface2DHeight>;
static constexpr max_surface_2d_height_t max_surface_2d_height{};

// Maximum 3D surface width
using max_surface_3d_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurface3DWidth>;
static constexpr max_surface_3d_width_t max_surface_3d_width{};

// Maximum 3D surface height
using max_surface_3d_height_t = __detail::__dev_attr<::cudaDevAttrMaxSurface3DHeight>;
static constexpr max_surface_3d_height_t max_surface_3d_height{};

// Maximum 3D surface depth
using max_surface_3d_depth_t = __detail::__dev_attr<::cudaDevAttrMaxSurface3DDepth>;
static constexpr max_surface_3d_depth_t max_surface_3d_depth{};

// Maximum 1D layered surface width
using max_surface_1d_layered_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurface1DLayeredWidth>;
static constexpr max_surface_1d_layered_width_t max_surface_1d_layered_width{};

// Maximum layers in a 1D layered surface
using max_surface_1d_layered_layers_t = __detail::__dev_attr<::cudaDevAttrMaxSurface1DLayeredLayers>;
static constexpr max_surface_1d_layered_layers_t max_surface_1d_layered_layers{};

// Maximum 2D layered surface width
using max_surface_2d_layered_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurface2DLayeredWidth>;
static constexpr max_surface_2d_layered_width_t max_surface_2d_layered_width{};

// Maximum 2D layered surface height
using max_surface_2d_layered_height_t = __detail::__dev_attr<::cudaDevAttrMaxSurface2DLayeredHeight>;
static constexpr max_surface_2d_layered_height_t max_surface_2d_layered_height{};

// Maximum layers in a 2D layered surface
using max_surface_2d_layered_layers_t = __detail::__dev_attr<::cudaDevAttrMaxSurface2DLayeredLayers>;
static constexpr max_surface_2d_layered_layers_t max_surface_2d_layered_layers{};

// Maximum cubemap surface width
using max_surface_cubemap_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurfaceCubemapWidth>;
static constexpr max_surface_cubemap_width_t max_surface_cubemap_width{};

// Maximum cubemap layered surface width
using max_surface_cubemap_layered_width_t = __detail::__dev_attr<::cudaDevAttrMaxSurfaceCubemapLayeredWidth>;
static constexpr max_surface_cubemap_layered_width_t max_surface_cubemap_layered_width{};

// Maximum layers in a cubemap layered surface
using max_surface_cubemap_layered_layers_t = __detail::__dev_attr<::cudaDevAttrMaxSurfaceCubemapLayeredLayers>;
static constexpr max_surface_cubemap_layered_layers_t max_surface_cubemap_layered_layers{};

// Maximum number of 32-bit registers available to a thread block
using max_registers_per_block_t = __detail::__dev_attr<::cudaDevAttrMaxRegistersPerBlock>;
static constexpr max_registers_per_block_t max_registers_per_block{};

// Peak clock frequency in kilohertz
using clock_rate_t = __detail::__dev_attr<::cudaDevAttrClockRate>;
static constexpr clock_rate_t clock_rate{};

// Alignment requirement; texture base addresses aligned to textureAlign bytes
// do not need an offset applied to texture fetches
using texture_alignment_t = __detail::__dev_attr<::cudaDevAttrTextureAlignment>;
static constexpr texture_alignment_t texture_alignment{};

// Pitch alignment requirement for 2D texture references bound to pitched memory
using texture_pitch_alignment_t = __detail::__dev_attr<::cudaDevAttrTexturePitchAlignment>;
static constexpr texture_pitch_alignment_t texture_pitch_alignment{};

// true if the device can concurrently copy memory between host and device
// while executing a kernel, or false if not
using gpu_overlap_t = __detail::__dev_attr<::cudaDevAttrGpuOverlap>;
static constexpr gpu_overlap_t gpu_overlap{};

// Number of multiprocessors on the device
using multiprocessor_count_t = __detail::__dev_attr<::cudaDevAttrMultiProcessorCount>;
static constexpr multiprocessor_count_t multiprocessor_count{};

// true if there is a run time limit for kernels executed on the device, or
// false if not
using kernel_exec_timeout_t = __detail::__dev_attr<::cudaDevAttrKernelExecTimeout>;
static constexpr kernel_exec_timeout_t kernel_exec_timeout{};

// true if the device is integrated with the memory subsystem, or false if not
using integrated_t = __detail::__dev_attr<::cudaDevAttrIntegrated>;
static constexpr integrated_t integrated{};

// true if the device can map host memory into CUDA address space
using can_map_host_memory_t = __detail::__dev_attr<::cudaDevAttrCanMapHostMemory>;
static constexpr can_map_host_memory_t can_map_host_memory{};

// Compute mode is the compute mode that the device is currently in.
using compute_mode_t = __detail::__dev_attr<::cudaDevAttrComputeMode>;
static constexpr compute_mode_t compute_mode{};

// true if the device supports executing multiple kernels within the same
// context simultaneously, or false if not. It is not guaranteed that multiple
// kernels will be resident on the device concurrently so this feature should
// not be relied upon for correctness.
using concurrent_kernels_t = __detail::__dev_attr<::cudaDevAttrConcurrentKernels>;
static constexpr concurrent_kernels_t concurrent_kernels{};

// true if error correction is enabled on the device, 0 if error correction is
// disabled or not supported by the device
using ecc_enabled_t = __detail::__dev_attr<::cudaDevAttrEccEnabled>;
static constexpr ecc_enabled_t ecc_enabled{};

// PCI bus identifier of the device
using pci_bus_id_t = __detail::__dev_attr<::cudaDevAttrPciBusId>;
static constexpr pci_bus_id_t pci_bus_id{};

// PCI device (also known as slot) identifier of the device
using pci_device_id_t = __detail::__dev_attr<::cudaDevAttrPciDeviceId>;
static constexpr pci_device_id_t pci_device_id{};

// true if the device is using a TCC driver. TCC is only available on Tesla
// hardware running Windows Vista or later.
using tcc_driver_t = __detail::__dev_attr<::cudaDevAttrTccDriver>;
static constexpr tcc_driver_t tcc_driver{};

// Peak memory clock frequency in kilohertz
using memory_clock_rate_t = __detail::__dev_attr<::cudaDevAttrMemoryClockRate>;
static constexpr memory_clock_rate_t memory_clock_rate{};

// Global memory bus width in bits
using global_memory_bus_width_t = __detail::__dev_attr<::cudaDevAttrGlobalMemoryBusWidth>;
static constexpr global_memory_bus_width_t global_memory_bus_width{};

// Size of L2 cache in bytes. 0 if the device doesn't have L2 cache.
using l2_cache_size_t = __detail::__dev_attr<::cudaDevAttrL2CacheSize>;
static constexpr l2_cache_size_t l2_cache_size{};

// Maximum resident threads per multiprocessor
using max_threads_per_multiprocessor_t = __detail::__dev_attr<::cudaDevAttrMaxThreadsPerMultiProcessor>;
static constexpr max_threads_per_multiprocessor_t max_threads_per_multiprocessor{};

// true if the device shares a unified address space with the host, or false
// if not
using unified_addressing_t = __detail::__dev_attr<::cudaDevAttrUnifiedAddressing>;
static constexpr unified_addressing_t unified_addressing{};

// Major compute capability version number
using compute_capability_major_t = __detail::__dev_attr<::cudaDevAttrComputeCapabilityMajor>;
static constexpr compute_capability_major_t compute_capability_major{};

// Minor compute capability version number
using compute_capability_minor_t = __detail::__dev_attr<::cudaDevAttrComputeCapabilityMinor>;
static constexpr compute_capability_minor_t compute_capability_minor{};

// true if the device supports stream priorities, or false if not
using stream_priorities_supported_t = __detail::__dev_attr<::cudaDevAttrStreamPrioritiesSupported>;
static constexpr stream_priorities_supported_t stream_priorities_supported{};

// true if device supports caching globals in L1 cache, false if not
using global_l1_cache_supported_t = __detail::__dev_attr<::cudaDevAttrGlobalL1CacheSupported>;
static constexpr global_l1_cache_supported_t global_l1_cache_supported{};

// true if device supports caching locals in L1 cache, false if not
using local_l1_cache_supported_t = __detail::__dev_attr<::cudaDevAttrLocalL1CacheSupported>;
static constexpr local_l1_cache_supported_t local_l1_cache_supported{};

// Maximum amount of shared memory available to a multiprocessor in bytes;
// this amount is shared by all thread blocks simultaneously resident on a
// multiprocessor
using max_shared_memory_per_multiprocessor_t = __detail::__dev_attr<::cudaDevAttrMaxSharedMemoryPerMultiprocessor>;
static constexpr max_shared_memory_per_multiprocessor_t max_shared_memory_per_multiprocessor{};

// Maximum number of 32-bit registers available to a multiprocessor; this
// number is shared by all thread blocks simultaneously resident on a
// multiprocessor
using max_registers_per_multiprocessor_t = __detail::__dev_attr<::cudaDevAttrMaxRegistersPerMultiprocessor>;
static constexpr max_registers_per_multiprocessor_t max_registers_per_multiprocessor{};

// true if device supports allocating managed memory, false if not
using managed_memory_t = __detail::__dev_attr<::cudaDevAttrManagedMemory>;
static constexpr managed_memory_t managed_memory{};

// true if device is on a multi-GPU board, false if not
using is_multi_gpu_board_t = __detail::__dev_attr<::cudaDevAttrIsMultiGpuBoard>;
static constexpr is_multi_gpu_board_t is_multi_gpu_board{};

// Unique identifier for a group of devices on the same multi-GPU board
using multi_gpu_board_group_id_t = __detail::__dev_attr<::cudaDevAttrMultiGpuBoardGroupID>;
static constexpr multi_gpu_board_group_id_t multi_gpu_board_group_id{};

// true if the link between the device and the host supports native atomic
// operations
using host_native_atomic_supported_t = __detail::__dev_attr<::cudaDevAttrHostNativeAtomicSupported>;
static constexpr host_native_atomic_supported_t host_native_atomic_supported{};

// Ratio of single precision performance (in floating-point operations per
// second) to double precision performance
using single_to_double_precision_perf_ratio_t = __detail::__dev_attr<::cudaDevAttrSingleToDoublePrecisionPerfRatio>;
static constexpr single_to_double_precision_perf_ratio_t single_to_double_precision_perf_ratio{};

// true if the device supports coherently accessing pageable memory without
// calling cudaHostRegister on it, and false otherwise
using pageable_memory_access_t = __detail::__dev_attr<::cudaDevAttrPageableMemoryAccess>;
static constexpr pageable_memory_access_t pageable_memory_access{};

// true if the device can coherently access managed memory concurrently with
// the CPU, and false otherwise
using concurrent_managed_access_t = __detail::__dev_attr<::cudaDevAttrConcurrentManagedAccess>;
static constexpr concurrent_managed_access_t concurrent_managed_access{};

// true if the device supports Compute Preemption, false if not
using compute_preemption_supported_t = __detail::__dev_attr<::cudaDevAttrComputePreemptionSupported>;
static constexpr compute_preemption_supported_t compute_preemption_supported{};

// true if the device can access host registered memory at the same virtual
// address as the CPU, and false otherwise
using can_use_host_pointer_for_registered_mem_t = __detail::__dev_attr<::cudaDevAttrCanUseHostPointerForRegisteredMem>;
static constexpr can_use_host_pointer_for_registered_mem_t can_use_host_pointer_for_registered_mem{};

// true if the device supports launching cooperative kernels via
// cudaLaunchCooperativeKernel, and false otherwise
using cooperative_launch_t = __detail::__dev_attr<::cudaDevAttrCooperativeLaunch>;
static constexpr cooperative_launch_t cooperative_launch{};

// true if the device supports flushing of outstanding remote writes, and
// false otherwise
using can_flush_remote_writes_t = __detail::__dev_attr<::cudaDevAttrCanFlushRemoteWrites>;
static constexpr can_flush_remote_writes_t can_flush_remote_writes{};

// true if the device supports host memory registration via cudaHostRegister,
// and false otherwise
using host_register_supported_t = __detail::__dev_attr<::cudaDevAttrHostRegisterSupported>;
static constexpr host_register_supported_t host_register_supported{};

// true if the device accesses pageable memory via the host's page tables, and
// false otherwise
using pageable_memory_access_uses_host_page_tables_t =
  __detail::__dev_attr<::cudaDevAttrPageableMemoryAccessUsesHostPageTables>;
static constexpr pageable_memory_access_uses_host_page_tables_t pageable_memory_access_uses_host_page_tables{};

// true if the host can directly access managed memory on the device without
// migration, and false otherwise
using direct_managed_mem_access_from_host_t = __detail::__dev_attr<::cudaDevAttrDirectManagedMemAccessFromHost>;
static constexpr direct_managed_mem_access_from_host_t direct_managed_mem_access_from_host{};

// Maximum per block shared memory size on the device. This value can be opted
// into when using dynamic_shared_memory with NonPortableSize set to true
using max_shared_memory_per_block_optin_t = __detail::__dev_attr<::cudaDevAttrMaxSharedMemoryPerBlockOptin>;
static constexpr max_shared_memory_per_block_optin_t max_shared_memory_per_block_optin{};

// Maximum number of thread blocks that can reside on a multiprocessor
using max_blocks_per_multiprocessor_t = __detail::__dev_attr<::cudaDevAttrMaxBlocksPerMultiprocessor>;
static constexpr max_blocks_per_multiprocessor_t max_blocks_per_multiprocessor{};

// Maximum L2 persisting lines capacity setting in bytes
using max_persisting_l2_cache_size_t = __detail::__dev_attr<::cudaDevAttrMaxPersistingL2CacheSize>;
static constexpr max_persisting_l2_cache_size_t max_persisting_l2_cache_size{};

// Maximum value of cudaAccessPolicyWindow::num_bytes
using max_access_policy_window_size_t = __detail::__dev_attr<::cudaDevAttrMaxAccessPolicyWindowSize>;
static constexpr max_access_policy_window_size_t max_access_policy_window_size{};

// Shared memory reserved by CUDA driver per block in bytes
using reserved_shared_memory_per_block_t = __detail::__dev_attr<::cudaDevAttrReservedSharedMemoryPerBlock>;
static constexpr reserved_shared_memory_per_block_t reserved_shared_memory_per_block{};

// true if the device supports sparse CUDA arrays and sparse CUDA mipmapped arrays.
using sparse_cuda_array_supported_t = __detail::__dev_attr<::cudaDevAttrSparseCudaArraySupported>;
static constexpr sparse_cuda_array_supported_t sparse_cuda_array_supported{};

// Device supports using the cudaHostRegister flag cudaHostRegisterReadOnly to
// register memory that must be mapped as read-only to the GPU
using host_register_read_only_supported_t = __detail::__dev_attr<::cudaDevAttrHostRegisterReadOnlySupported>;
static constexpr host_register_read_only_supported_t host_register_read_only_supported{};

// true if the device supports using the cudaMallocAsync and cudaMemPool
// family of APIs, and false otherwise
using memory_pools_supported_t = __detail::__dev_attr<::cudaDevAttrMemoryPoolsSupported>;
static constexpr memory_pools_supported_t memory_pools_supported{};

// true if the device supports GPUDirect RDMA APIs, and false otherwise
using gpu_direct_rdma_supported_t = __detail::__dev_attr<::cudaDevAttrGPUDirectRDMASupported>;
static constexpr gpu_direct_rdma_supported_t gpu_direct_rdma_supported{};

// bitmask to be interpreted according to the
// cudaFlushGPUDirectRDMAWritesOptions enum
using gpu_direct_rdma_flush_writes_options_t = __detail::__dev_attr<::cudaDevAttrGPUDirectRDMAFlushWritesOptions>;
static constexpr gpu_direct_rdma_flush_writes_options_t gpu_direct_rdma_flush_writes_options{};

// see the cudaGPUDirectRDMAWritesOrdering enum for numerical values
using gpu_direct_rdma_writes_ordering_t = __detail::__dev_attr<::cudaDevAttrGPUDirectRDMAWritesOrdering>;
static constexpr gpu_direct_rdma_writes_ordering_t gpu_direct_rdma_writes_ordering{};

// Bitmask of handle types supported with mempool based IPC
using memory_pool_supported_handle_types_t = __detail::__dev_attr<::cudaDevAttrMemoryPoolSupportedHandleTypes>;
static constexpr memory_pool_supported_handle_types_t memory_pool_supported_handle_types{};

// true if the device supports deferred mapping CUDA arrays and CUDA mipmapped
// arrays.
using deferred_mapping_cuda_array_supported_t = __detail::__dev_attr<::cudaDevAttrDeferredMappingCudaArraySupported>;
static constexpr deferred_mapping_cuda_array_supported_t deferred_mapping_cuda_array_supported{};

// true if the device supports IPC Events, false otherwise.
using ipc_event_support_t = __detail::__dev_attr<::cudaDevAttrIpcEventSupport>;
static constexpr ipc_event_support_t ipc_event_support{};

#  if _CCCL_CTK_AT_LEAST(12, 2)
// NUMA configuration of a device: value is of type cudaDeviceNumaConfig enum
using numa_config_t = __detail::__dev_attr<::cudaDevAttrNumaConfig>;
static constexpr numa_config_t numa_config{};

// NUMA node ID of the GPU memory
using numa_id_t = __detail::__dev_attr<::cudaDevAttrNumaId>;
static constexpr numa_id_t numa_id{};
#  endif // _CCCL_CTK_AT_LEAST(12, 2)

// Combines major and minor compute capability in a 100 * major + 10 * minor format, allows to query full compute
// capability in a single query
struct compute_capability_t
{
  [[nodiscard]] int operator()(device_ref __dev_id) const
  {
    return 10 * ::cuda::device_attributes::compute_capability_major(__dev_id)
         + ::cuda::device_attributes::compute_capability_minor(__dev_id);
  }
};
static constexpr compute_capability_t compute_capability{};
} // namespace device_attributes

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DEVICE_ATTRIBUTES_H
