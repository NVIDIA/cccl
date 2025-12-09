//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// WANTS_CUDADEVRT.

#include <cuda/devices>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_macros.h"

template <cudaDeviceAttr ExpAttr, class ExpType, class Attr>
__host__ __device__ void test_attribute(const Attr& attr, const cuda::device_ref device)
{
  // 1. Test Attr::type.
  static_assert(cuda::std::is_same_v<ExpType, typename Attr::type>);

  // 2. Test Attr::operator cudaDeviceAttr().
  static_assert(cuda::std::is_same_v<cudaDeviceAttr, decltype(attr.operator cudaDeviceAttr())>);
  static_assert(noexcept(attr.operator cudaDeviceAttr()));
  static_assert(ExpAttr == Attr{}.operator cudaDeviceAttr());

  // 3. Test Attr::operator()(device_ref)
  static_assert(cuda::std::is_same_v<ExpType, decltype(attr(device))>);
  static_assert(!noexcept(attr(device)));
  [[maybe_unused]] const auto result = attr(device);

  // 4. Test cuda::device_attribute_result_t<ExpAttr>
  static_assert(cuda::std::is_same_v<cuda::device_attribute_result_t<ExpAttr>, ExpType>);
}

__host__ __device__ cuda::device_ref test_device()
{
  NV_IF_ELSE_TARGET(NV_IS_HOST, (return cuda::devices[0];), (return cuda::device::current_device();))
}

__host__ __device__ void test()
{
  namespace attrs = cuda::device_attributes;

  const auto device = test_device();

  // clang-format off
  test_attribute<cudaDevAttrMaxThreadsPerBlock, int>(attrs::max_threads_per_block, device);
  test_attribute<cudaDevAttrMaxBlockDimX, int>(attrs::max_block_dim_x, device);
  test_attribute<cudaDevAttrMaxBlockDimY, int>(attrs::max_block_dim_y, device);
  test_attribute<cudaDevAttrMaxBlockDimZ, int>(attrs::max_block_dim_z, device);
  test_attribute<cudaDevAttrMaxGridDimX, int>(attrs::max_grid_dim_x, device);
  test_attribute<cudaDevAttrMaxGridDimY, int>(attrs::max_grid_dim_y, device);
  test_attribute<cudaDevAttrMaxGridDimZ, int>(attrs::max_grid_dim_z, device);
  test_attribute<cudaDevAttrMaxSharedMemoryPerBlock, cuda::std::size_t>(attrs::max_shared_memory_per_block, device);
  test_attribute<cudaDevAttrTotalConstantMemory, cuda::std::size_t>(attrs::total_constant_memory, device);
  test_attribute<cudaDevAttrWarpSize, int>(attrs::warp_size, device);
  test_attribute<cudaDevAttrMaxPitch, cuda::std::size_t>(attrs::max_pitch, device);
  test_attribute<cudaDevAttrMaxTexture1DWidth, int>(attrs::max_texture_1d_width, device);
  test_attribute<cudaDevAttrMaxTexture1DLinearWidth, int>(attrs::max_texture_1d_linear_width, device);
  test_attribute<cudaDevAttrMaxTexture1DMipmappedWidth, int>(attrs::max_texture_1d_mipmapped_width, device);
  test_attribute<cudaDevAttrMaxTexture2DWidth, int>(attrs::max_texture_2d_width, device);
  test_attribute<cudaDevAttrMaxTexture2DHeight, int>(attrs::max_texture_2d_height, device);
  test_attribute<cudaDevAttrMaxTexture2DLinearWidth, int>(attrs::max_texture_2d_linear_width, device);
  test_attribute<cudaDevAttrMaxTexture2DLinearHeight, int>(attrs::max_texture_2d_linear_height, device);
  test_attribute<cudaDevAttrMaxTexture2DLinearPitch, cuda::std::size_t>(attrs::max_texture_2d_linear_pitch, device);
  test_attribute<cudaDevAttrMaxTexture2DMipmappedWidth, int>(attrs::max_texture_2d_mipmapped_width, device);
  test_attribute<cudaDevAttrMaxTexture2DMipmappedHeight, int>(attrs::max_texture_2d_mipmapped_height, device);
  test_attribute<cudaDevAttrMaxTexture3DWidth, int>(attrs::max_texture_3d_width, device);
  test_attribute<cudaDevAttrMaxTexture3DHeight, int>(attrs::max_texture_3d_height, device);
  test_attribute<cudaDevAttrMaxTexture3DDepth, int>(attrs::max_texture_3d_depth, device);
  test_attribute<cudaDevAttrMaxTexture3DWidthAlt, int>(attrs::max_texture_3d_width_alt, device);
  test_attribute<cudaDevAttrMaxTexture3DHeightAlt, int>(attrs::max_texture_3d_height_alt, device);
  test_attribute<cudaDevAttrMaxTexture3DDepthAlt, int>(attrs::max_texture_3d_depth_alt, device);
  test_attribute<cudaDevAttrMaxTextureCubemapWidth, int>(attrs::max_texture_cubemap_width, device);
  test_attribute<cudaDevAttrMaxTexture1DLayeredWidth, int>(attrs::max_texture_1d_layered_width, device);
  test_attribute<cudaDevAttrMaxTexture1DLayeredLayers, int>(attrs::max_texture_1d_layered_layers, device);
  test_attribute<cudaDevAttrMaxTexture2DLayeredWidth, int>(attrs::max_texture_2d_layered_width, device);
  test_attribute<cudaDevAttrMaxTexture2DLayeredHeight, int>(attrs::max_texture_2d_layered_height, device);
  test_attribute<cudaDevAttrMaxTexture2DLayeredLayers, int>(attrs::max_texture_2d_layered_layers, device);
  test_attribute<cudaDevAttrMaxTextureCubemapLayeredWidth, int>(attrs::max_texture_cubemap_layered_width, device);
  test_attribute<cudaDevAttrMaxTextureCubemapLayeredLayers, int>(attrs::max_texture_cubemap_layered_layers, device);
  test_attribute<cudaDevAttrMaxSurface1DWidth, int>(attrs::max_surface_1d_width, device);
  test_attribute<cudaDevAttrMaxSurface2DWidth, int>(attrs::max_surface_2d_width, device);
  test_attribute<cudaDevAttrMaxSurface2DHeight, int>(attrs::max_surface_2d_height, device);
  test_attribute<cudaDevAttrMaxSurface3DWidth, int>(attrs::max_surface_3d_width, device);
  test_attribute<cudaDevAttrMaxSurface3DHeight, int>(attrs::max_surface_3d_height, device);
  test_attribute<cudaDevAttrMaxSurface3DDepth, int>(attrs::max_surface_3d_depth, device);
  test_attribute<cudaDevAttrMaxSurface1DLayeredWidth, int>(attrs::max_surface_1d_layered_width, device);
  test_attribute<cudaDevAttrMaxSurface1DLayeredLayers, int>(attrs::max_surface_1d_layered_layers, device);
  test_attribute<cudaDevAttrMaxSurface2DLayeredWidth, int>(attrs::max_surface_2d_layered_width, device);
  test_attribute<cudaDevAttrMaxSurface2DLayeredHeight, int>(attrs::max_surface_2d_layered_height, device);
  test_attribute<cudaDevAttrMaxSurface2DLayeredLayers, int>(attrs::max_surface_2d_layered_layers, device);
  test_attribute<cudaDevAttrMaxSurfaceCubemapWidth, int>(attrs::max_surface_cubemap_width, device);
  test_attribute<cudaDevAttrMaxSurfaceCubemapLayeredWidth, int>(attrs::max_surface_cubemap_layered_width, device);
  test_attribute<cudaDevAttrMaxSurfaceCubemapLayeredLayers, int>(attrs::max_surface_cubemap_layered_layers, device);
  test_attribute<cudaDevAttrMaxRegistersPerBlock, int>(attrs::max_registers_per_block, device);
  test_attribute<cudaDevAttrClockRate, int>(attrs::clock_rate, device);
  test_attribute<cudaDevAttrTextureAlignment, cuda::std::size_t>(attrs::texture_alignment, device);
  test_attribute<cudaDevAttrTexturePitchAlignment, cuda::std::size_t>(attrs::texture_pitch_alignment, device);
  test_attribute<cudaDevAttrGpuOverlap, bool>(attrs::gpu_overlap, device);
  test_attribute<cudaDevAttrMultiProcessorCount, int>(attrs::multiprocessor_count, device);
  test_attribute<cudaDevAttrKernelExecTimeout, bool>(attrs::kernel_exec_timeout, device);
  test_attribute<cudaDevAttrIntegrated, bool>(attrs::integrated, device);
  test_attribute<cudaDevAttrCanMapHostMemory, bool>(attrs::can_map_host_memory, device);
  test_attribute<cudaDevAttrComputeMode, cudaComputeMode>(attrs::compute_mode, device);
  test_attribute<cudaDevAttrConcurrentKernels, bool>(attrs::concurrent_kernels, device);
  test_attribute<cudaDevAttrEccEnabled, bool>(attrs::ecc_enabled, device);
  test_attribute<cudaDevAttrPciBusId, int>(attrs::pci_bus_id, device);
  test_attribute<cudaDevAttrPciDeviceId, int>(attrs::pci_device_id, device);
  test_attribute<cudaDevAttrTccDriver, bool>(attrs::tcc_driver, device);
  test_attribute<cudaDevAttrL2CacheSize, cuda::std::size_t>(attrs::l2_cache_size, device);
  test_attribute<cudaDevAttrMaxThreadsPerMultiProcessor, int>(attrs::max_threads_per_multiprocessor, device);
  test_attribute<cudaDevAttrUnifiedAddressing, bool>(attrs::unified_addressing, device);
  test_attribute<cudaDevAttrComputeCapabilityMajor, int>(attrs::compute_capability_major, device);
  test_attribute<cudaDevAttrComputeCapabilityMinor, int>(attrs::compute_capability_minor, device);
  test_attribute<cudaDevAttrStreamPrioritiesSupported, bool>(attrs::stream_priorities_supported, device);
  test_attribute<cudaDevAttrGlobalL1CacheSupported, bool>(attrs::global_l1_cache_supported, device);
  test_attribute<cudaDevAttrLocalL1CacheSupported, bool>(attrs::local_l1_cache_supported, device);
  test_attribute<cudaDevAttrMaxSharedMemoryPerMultiprocessor, cuda::std::size_t>(attrs::max_shared_memory_per_multiprocessor, device);
  test_attribute<cudaDevAttrMaxRegistersPerMultiprocessor, int>(attrs::max_registers_per_multiprocessor, device);
  test_attribute<cudaDevAttrIsMultiGpuBoard, bool>(attrs::is_multi_gpu_board, device);
  test_attribute<cudaDevAttrMultiGpuBoardGroupID, int>(attrs::multi_gpu_board_group_id, device);
  test_attribute<cudaDevAttrHostNativeAtomicSupported, bool>(attrs::host_native_atomic_supported, device);
  test_attribute<cudaDevAttrSingleToDoublePrecisionPerfRatio, int>(attrs::single_to_double_precision_perf_ratio, device);
  test_attribute<cudaDevAttrPageableMemoryAccess, bool>(attrs::pageable_memory_access, device);
  test_attribute<cudaDevAttrConcurrentManagedAccess, bool>(attrs::concurrent_managed_access, device);
  test_attribute<cudaDevAttrComputePreemptionSupported, bool>(attrs::compute_preemption_supported, device);
  test_attribute<cudaDevAttrCanUseHostPointerForRegisteredMem, bool>(attrs::can_use_host_pointer_for_registered_mem, device);
  test_attribute<cudaDevAttrCooperativeLaunch, bool>(attrs::cooperative_launch, device);
  test_attribute<cudaDevAttrCanFlushRemoteWrites, bool>(attrs::can_flush_remote_writes, device);
  test_attribute<cudaDevAttrHostRegisterSupported, bool>(attrs::host_register_supported, device);
  test_attribute<cudaDevAttrPageableMemoryAccessUsesHostPageTables, bool>(attrs::pageable_memory_access_uses_host_page_tables, device);
  test_attribute<cudaDevAttrDirectManagedMemAccessFromHost, bool>(attrs::direct_managed_mem_access_from_host, device);
  test_attribute<cudaDevAttrMaxSharedMemoryPerBlockOptin, cuda::std::size_t>(attrs::max_shared_memory_per_block_optin, device);
  test_attribute<cudaDevAttrMaxBlocksPerMultiprocessor, int>(attrs::max_blocks_per_multiprocessor, device);
  test_attribute<cudaDevAttrMaxPersistingL2CacheSize, cuda::std::size_t>(attrs::max_persisting_l2_cache_size, device);
  test_attribute<cudaDevAttrMaxAccessPolicyWindowSize, cuda::std::size_t>(attrs::max_access_policy_window_size, device);
  test_attribute<cudaDevAttrReservedSharedMemoryPerBlock, cuda::std::size_t>(attrs::reserved_shared_memory_per_block, device);
  test_attribute<cudaDevAttrSparseCudaArraySupported, bool>(attrs::sparse_cuda_array_supported, device);
  test_attribute<cudaDevAttrHostRegisterReadOnlySupported, bool>(attrs::host_register_read_only_supported, device);
  test_attribute<cudaDevAttrMemoryPoolsSupported, bool>(attrs::memory_pools_supported, device);
  test_attribute<cudaDevAttrGPUDirectRDMASupported, bool>(attrs::gpu_direct_rdma_supported, device);
  test_attribute<cudaDevAttrGPUDirectRDMAFlushWritesOptions, cudaFlushGPUDirectRDMAWritesOptions>(attrs::gpu_direct_rdma_flush_writes_options, device);
  test_attribute<cudaDevAttrGPUDirectRDMAWritesOrdering, cudaGPUDirectRDMAWritesOrdering>(attrs::gpu_direct_rdma_writes_ordering, device);
  test_attribute<cudaDevAttrMemoryPoolSupportedHandleTypes, cudaMemAllocationHandleType>(attrs::memory_pool_supported_handle_types, device);
  test_attribute<cudaDevAttrDeferredMappingCudaArraySupported, bool>(attrs::deferred_mapping_cuda_array_supported, device);
  test_attribute<cudaDevAttrIpcEventSupport, bool>(attrs::ipc_event_support, device);
  // clang-format on

#if _CCCL_CTK_AT_LEAST(12, 2)
  test_attribute<cudaDevAttrNumaConfig, cudaDeviceNumaConfig>(attrs::numa_config, device);

  // todo: investigate - fails on device
  NV_IF_TARGET(NV_IS_HOST, (test_attribute<cudaDevAttrNumaId, int>(attrs::numa_id, device);))
#endif // _CCCL_CTK_AT_LEAST(12, 2)

  // compute mode
  {
    static_assert(attrs::compute_mode.default_mode == cudaComputeModeDefault);
    static_assert(attrs::compute_mode.prohibited_mode == cudaComputeModeProhibited);
    static_assert(attrs::compute_mode.exclusive_process_mode == cudaComputeModeExclusiveProcess);

    const auto mode = attrs::compute_mode(device);
    assert(mode == attrs::compute_mode.default_mode || mode == attrs::compute_mode.prohibited_mode
           || mode == attrs::compute_mode.exclusive_process_mode);
  }

  // gpu_direct_rdma_flush_writes_options
  {
    static_assert(attrs::gpu_direct_rdma_flush_writes_options.host == cudaFlushGPUDirectRDMAWritesOptionHost);
    static_assert(attrs::gpu_direct_rdma_flush_writes_options.mem_ops == cudaFlushGPUDirectRDMAWritesOptionMemOps);

    constexpr auto all_options =
      attrs::gpu_direct_rdma_flush_writes_options.host | attrs::gpu_direct_rdma_flush_writes_options.mem_ops;

    const auto options = attrs::gpu_direct_rdma_flush_writes_options(device);
    assert(options >= 0 && options <= all_options);
  }

  // gpu_direct_rdma_writes_ordering
  {
    static_assert(attrs::gpu_direct_rdma_writes_ordering.none == cudaGPUDirectRDMAWritesOrderingNone);
    static_assert(attrs::gpu_direct_rdma_writes_ordering.owner == cudaGPUDirectRDMAWritesOrderingOwner);
    static_assert(attrs::gpu_direct_rdma_writes_ordering.all_devices == cudaGPUDirectRDMAWritesOrderingAllDevices);

    const auto ordering = attrs::gpu_direct_rdma_writes_ordering(device);
    assert(ordering == attrs::gpu_direct_rdma_writes_ordering.none
           || ordering == attrs::gpu_direct_rdma_writes_ordering.owner
           || ordering == attrs::gpu_direct_rdma_writes_ordering.all_devices);
  }

  // memory_pool_supported_handle_types
  {
    static_assert(attrs::memory_pool_supported_handle_types.none == cudaMemHandleTypeNone);
    static_assert(
      attrs::memory_pool_supported_handle_types.posix_file_descriptor == cudaMemHandleTypePosixFileDescriptor);
    static_assert(attrs::memory_pool_supported_handle_types.win32 == cudaMemHandleTypeWin32);
    static_assert(attrs::memory_pool_supported_handle_types.win32_kmt == cudaMemHandleTypeWin32Kmt);
#if _CCCL_CTK_AT_LEAST(12, 4)
    static_assert(cudaMemHandleTypeFabric == 0x8);
    static_assert(attrs::memory_pool_supported_handle_types.fabric == cudaMemHandleTypeFabric);
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^ / vvv _CCCL_CTK_BELOW(12, 4) vvv
    static_assert(attrs::memory_pool_supported_handle_types.fabric == 0x8);
#endif // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^

    constexpr int all_handle_types =
      attrs::memory_pool_supported_handle_types.none | attrs::memory_pool_supported_handle_types.posix_file_descriptor
      | attrs::memory_pool_supported_handle_types.win32 | attrs::memory_pool_supported_handle_types.win32_kmt
      | attrs::memory_pool_supported_handle_types.fabric;

    const auto handle_types = attrs::memory_pool_supported_handle_types(device);
    assert(static_cast<int>(handle_types) <= static_cast<int>(all_handle_types));
  }

#if _CCCL_CTK_AT_LEAST(12, 2)
  // numa_config
  {
    static_assert(attrs::numa_config.none == cudaDeviceNumaConfigNone);
    static_assert(attrs::numa_config.numa_node == cudaDeviceNumaConfigNumaNode);

    const auto config = attrs::numa_config(device);
    assert(config == attrs::numa_config.none || config == attrs::numa_config.numa_node);
  }
#endif // _CCCL_CTK_AT_LEAST(12, 2)

  // compute capability
  {
    const auto cc       = attrs::compute_capability(device);
    const auto cc_major = attrs::compute_capability_major(device);
    const auto cc_minor = attrs::compute_capability_minor(device);
    assert((cc == cuda::compute_capability{cc_major, cc_minor}));
  }
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
#if !defined(TEST_NO_CUDADEVRT)
  NV_IF_TARGET(NV_IS_DEVICE, (test();))
#endif // !TEST_NO_CUDADEVRT
  return 0;
}
