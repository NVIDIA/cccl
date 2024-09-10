//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/device.cuh>

#include "cuda/std/__type_traits/is_same.h"
#include <testing.cuh>

namespace
{
template <const auto& Attr, ::cudaDeviceAttr ExpectedAttr, class ExpectedResult>
[[maybe_unused]] auto test_device_attribute()
{
  cudax::device_ref dev0(0);
  STATIC_REQUIRE(Attr == ExpectedAttr);
  STATIC_REQUIRE(::cuda::std::is_same_v<cudax::device::attr_result_t<Attr>, ExpectedResult>);

  auto result = dev0.attr(Attr);
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(result), ExpectedResult>);
  CUDAX_REQUIRE(result == dev0.attr<ExpectedAttr>());
  CUDAX_REQUIRE(result == Attr(dev0));
  return result;
}
} // namespace

TEST_CASE("Smoke", "[device]")
{
  using cudax::device;
  using cudax::device_ref;

  SECTION("Compare")
  {
    CUDAX_REQUIRE(device_ref{0} == device_ref{0});
    CUDAX_REQUIRE(device_ref{0} == 0);
    CUDAX_REQUIRE(0 == device_ref{0});
    CUDAX_REQUIRE(device_ref{1} != device_ref{0});
    CUDAX_REQUIRE(device_ref{1} != 2);
    CUDAX_REQUIRE(1 != device_ref{2});
  }

  SECTION("Attributes")
  {
    ::test_device_attribute<device::attrs::max_threads_per_block, ::cudaDevAttrMaxThreadsPerBlock, int>();
    ::test_device_attribute<device::attrs::max_block_dim_x, ::cudaDevAttrMaxBlockDimX, int>();
    ::test_device_attribute<device::attrs::max_block_dim_y, ::cudaDevAttrMaxBlockDimY, int>();
    ::test_device_attribute<device::attrs::max_block_dim_z, ::cudaDevAttrMaxBlockDimZ, int>();
    ::test_device_attribute<device::attrs::max_grid_dim_x, ::cudaDevAttrMaxGridDimX, int>();
    ::test_device_attribute<device::attrs::max_grid_dim_y, ::cudaDevAttrMaxGridDimY, int>();
    ::test_device_attribute<device::attrs::max_grid_dim_z, ::cudaDevAttrMaxGridDimZ, int>();
    ::test_device_attribute<device::attrs::max_shared_memory_per_block, ::cudaDevAttrMaxSharedMemoryPerBlock, int>();
    ::test_device_attribute<device::attrs::total_constant_memory, ::cudaDevAttrTotalConstantMemory, int>();
    ::test_device_attribute<device::attrs::warp_size, ::cudaDevAttrWarpSize, int>();
    ::test_device_attribute<device::attrs::max_pitch, ::cudaDevAttrMaxPitch, int>();
    ::test_device_attribute<device::attrs::max_texture_1d_width, ::cudaDevAttrMaxTexture1DWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_1d_linear_width, ::cudaDevAttrMaxTexture1DLinearWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_1d_mipmapped_width, ::cudaDevAttrMaxTexture1DMipmappedWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_width, ::cudaDevAttrMaxTexture2DWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_height, ::cudaDevAttrMaxTexture2DHeight, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_linear_width, ::cudaDevAttrMaxTexture2DLinearWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_linear_height, ::cudaDevAttrMaxTexture2DLinearHeight, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_linear_pitch, ::cudaDevAttrMaxTexture2DLinearPitch, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_mipmapped_width, ::cudaDevAttrMaxTexture2DMipmappedWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_mipmapped_height,
                            ::cudaDevAttrMaxTexture2DMipmappedHeight,
                            int>();
    ::test_device_attribute<device::attrs::max_texture_3d_width, ::cudaDevAttrMaxTexture3DWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_3d_height, ::cudaDevAttrMaxTexture3DHeight, int>();
    ::test_device_attribute<device::attrs::max_texture_3d_depth, ::cudaDevAttrMaxTexture3DDepth, int>();
    ::test_device_attribute<device::attrs::max_texture_3d_width_alt, ::cudaDevAttrMaxTexture3DWidthAlt, int>();
    ::test_device_attribute<device::attrs::max_texture_3d_height_alt, ::cudaDevAttrMaxTexture3DHeightAlt, int>();
    ::test_device_attribute<device::attrs::max_texture_3d_depth_alt, ::cudaDevAttrMaxTexture3DDepthAlt, int>();
    ::test_device_attribute<device::attrs::max_texture_cubemap_width, ::cudaDevAttrMaxTextureCubemapWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_1d_layered_width, ::cudaDevAttrMaxTexture1DLayeredWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_1d_layered_layers, ::cudaDevAttrMaxTexture1DLayeredLayers, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_layered_width, ::cudaDevAttrMaxTexture2DLayeredWidth, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_layered_height, ::cudaDevAttrMaxTexture2DLayeredHeight, int>();
    ::test_device_attribute<device::attrs::max_texture_2d_layered_layers, ::cudaDevAttrMaxTexture2DLayeredLayers, int>();
    ::test_device_attribute<device::attrs::max_texture_cubemap_layered_width,
                            ::cudaDevAttrMaxTextureCubemapLayeredWidth,
                            int>();
    ::test_device_attribute<device::attrs::max_texture_cubemap_layered_layers,
                            ::cudaDevAttrMaxTextureCubemapLayeredLayers,
                            int>();
    ::test_device_attribute<device::attrs::max_surface_1d_width, ::cudaDevAttrMaxSurface1DWidth, int>();
    ::test_device_attribute<device::attrs::max_surface_2d_width, ::cudaDevAttrMaxSurface2DWidth, int>();
    ::test_device_attribute<device::attrs::max_surface_2d_height, ::cudaDevAttrMaxSurface2DHeight, int>();
    ::test_device_attribute<device::attrs::max_surface_3d_width, ::cudaDevAttrMaxSurface3DWidth, int>();
    ::test_device_attribute<device::attrs::max_surface_3d_height, ::cudaDevAttrMaxSurface3DHeight, int>();
    ::test_device_attribute<device::attrs::max_surface_3d_depth, ::cudaDevAttrMaxSurface3DDepth, int>();
    ::test_device_attribute<device::attrs::max_surface_1d_layered_width, ::cudaDevAttrMaxSurface1DLayeredWidth, int>();
    ::test_device_attribute<device::attrs::max_surface_1d_layered_layers, ::cudaDevAttrMaxSurface1DLayeredLayers, int>();
    ::test_device_attribute<device::attrs::max_surface_2d_layered_width, ::cudaDevAttrMaxSurface2DLayeredWidth, int>();
    ::test_device_attribute<device::attrs::max_surface_2d_layered_height, ::cudaDevAttrMaxSurface2DLayeredHeight, int>();
    ::test_device_attribute<device::attrs::max_surface_2d_layered_layers, ::cudaDevAttrMaxSurface2DLayeredLayers, int>();
    ::test_device_attribute<device::attrs::max_surface_cubemap_width, ::cudaDevAttrMaxSurfaceCubemapWidth, int>();
    ::test_device_attribute<device::attrs::max_surface_cubemap_layered_width,
                            ::cudaDevAttrMaxSurfaceCubemapLayeredWidth,
                            int>();
    ::test_device_attribute<device::attrs::max_surface_cubemap_layered_layers,
                            ::cudaDevAttrMaxSurfaceCubemapLayeredLayers,
                            int>();
    ::test_device_attribute<device::attrs::max_registers_per_block, ::cudaDevAttrMaxRegistersPerBlock, int>();
    ::test_device_attribute<device::attrs::clock_rate, ::cudaDevAttrClockRate, int>();
    ::test_device_attribute<device::attrs::texture_alignment, ::cudaDevAttrTextureAlignment, int>();
    ::test_device_attribute<device::attrs::texture_pitch_alignment, ::cudaDevAttrTexturePitchAlignment, int>();
    ::test_device_attribute<device::attrs::gpu_overlap, ::cudaDevAttrGpuOverlap, bool>();
    ::test_device_attribute<device::attrs::multiprocessor_count, ::cudaDevAttrMultiProcessorCount, int>();
    ::test_device_attribute<device::attrs::kernel_exec_timeout, ::cudaDevAttrKernelExecTimeout, bool>();
    ::test_device_attribute<device::attrs::integrated, ::cudaDevAttrIntegrated, bool>();
    ::test_device_attribute<device::attrs::can_map_host_memory, ::cudaDevAttrCanMapHostMemory, bool>();
    ::test_device_attribute<device::attrs::compute_mode, ::cudaDevAttrComputeMode, ::cudaComputeMode>();
    ::test_device_attribute<device::attrs::concurrent_kernels, ::cudaDevAttrConcurrentKernels, bool>();
    ::test_device_attribute<device::attrs::ecc_enabled, ::cudaDevAttrEccEnabled, bool>();
    ::test_device_attribute<device::attrs::pci_bus_id, ::cudaDevAttrPciBusId, int>();
    ::test_device_attribute<device::attrs::pci_device_id, ::cudaDevAttrPciDeviceId, int>();
    ::test_device_attribute<device::attrs::tcc_driver, ::cudaDevAttrTccDriver, bool>();
    ::test_device_attribute<device::attrs::memory_clock_rate, ::cudaDevAttrMemoryClockRate, int>();
    ::test_device_attribute<device::attrs::global_memory_bus_width, ::cudaDevAttrGlobalMemoryBusWidth, int>();
    ::test_device_attribute<device::attrs::l2_cache_size, ::cudaDevAttrL2CacheSize, int>();
    ::test_device_attribute<device::attrs::max_threads_per_multiprocessor,
                            ::cudaDevAttrMaxThreadsPerMultiProcessor,
                            int>();
    ::test_device_attribute<device::attrs::unified_addressing, ::cudaDevAttrUnifiedAddressing, bool>();
    ::test_device_attribute<device::attrs::compute_capability_major, ::cudaDevAttrComputeCapabilityMajor, int>();
    ::test_device_attribute<device::attrs::compute_capability_minor, ::cudaDevAttrComputeCapabilityMinor, int>();
    ::test_device_attribute<device::attrs::stream_priorities_supported, ::cudaDevAttrStreamPrioritiesSupported, bool>();
    ::test_device_attribute<device::attrs::global_l1_cache_supported, ::cudaDevAttrGlobalL1CacheSupported, bool>();
    ::test_device_attribute<device::attrs::local_l1_cache_supported, ::cudaDevAttrLocalL1CacheSupported, bool>();
    ::test_device_attribute<device::attrs::max_shared_memory_per_multiprocessor,
                            ::cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                            int>();
    ::test_device_attribute<device::attrs::max_registers_per_multiprocessor,
                            ::cudaDevAttrMaxRegistersPerMultiprocessor,
                            int>();
    ::test_device_attribute<device::attrs::managed_memory, ::cudaDevAttrManagedMemory, bool>();
    ::test_device_attribute<device::attrs::is_multi_gpu_board, ::cudaDevAttrIsMultiGpuBoard, bool>();
    ::test_device_attribute<device::attrs::multi_gpu_board_group_id, ::cudaDevAttrMultiGpuBoardGroupID, int>();
    ::test_device_attribute<device::attrs::host_native_atomic_supported, ::cudaDevAttrHostNativeAtomicSupported, bool>();
    ::test_device_attribute<device::attrs::single_to_double_precision_perf_ratio,
                            ::cudaDevAttrSingleToDoublePrecisionPerfRatio,
                            int>();
    ::test_device_attribute<device::attrs::pageable_memory_access, ::cudaDevAttrPageableMemoryAccess, bool>();
    ::test_device_attribute<device::attrs::concurrent_managed_access, ::cudaDevAttrConcurrentManagedAccess, bool>();
    ::test_device_attribute<device::attrs::compute_preemption_supported, ::cudaDevAttrComputePreemptionSupported, bool>();
    ::test_device_attribute<device::attrs::can_use_host_pointer_for_registered_mem,
                            ::cudaDevAttrCanUseHostPointerForRegisteredMem,
                            bool>();
    ::test_device_attribute<device::attrs::cooperative_launch, ::cudaDevAttrCooperativeLaunch, bool>();
    ::test_device_attribute<device::attrs::cooperative_multi_device_launch,
                            ::cudaDevAttrCooperativeMultiDeviceLaunch,
                            bool>();
    ::test_device_attribute<device::attrs::can_flush_remote_writes, ::cudaDevAttrCanFlushRemoteWrites, bool>();
    ::test_device_attribute<device::attrs::host_register_supported, ::cudaDevAttrHostRegisterSupported, bool>();
    ::test_device_attribute<device::attrs::pageable_memory_access_uses_host_page_tables,
                            ::cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                            bool>();
    ::test_device_attribute<device::attrs::direct_managed_mem_access_from_host,
                            ::cudaDevAttrDirectManagedMemAccessFromHost,
                            bool>();
    ::test_device_attribute<device::attrs::max_shared_memory_per_block_optin,
                            ::cudaDevAttrMaxSharedMemoryPerBlockOptin,
                            int>();
    ::test_device_attribute<device::attrs::max_blocks_per_multiprocessor, ::cudaDevAttrMaxBlocksPerMultiprocessor, int>();
    ::test_device_attribute<device::attrs::max_persisting_l2_cache_size, ::cudaDevAttrMaxPersistingL2CacheSize, int>();
    ::test_device_attribute<device::attrs::max_access_policy_window_size, ::cudaDevAttrMaxAccessPolicyWindowSize, int>();
    ::test_device_attribute<device::attrs::reserved_shared_memory_per_block,
                            ::cudaDevAttrReservedSharedMemoryPerBlock,
                            int>();
    ::test_device_attribute<device::attrs::sparse_cuda_array_supported, ::cudaDevAttrSparseCudaArraySupported, bool>();
    ::test_device_attribute<device::attrs::host_register_read_only_supported,
                            ::cudaDevAttrHostRegisterReadOnlySupported,
                            bool>();
    ::test_device_attribute<device::attrs::memory_pools_supported, ::cudaDevAttrMemoryPoolsSupported, bool>();
    ::test_device_attribute<device::attrs::gpu_direct_rdma_supported, ::cudaDevAttrGPUDirectRDMASupported, bool>();
    ::test_device_attribute<device::attrs::gpu_direct_rdma_flush_writes_options,
                            ::cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                            ::cudaFlushGPUDirectRDMAWritesOptions>();
    ::test_device_attribute<device::attrs::gpu_direct_rdma_writes_ordering,
                            ::cudaDevAttrGPUDirectRDMAWritesOrdering,
                            ::cudaGPUDirectRDMAWritesOrdering>();
    ::test_device_attribute<device::attrs::memory_pool_supported_handle_types,
                            ::cudaDevAttrMemoryPoolSupportedHandleTypes,
                            ::cudaMemAllocationHandleType>();
    ::test_device_attribute<device::attrs::deferred_mapping_cuda_array_supported,
                            ::cudaDevAttrDeferredMappingCudaArraySupported,
                            bool>();
    ::test_device_attribute<device::attrs::ipc_event_support, ::cudaDevAttrIpcEventSupport, bool>();

#if CUDART_VERSION >= 12020
    ::test_device_attribute<device::attrs::numa_config, ::cudaDevAttrNumaConfig, ::cudaDeviceNumaConfig>();
    ::test_device_attribute<device::attrs::numa_id, ::cudaDevAttrNumaId, int>();
#endif

    SECTION("compute_mode")
    {
      STATIC_REQUIRE(::cudaComputeModeDefault == device::attrs::compute_mode.default_mode);
      STATIC_REQUIRE(::cudaComputeModeProhibited == device::attrs::compute_mode.prohibited_mode);
      STATIC_REQUIRE(::cudaComputeModeExclusiveProcess == device::attrs::compute_mode.exclusive_process_mode);

      auto mode = device_ref(0).attr(device::attrs::compute_mode);
      CUDAX_REQUIRE((mode == device::attrs::compute_mode.default_mode || //
                     mode == device::attrs::compute_mode.prohibited_mode || //
                     mode == device::attrs::compute_mode.exclusive_process_mode));
    }

    SECTION("gpu_direct_rdma_flush_writes_options")
    {
      STATIC_REQUIRE(
        ::cudaFlushGPUDirectRDMAWritesOptionHost == device::attrs::gpu_direct_rdma_flush_writes_options.host);
      STATIC_REQUIRE(
        ::cudaFlushGPUDirectRDMAWritesOptionMemOps == device::attrs::gpu_direct_rdma_flush_writes_options.mem_ops);

      auto options = device_ref(0).attr(device::attrs::gpu_direct_rdma_flush_writes_options);
      CUDAX_REQUIRE((options == device::attrs::gpu_direct_rdma_flush_writes_options.host || //
                     options == device::attrs::gpu_direct_rdma_flush_writes_options.mem_ops));
    }

    SECTION("gpu_direct_rdma_writes_ordering")
    {
      STATIC_REQUIRE(::cudaGPUDirectRDMAWritesOrderingNone == device::attrs::gpu_direct_rdma_writes_ordering.none);
      STATIC_REQUIRE(::cudaGPUDirectRDMAWritesOrderingOwner == device::attrs::gpu_direct_rdma_writes_ordering.owner);
      STATIC_REQUIRE(
        ::cudaGPUDirectRDMAWritesOrderingAllDevices == device::attrs::gpu_direct_rdma_writes_ordering.all_devices);

      auto ordering = device_ref(0).attr(device::attrs::gpu_direct_rdma_writes_ordering);
      CUDAX_REQUIRE((ordering == device::attrs::gpu_direct_rdma_writes_ordering.none || //
                     ordering == device::attrs::gpu_direct_rdma_writes_ordering.owner || //
                     ordering == device::attrs::gpu_direct_rdma_writes_ordering.all_devices));
    }

    SECTION("memory_pool_supported_handle_types")
    {
      STATIC_REQUIRE(::cudaMemHandleTypeNone == device::attrs::memory_pool_supported_handle_types.none);
      STATIC_REQUIRE(::cudaMemHandleTypePosixFileDescriptor
                     == device::attrs::memory_pool_supported_handle_types.posix_file_descriptor);
      STATIC_REQUIRE(::cudaMemHandleTypeWin32 == device::attrs::memory_pool_supported_handle_types.win32);
      STATIC_REQUIRE(::cudaMemHandleTypeWin32Kmt == device::attrs::memory_pool_supported_handle_types.win32_kmt);
#if CUDART_VERSION >= 12040
      STATIC_REQUIRE(::cudaMemHandleTypeFabric == 0x8);
      STATIC_REQUIRE(::cudaMemHandleTypeFabric == device::attrs::memory_pool_supported_handle_types.fabric);
#else
      STATIC_REQUIRE(0x8 == device::attrs::memory_pool_supported_handle_types.fabric);
#endif

      constexpr int all_handle_types =
        device::attrs::memory_pool_supported_handle_types.none
        | device::attrs::memory_pool_supported_handle_types.posix_file_descriptor
        | device::attrs::memory_pool_supported_handle_types.win32
        | device::attrs::memory_pool_supported_handle_types.win32_kmt
        | device::attrs::memory_pool_supported_handle_types.fabric;
      auto handle_types = device_ref(0).attr(device::attrs::memory_pool_supported_handle_types);
      CUDAX_REQUIRE(handle_types <= all_handle_types);
    }

#if CUDART_VERSION >= 12020
    SECTION("numa_config")
    {
      STATIC_REQUIRE(::cudaDeviceNumaConfigNone == device::attrs::numa_config.none);
      STATIC_REQUIRE(::cudaDeviceNumaConfigNumaNode == device::attrs::numa_config.numa_node);

      auto config = device_ref(0).attr(device::attrs::numa_config);
      CUDAX_REQUIRE((config == device::attrs::numa_config.none || //
                     config == device::attrs::numa_config.numa_node));
    }
#endif
    SECTION("Compute capability")
    {
      int compute_cap       = device_ref(0).attr(device::attrs::compute_capability);
      int compute_cap_major = device_ref(0).attr(device::attrs::compute_capability_major);
      int compute_cap_minor = device_ref(0).attr(device::attrs::compute_capability_minor);
      CUDAX_REQUIRE(compute_cap == 100 * compute_cap_major + 10 * compute_cap_minor);
    }
  }
}

TEST_CASE("global devices vector", "[device]")
{
  CUDAX_REQUIRE(cudax::devices.size() > 0);
  CUDAX_REQUIRE(cudax::devices.begin() != cudax::devices.end());
  CUDAX_REQUIRE(cudax::devices.begin() == cudax::devices.begin());
  CUDAX_REQUIRE(cudax::devices.end() == cudax::devices.end());
  CUDAX_REQUIRE(cudax::devices.size() == static_cast<size_t>(cudax::devices.end() - cudax::devices.begin()));

  CUDAX_REQUIRE(0 == cudax::devices[0].get());
  CUDAX_REQUIRE(cudax::device_ref{0} == cudax::devices[0]);

  CUDAX_REQUIRE(0 == (*cudax::devices.begin()).get());
  CUDAX_REQUIRE(cudax::device_ref{0} == *cudax::devices.begin());

  CUDAX_REQUIRE(0 == cudax::devices.begin()->get());
  CUDAX_REQUIRE(0 == cudax::devices.begin()[0].get());

  if (cudax::devices.size() > 1)
  {
    CUDAX_REQUIRE(1 == cudax::devices[1].get());
    CUDAX_REQUIRE(cudax::device_ref{0} != cudax::devices[1].get());

    CUDAX_REQUIRE(1 == (*std::next(cudax::devices.begin())).get());
    CUDAX_REQUIRE(1 == std::next(cudax::devices.begin())->get());
    CUDAX_REQUIRE(1 == cudax::devices.begin()[1].get());

    CUDAX_REQUIRE(cudax::devices.size() - 1 == (*std::prev(cudax::devices.end())).get());
    CUDAX_REQUIRE(cudax::devices.size() - 1 == std::prev(cudax::devices.end())->get());
    CUDAX_REQUIRE(cudax::devices.size() - 1 == cudax::devices.end()[-1].get());
  }

  try
  {
    [[maybe_unused]] const cudax::device& dev = cudax::devices.at(cudax::devices.size());
    CUDAX_REQUIRE(false); // should not get here
  }
  catch (const std::out_of_range&)
  {
    CUDAX_REQUIRE(true); // expected
  }
}
