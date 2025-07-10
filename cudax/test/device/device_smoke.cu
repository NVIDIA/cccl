//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/device.cuh>

#include <utility.cuh>

namespace
{
template <const auto& Attr, ::cudaDeviceAttr ExpectedAttr, class ExpectedResult>
[[maybe_unused]] auto test_device_attribute()
{
  cudax::device_ref dev0(0);
  STATIC_REQUIRE(Attr == ExpectedAttr);
  STATIC_REQUIRE(::cuda::std::is_same_v<cudax::device::attribute_result_t<Attr>, ExpectedResult>);

  auto result = dev0.attribute(Attr);
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(result), ExpectedResult>);
  CUDAX_REQUIRE(result == dev0.attribute<ExpectedAttr>());
  CUDAX_REQUIRE(result == Attr(dev0));
  return result;
}
} // namespace

C2H_CCCLRT_TEST("Smoke", "[device]")
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
    ::test_device_attribute<device::attributes::max_threads_per_block, ::cudaDevAttrMaxThreadsPerBlock, int>();
    ::test_device_attribute<device::attributes::max_block_dim_x, ::cudaDevAttrMaxBlockDimX, int>();
    ::test_device_attribute<device::attributes::max_block_dim_y, ::cudaDevAttrMaxBlockDimY, int>();
    ::test_device_attribute<device::attributes::max_block_dim_z, ::cudaDevAttrMaxBlockDimZ, int>();
    ::test_device_attribute<device::attributes::max_grid_dim_x, ::cudaDevAttrMaxGridDimX, int>();
    ::test_device_attribute<device::attributes::max_grid_dim_y, ::cudaDevAttrMaxGridDimY, int>();
    ::test_device_attribute<device::attributes::max_grid_dim_z, ::cudaDevAttrMaxGridDimZ, int>();
    ::test_device_attribute<device::attributes::max_shared_memory_per_block, ::cudaDevAttrMaxSharedMemoryPerBlock, int>();
    ::test_device_attribute<device::attributes::total_constant_memory, ::cudaDevAttrTotalConstantMemory, int>();
    ::test_device_attribute<device::attributes::warp_size, ::cudaDevAttrWarpSize, int>();
    ::test_device_attribute<device::attributes::max_pitch, ::cudaDevAttrMaxPitch, int>();
    ::test_device_attribute<device::attributes::max_texture_1d_width, ::cudaDevAttrMaxTexture1DWidth, int>();
    ::test_device_attribute<device::attributes::max_texture_1d_linear_width, ::cudaDevAttrMaxTexture1DLinearWidth, int>();
    ::test_device_attribute<device::attributes::max_texture_1d_mipmapped_width,
                            ::cudaDevAttrMaxTexture1DMipmappedWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_2d_width, ::cudaDevAttrMaxTexture2DWidth, int>();
    ::test_device_attribute<device::attributes::max_texture_2d_height, ::cudaDevAttrMaxTexture2DHeight, int>();
    ::test_device_attribute<device::attributes::max_texture_2d_linear_width, ::cudaDevAttrMaxTexture2DLinearWidth, int>();
    ::test_device_attribute<device::attributes::max_texture_2d_linear_height,
                            ::cudaDevAttrMaxTexture2DLinearHeight,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_2d_linear_pitch, ::cudaDevAttrMaxTexture2DLinearPitch, int>();
    ::test_device_attribute<device::attributes::max_texture_2d_mipmapped_width,
                            ::cudaDevAttrMaxTexture2DMipmappedWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_2d_mipmapped_height,
                            ::cudaDevAttrMaxTexture2DMipmappedHeight,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_3d_width, ::cudaDevAttrMaxTexture3DWidth, int>();
    ::test_device_attribute<device::attributes::max_texture_3d_height, ::cudaDevAttrMaxTexture3DHeight, int>();
    ::test_device_attribute<device::attributes::max_texture_3d_depth, ::cudaDevAttrMaxTexture3DDepth, int>();
    ::test_device_attribute<device::attributes::max_texture_3d_width_alt, ::cudaDevAttrMaxTexture3DWidthAlt, int>();
    ::test_device_attribute<device::attributes::max_texture_3d_height_alt, ::cudaDevAttrMaxTexture3DHeightAlt, int>();
    ::test_device_attribute<device::attributes::max_texture_3d_depth_alt, ::cudaDevAttrMaxTexture3DDepthAlt, int>();
    ::test_device_attribute<device::attributes::max_texture_cubemap_width, ::cudaDevAttrMaxTextureCubemapWidth, int>();
    ::test_device_attribute<device::attributes::max_texture_1d_layered_width,
                            ::cudaDevAttrMaxTexture1DLayeredWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_1d_layered_layers,
                            ::cudaDevAttrMaxTexture1DLayeredLayers,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_2d_layered_width,
                            ::cudaDevAttrMaxTexture2DLayeredWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_2d_layered_height,
                            ::cudaDevAttrMaxTexture2DLayeredHeight,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_2d_layered_layers,
                            ::cudaDevAttrMaxTexture2DLayeredLayers,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_cubemap_layered_width,
                            ::cudaDevAttrMaxTextureCubemapLayeredWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_texture_cubemap_layered_layers,
                            ::cudaDevAttrMaxTextureCubemapLayeredLayers,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_1d_width, ::cudaDevAttrMaxSurface1DWidth, int>();
    ::test_device_attribute<device::attributes::max_surface_2d_width, ::cudaDevAttrMaxSurface2DWidth, int>();
    ::test_device_attribute<device::attributes::max_surface_2d_height, ::cudaDevAttrMaxSurface2DHeight, int>();
    ::test_device_attribute<device::attributes::max_surface_3d_width, ::cudaDevAttrMaxSurface3DWidth, int>();
    ::test_device_attribute<device::attributes::max_surface_3d_height, ::cudaDevAttrMaxSurface3DHeight, int>();
    ::test_device_attribute<device::attributes::max_surface_3d_depth, ::cudaDevAttrMaxSurface3DDepth, int>();
    ::test_device_attribute<device::attributes::max_surface_1d_layered_width,
                            ::cudaDevAttrMaxSurface1DLayeredWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_1d_layered_layers,
                            ::cudaDevAttrMaxSurface1DLayeredLayers,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_2d_layered_width,
                            ::cudaDevAttrMaxSurface2DLayeredWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_2d_layered_height,
                            ::cudaDevAttrMaxSurface2DLayeredHeight,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_2d_layered_layers,
                            ::cudaDevAttrMaxSurface2DLayeredLayers,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_cubemap_width, ::cudaDevAttrMaxSurfaceCubemapWidth, int>();
    ::test_device_attribute<device::attributes::max_surface_cubemap_layered_width,
                            ::cudaDevAttrMaxSurfaceCubemapLayeredWidth,
                            int>();
    ::test_device_attribute<device::attributes::max_surface_cubemap_layered_layers,
                            ::cudaDevAttrMaxSurfaceCubemapLayeredLayers,
                            int>();
    ::test_device_attribute<device::attributes::max_registers_per_block, ::cudaDevAttrMaxRegistersPerBlock, int>();
    ::test_device_attribute<device::attributes::clock_rate, ::cudaDevAttrClockRate, int>();
    ::test_device_attribute<device::attributes::texture_alignment, ::cudaDevAttrTextureAlignment, int>();
    ::test_device_attribute<device::attributes::texture_pitch_alignment, ::cudaDevAttrTexturePitchAlignment, int>();
    ::test_device_attribute<device::attributes::gpu_overlap, ::cudaDevAttrGpuOverlap, bool>();
    ::test_device_attribute<device::attributes::multiprocessor_count, ::cudaDevAttrMultiProcessorCount, int>();
    ::test_device_attribute<device::attributes::kernel_exec_timeout, ::cudaDevAttrKernelExecTimeout, bool>();
    ::test_device_attribute<device::attributes::integrated, ::cudaDevAttrIntegrated, bool>();
    ::test_device_attribute<device::attributes::can_map_host_memory, ::cudaDevAttrCanMapHostMemory, bool>();
    ::test_device_attribute<device::attributes::compute_mode, ::cudaDevAttrComputeMode, ::cudaComputeMode>();
    ::test_device_attribute<device::attributes::concurrent_kernels, ::cudaDevAttrConcurrentKernels, bool>();
    ::test_device_attribute<device::attributes::ecc_enabled, ::cudaDevAttrEccEnabled, bool>();
    ::test_device_attribute<device::attributes::pci_bus_id, ::cudaDevAttrPciBusId, int>();
    ::test_device_attribute<device::attributes::pci_device_id, ::cudaDevAttrPciDeviceId, int>();
    ::test_device_attribute<device::attributes::tcc_driver, ::cudaDevAttrTccDriver, bool>();
    ::test_device_attribute<device::attributes::memory_clock_rate, ::cudaDevAttrMemoryClockRate, int>();
    ::test_device_attribute<device::attributes::global_memory_bus_width, ::cudaDevAttrGlobalMemoryBusWidth, int>();
    ::test_device_attribute<device::attributes::l2_cache_size, ::cudaDevAttrL2CacheSize, int>();
    ::test_device_attribute<device::attributes::max_threads_per_multiprocessor,
                            ::cudaDevAttrMaxThreadsPerMultiProcessor,
                            int>();
    ::test_device_attribute<device::attributes::unified_addressing, ::cudaDevAttrUnifiedAddressing, bool>();
    ::test_device_attribute<device::attributes::compute_capability_major, ::cudaDevAttrComputeCapabilityMajor, int>();
    ::test_device_attribute<device::attributes::compute_capability_minor, ::cudaDevAttrComputeCapabilityMinor, int>();
    ::test_device_attribute<device::attributes::stream_priorities_supported,
                            ::cudaDevAttrStreamPrioritiesSupported,
                            bool>();
    ::test_device_attribute<device::attributes::global_l1_cache_supported, ::cudaDevAttrGlobalL1CacheSupported, bool>();
    ::test_device_attribute<device::attributes::local_l1_cache_supported, ::cudaDevAttrLocalL1CacheSupported, bool>();
    ::test_device_attribute<device::attributes::max_shared_memory_per_multiprocessor,
                            ::cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                            int>();
    ::test_device_attribute<device::attributes::max_registers_per_multiprocessor,
                            ::cudaDevAttrMaxRegistersPerMultiprocessor,
                            int>();
    ::test_device_attribute<device::attributes::managed_memory, ::cudaDevAttrManagedMemory, bool>();
    ::test_device_attribute<device::attributes::is_multi_gpu_board, ::cudaDevAttrIsMultiGpuBoard, bool>();
    ::test_device_attribute<device::attributes::multi_gpu_board_group_id, ::cudaDevAttrMultiGpuBoardGroupID, int>();
    ::test_device_attribute<device::attributes::host_native_atomic_supported,
                            ::cudaDevAttrHostNativeAtomicSupported,
                            bool>();
    ::test_device_attribute<device::attributes::single_to_double_precision_perf_ratio,
                            ::cudaDevAttrSingleToDoublePrecisionPerfRatio,
                            int>();
    ::test_device_attribute<device::attributes::pageable_memory_access, ::cudaDevAttrPageableMemoryAccess, bool>();
    ::test_device_attribute<device::attributes::concurrent_managed_access, ::cudaDevAttrConcurrentManagedAccess, bool>();
    ::test_device_attribute<device::attributes::compute_preemption_supported,
                            ::cudaDevAttrComputePreemptionSupported,
                            bool>();
    ::test_device_attribute<device::attributes::can_use_host_pointer_for_registered_mem,
                            ::cudaDevAttrCanUseHostPointerForRegisteredMem,
                            bool>();
    ::test_device_attribute<device::attributes::cooperative_launch, ::cudaDevAttrCooperativeLaunch, bool>();
    ::test_device_attribute<device::attributes::can_flush_remote_writes, ::cudaDevAttrCanFlushRemoteWrites, bool>();
    ::test_device_attribute<device::attributes::host_register_supported, ::cudaDevAttrHostRegisterSupported, bool>();
    ::test_device_attribute<device::attributes::pageable_memory_access_uses_host_page_tables,
                            ::cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                            bool>();
    ::test_device_attribute<device::attributes::direct_managed_mem_access_from_host,
                            ::cudaDevAttrDirectManagedMemAccessFromHost,
                            bool>();
    ::test_device_attribute<device::attributes::max_shared_memory_per_block_optin,
                            ::cudaDevAttrMaxSharedMemoryPerBlockOptin,
                            int>();
    ::test_device_attribute<device::attributes::max_blocks_per_multiprocessor,
                            ::cudaDevAttrMaxBlocksPerMultiprocessor,
                            int>();
    ::test_device_attribute<device::attributes::max_persisting_l2_cache_size,
                            ::cudaDevAttrMaxPersistingL2CacheSize,
                            int>();
    ::test_device_attribute<device::attributes::max_access_policy_window_size,
                            ::cudaDevAttrMaxAccessPolicyWindowSize,
                            int>();
    ::test_device_attribute<device::attributes::reserved_shared_memory_per_block,
                            ::cudaDevAttrReservedSharedMemoryPerBlock,
                            int>();
    ::test_device_attribute<device::attributes::sparse_cuda_array_supported,
                            ::cudaDevAttrSparseCudaArraySupported,
                            bool>();
    ::test_device_attribute<device::attributes::host_register_read_only_supported,
                            ::cudaDevAttrHostRegisterReadOnlySupported,
                            bool>();
    ::test_device_attribute<device::attributes::memory_pools_supported, ::cudaDevAttrMemoryPoolsSupported, bool>();
    ::test_device_attribute<device::attributes::gpu_direct_rdma_supported, ::cudaDevAttrGPUDirectRDMASupported, bool>();
    ::test_device_attribute<device::attributes::gpu_direct_rdma_flush_writes_options,
                            ::cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                            ::cudaFlushGPUDirectRDMAWritesOptions>();
    ::test_device_attribute<device::attributes::gpu_direct_rdma_writes_ordering,
                            ::cudaDevAttrGPUDirectRDMAWritesOrdering,
                            ::cudaGPUDirectRDMAWritesOrdering>();
    ::test_device_attribute<device::attributes::memory_pool_supported_handle_types,
                            ::cudaDevAttrMemoryPoolSupportedHandleTypes,
                            ::cudaMemAllocationHandleType>();
    ::test_device_attribute<device::attributes::deferred_mapping_cuda_array_supported,
                            ::cudaDevAttrDeferredMappingCudaArraySupported,
                            bool>();
    ::test_device_attribute<device::attributes::ipc_event_support, ::cudaDevAttrIpcEventSupport, bool>();

#if CUDART_VERSION >= 12020
    ::test_device_attribute<device::attributes::numa_config, ::cudaDevAttrNumaConfig, ::cudaDeviceNumaConfig>();
    ::test_device_attribute<device::attributes::numa_id, ::cudaDevAttrNumaId, int>();
#endif

    SECTION("compute_mode")
    {
      STATIC_REQUIRE(::cudaComputeModeDefault == device::attributes::compute_mode.default_mode);
      STATIC_REQUIRE(::cudaComputeModeProhibited == device::attributes::compute_mode.prohibited_mode);
      STATIC_REQUIRE(::cudaComputeModeExclusiveProcess == device::attributes::compute_mode.exclusive_process_mode);

      auto mode = device_ref(0).attribute(device::attributes::compute_mode);
      CUDAX_REQUIRE((mode == device::attributes::compute_mode.default_mode || //
                     mode == device::attributes::compute_mode.prohibited_mode || //
                     mode == device::attributes::compute_mode.exclusive_process_mode));
    }

    SECTION("gpu_direct_rdma_flush_writes_options")
    {
      STATIC_REQUIRE(
        ::cudaFlushGPUDirectRDMAWritesOptionHost == device::attributes::gpu_direct_rdma_flush_writes_options.host);
      STATIC_REQUIRE(
        ::cudaFlushGPUDirectRDMAWritesOptionMemOps == device::attributes::gpu_direct_rdma_flush_writes_options.mem_ops);

      [[maybe_unused]] auto options = device_ref(0).attribute(device::attributes::gpu_direct_rdma_flush_writes_options);
#if !_CCCL_COMPILER(MSVC)
      CUDAX_REQUIRE((options == device::attributes::gpu_direct_rdma_flush_writes_options.host || //
                     options == device::attributes::gpu_direct_rdma_flush_writes_options.mem_ops));
#endif
    }

    SECTION("gpu_direct_rdma_writes_ordering")
    {
      STATIC_REQUIRE(::cudaGPUDirectRDMAWritesOrderingNone == device::attributes::gpu_direct_rdma_writes_ordering.none);
      STATIC_REQUIRE(
        ::cudaGPUDirectRDMAWritesOrderingOwner == device::attributes::gpu_direct_rdma_writes_ordering.owner);
      STATIC_REQUIRE(
        ::cudaGPUDirectRDMAWritesOrderingAllDevices == device::attributes::gpu_direct_rdma_writes_ordering.all_devices);

      auto ordering = device_ref(0).attribute(device::attributes::gpu_direct_rdma_writes_ordering);
      CUDAX_REQUIRE((ordering == device::attributes::gpu_direct_rdma_writes_ordering.none || //
                     ordering == device::attributes::gpu_direct_rdma_writes_ordering.owner || //
                     ordering == device::attributes::gpu_direct_rdma_writes_ordering.all_devices));
    }

    SECTION("memory_pool_supported_handle_types")
    {
      STATIC_REQUIRE(::cudaMemHandleTypeNone == device::attributes::memory_pool_supported_handle_types.none);
      STATIC_REQUIRE(::cudaMemHandleTypePosixFileDescriptor
                     == device::attributes::memory_pool_supported_handle_types.posix_file_descriptor);
      STATIC_REQUIRE(::cudaMemHandleTypeWin32 == device::attributes::memory_pool_supported_handle_types.win32);
      STATIC_REQUIRE(::cudaMemHandleTypeWin32Kmt == device::attributes::memory_pool_supported_handle_types.win32_kmt);
#if CUDART_VERSION >= 12040
      STATIC_REQUIRE(::cudaMemHandleTypeFabric == 0x8);
      STATIC_REQUIRE(::cudaMemHandleTypeFabric == device::attributes::memory_pool_supported_handle_types.fabric);
#else
      STATIC_REQUIRE(0x8 == device::attributes::memory_pool_supported_handle_types.fabric);
#endif

      constexpr int all_handle_types =
        device::attributes::memory_pool_supported_handle_types.none
        | device::attributes::memory_pool_supported_handle_types.posix_file_descriptor
        | device::attributes::memory_pool_supported_handle_types.win32
        | device::attributes::memory_pool_supported_handle_types.win32_kmt
        | device::attributes::memory_pool_supported_handle_types.fabric;
      auto handle_types = device_ref(0).attribute(device::attributes::memory_pool_supported_handle_types);
      CUDAX_REQUIRE(static_cast<int>(handle_types) <= static_cast<int>(all_handle_types));
    }

#if CUDART_VERSION >= 12020
    SECTION("numa_config")
    {
      STATIC_REQUIRE(::cudaDeviceNumaConfigNone == device::attributes::numa_config.none);
      STATIC_REQUIRE(::cudaDeviceNumaConfigNumaNode == device::attributes::numa_config.numa_node);

      auto config = device_ref(0).attribute(device::attributes::numa_config);
      CUDAX_REQUIRE((config == device::attributes::numa_config.none || //
                     config == device::attributes::numa_config.numa_node));
    }
#endif
    SECTION("Compute capability")
    {
      int compute_cap       = device_ref(0).attribute(device::attributes::compute_capability);
      int compute_cap_major = device_ref(0).attribute(device::attributes::compute_capability_major);
      int compute_cap_minor = device_ref(0).attribute(device::attributes::compute_capability_minor);
      CUDAX_REQUIRE(compute_cap == 100 * compute_cap_major + 10 * compute_cap_minor);
    }
  }
  SECTION("Name")
  {
    std::string name = device_ref(0).name();
    CUDAX_REQUIRE(name.length() != 0);
    CUDAX_REQUIRE(name[0] != 0);
  }
}

C2H_CCCLRT_TEST("global devices vector", "[device]")
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

    CUDAX_REQUIRE(cudax::devices.size() - 1 == static_cast<std::size_t>((*std::prev(cudax::devices.end())).get()));
    CUDAX_REQUIRE(cudax::devices.size() - 1 == static_cast<std::size_t>(std::prev(cudax::devices.end())->get()));
    CUDAX_REQUIRE(cudax::devices.size() - 1 == static_cast<std::size_t>(cudax::devices.end()[-1].get()));

    auto peers = cudax::devices[0].peer_devices();
    for (auto peer : peers)
    {
      CUDAX_REQUIRE(cudax::devices[0].has_peer_access_to(peer))
      CUDAX_REQUIRE(peer.has_peer_access_to(cudax::devices[0]));
    }
  }

#if _CCCL_HAS_EXCEPTIONS()
  try
  {
    [[maybe_unused]] const cudax::device& dev = cudax::devices[cudax::devices.size()];
    CUDAX_REQUIRE(false); // should not get here
  }
  catch (const std::out_of_range&)
  {
    CUDAX_REQUIRE(true); // expected
  }
#endif // _CCCL_HAS_EXCEPTIONS()
}
