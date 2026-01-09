//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstddef>

#include <testing.cuh>

namespace
{
template <const auto& Attr, ::cudaDeviceAttr ExpectedAttr, class ExpectedResult>
[[maybe_unused]] auto test_device_attribute()
{
  cuda::device_ref dev0(0);
  STATIC_REQUIRE(Attr == ExpectedAttr);
  STATIC_REQUIRE(::cuda::std::is_same_v<cuda::device_attribute_result_t<Attr>, ExpectedResult>);

  auto result = dev0.attribute(Attr);
  STATIC_REQUIRE(::cuda::std::is_same_v<decltype(result), ExpectedResult>);
  CCCLRT_REQUIRE(result == dev0.attribute<ExpectedAttr>());
  CCCLRT_REQUIRE(result == Attr(dev0));
  return result;
}
} // namespace

C2H_CCCLRT_TEST("init", "[device]")
{
  cuda::device_ref dev{0};
  dev.init();
  CCCLRT_REQUIRE(cuda::__driver::__isPrimaryCtxActive(cuda::__driver::__deviceGet(0)));
}

C2H_CCCLRT_TEST("Smoke", "[device]")
{
  namespace attributes = cuda::device_attributes;
  using cuda::device_ref;

  SECTION("Compare")
  {
    CCCLRT_REQUIRE(device_ref{0} == device_ref{0});
    CCCLRT_REQUIRE(device_ref{0} == 0);
    CCCLRT_REQUIRE(0 == device_ref{0});
    if (cuda::devices.size() > 1)
    {
      CCCLRT_REQUIRE(device_ref{1} != device_ref{0});
      CCCLRT_REQUIRE(device_ref{1} != 0);
      CCCLRT_REQUIRE(0 != device_ref{1});
    }
  }

  SECTION("Attributes")
  {
    ::test_device_attribute<attributes::max_threads_per_block, ::cudaDevAttrMaxThreadsPerBlock, int>();
    ::test_device_attribute<attributes::max_block_dim_x, ::cudaDevAttrMaxBlockDimX, int>();
    ::test_device_attribute<attributes::max_block_dim_y, ::cudaDevAttrMaxBlockDimY, int>();
    ::test_device_attribute<attributes::max_block_dim_z, ::cudaDevAttrMaxBlockDimZ, int>();
    ::test_device_attribute<attributes::max_grid_dim_x, ::cudaDevAttrMaxGridDimX, int>();
    ::test_device_attribute<attributes::max_grid_dim_y, ::cudaDevAttrMaxGridDimY, int>();
    ::test_device_attribute<attributes::max_grid_dim_z, ::cudaDevAttrMaxGridDimZ, int>();
    ::test_device_attribute<attributes::max_shared_memory_per_block,
                            ::cudaDevAttrMaxSharedMemoryPerBlock,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::total_constant_memory, ::cudaDevAttrTotalConstantMemory, cuda::std::size_t>();
    ::test_device_attribute<attributes::warp_size, ::cudaDevAttrWarpSize, int>();
    ::test_device_attribute<attributes::max_pitch, ::cudaDevAttrMaxPitch, cuda::std::size_t>();
    ::test_device_attribute<attributes::max_texture_1d_width, ::cudaDevAttrMaxTexture1DWidth, int>();
    ::test_device_attribute<attributes::max_texture_1d_linear_width, ::cudaDevAttrMaxTexture1DLinearWidth, int>();
    ::test_device_attribute<attributes::max_texture_1d_mipmapped_width, ::cudaDevAttrMaxTexture1DMipmappedWidth, int>();
    ::test_device_attribute<attributes::max_texture_2d_width, ::cudaDevAttrMaxTexture2DWidth, int>();
    ::test_device_attribute<attributes::max_texture_2d_height, ::cudaDevAttrMaxTexture2DHeight, int>();
    ::test_device_attribute<attributes::max_texture_2d_linear_width, ::cudaDevAttrMaxTexture2DLinearWidth, int>();
    ::test_device_attribute<attributes::max_texture_2d_linear_height, ::cudaDevAttrMaxTexture2DLinearHeight, int>();
    ::test_device_attribute<attributes::max_texture_2d_linear_pitch,
                            ::cudaDevAttrMaxTexture2DLinearPitch,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::max_texture_2d_mipmapped_width, ::cudaDevAttrMaxTexture2DMipmappedWidth, int>();
    ::test_device_attribute<attributes::max_texture_2d_mipmapped_height, ::cudaDevAttrMaxTexture2DMipmappedHeight, int>();
    ::test_device_attribute<attributes::max_texture_3d_width, ::cudaDevAttrMaxTexture3DWidth, int>();
    ::test_device_attribute<attributes::max_texture_3d_height, ::cudaDevAttrMaxTexture3DHeight, int>();
    ::test_device_attribute<attributes::max_texture_3d_depth, ::cudaDevAttrMaxTexture3DDepth, int>();
    ::test_device_attribute<attributes::max_texture_3d_width_alt, ::cudaDevAttrMaxTexture3DWidthAlt, int>();
    ::test_device_attribute<attributes::max_texture_3d_height_alt, ::cudaDevAttrMaxTexture3DHeightAlt, int>();
    ::test_device_attribute<attributes::max_texture_3d_depth_alt, ::cudaDevAttrMaxTexture3DDepthAlt, int>();
    ::test_device_attribute<attributes::max_texture_cubemap_width, ::cudaDevAttrMaxTextureCubemapWidth, int>();
    ::test_device_attribute<attributes::max_texture_1d_layered_width, ::cudaDevAttrMaxTexture1DLayeredWidth, int>();
    ::test_device_attribute<attributes::max_texture_1d_layered_layers, ::cudaDevAttrMaxTexture1DLayeredLayers, int>();
    ::test_device_attribute<attributes::max_texture_2d_layered_width, ::cudaDevAttrMaxTexture2DLayeredWidth, int>();
    ::test_device_attribute<attributes::max_texture_2d_layered_height, ::cudaDevAttrMaxTexture2DLayeredHeight, int>();
    ::test_device_attribute<attributes::max_texture_2d_layered_layers, ::cudaDevAttrMaxTexture2DLayeredLayers, int>();
    ::test_device_attribute<attributes::max_texture_cubemap_layered_width,
                            ::cudaDevAttrMaxTextureCubemapLayeredWidth,
                            int>();
    ::test_device_attribute<attributes::max_texture_cubemap_layered_layers,
                            ::cudaDevAttrMaxTextureCubemapLayeredLayers,
                            int>();
    ::test_device_attribute<attributes::max_surface_1d_width, ::cudaDevAttrMaxSurface1DWidth, int>();
    ::test_device_attribute<attributes::max_surface_2d_width, ::cudaDevAttrMaxSurface2DWidth, int>();
    ::test_device_attribute<attributes::max_surface_2d_height, ::cudaDevAttrMaxSurface2DHeight, int>();
    ::test_device_attribute<attributes::max_surface_3d_width, ::cudaDevAttrMaxSurface3DWidth, int>();
    ::test_device_attribute<attributes::max_surface_3d_height, ::cudaDevAttrMaxSurface3DHeight, int>();
    ::test_device_attribute<attributes::max_surface_3d_depth, ::cudaDevAttrMaxSurface3DDepth, int>();
    ::test_device_attribute<attributes::max_surface_1d_layered_width, ::cudaDevAttrMaxSurface1DLayeredWidth, int>();
    ::test_device_attribute<attributes::max_surface_1d_layered_layers, ::cudaDevAttrMaxSurface1DLayeredLayers, int>();
    ::test_device_attribute<attributes::max_surface_2d_layered_width, ::cudaDevAttrMaxSurface2DLayeredWidth, int>();
    ::test_device_attribute<attributes::max_surface_2d_layered_height, ::cudaDevAttrMaxSurface2DLayeredHeight, int>();
    ::test_device_attribute<attributes::max_surface_2d_layered_layers, ::cudaDevAttrMaxSurface2DLayeredLayers, int>();
    ::test_device_attribute<attributes::max_surface_cubemap_width, ::cudaDevAttrMaxSurfaceCubemapWidth, int>();
    ::test_device_attribute<attributes::max_surface_cubemap_layered_width,
                            ::cudaDevAttrMaxSurfaceCubemapLayeredWidth,
                            int>();
    ::test_device_attribute<attributes::max_surface_cubemap_layered_layers,
                            ::cudaDevAttrMaxSurfaceCubemapLayeredLayers,
                            int>();
    ::test_device_attribute<attributes::max_registers_per_block, ::cudaDevAttrMaxRegistersPerBlock, int>();
    ::test_device_attribute<attributes::clock_rate, ::cudaDevAttrClockRate, int>();
    ::test_device_attribute<attributes::texture_alignment, ::cudaDevAttrTextureAlignment, cuda::std::size_t>();
    ::test_device_attribute<attributes::texture_pitch_alignment, ::cudaDevAttrTexturePitchAlignment, cuda::std::size_t>();
    ::test_device_attribute<attributes::gpu_overlap, ::cudaDevAttrGpuOverlap, bool>();
    ::test_device_attribute<attributes::multiprocessor_count, ::cudaDevAttrMultiProcessorCount, int>();
    ::test_device_attribute<attributes::kernel_exec_timeout, ::cudaDevAttrKernelExecTimeout, bool>();
    ::test_device_attribute<attributes::integrated, ::cudaDevAttrIntegrated, bool>();
    ::test_device_attribute<attributes::can_map_host_memory, ::cudaDevAttrCanMapHostMemory, bool>();
    ::test_device_attribute<attributes::compute_mode, ::cudaDevAttrComputeMode, ::cudaComputeMode>();
    ::test_device_attribute<attributes::concurrent_kernels, ::cudaDevAttrConcurrentKernels, bool>();
    ::test_device_attribute<attributes::ecc_enabled, ::cudaDevAttrEccEnabled, bool>();
    ::test_device_attribute<attributes::pci_bus_id, ::cudaDevAttrPciBusId, int>();
    ::test_device_attribute<attributes::pci_device_id, ::cudaDevAttrPciDeviceId, int>();
    ::test_device_attribute<attributes::tcc_driver, ::cudaDevAttrTccDriver, bool>();
    ::test_device_attribute<attributes::l2_cache_size, ::cudaDevAttrL2CacheSize, cuda::std::size_t>();
    ::test_device_attribute<attributes::max_threads_per_multiprocessor, ::cudaDevAttrMaxThreadsPerMultiProcessor, int>();
    ::test_device_attribute<attributes::unified_addressing, ::cudaDevAttrUnifiedAddressing, bool>();
    ::test_device_attribute<attributes::compute_capability_major, ::cudaDevAttrComputeCapabilityMajor, int>();
    ::test_device_attribute<attributes::compute_capability_minor, ::cudaDevAttrComputeCapabilityMinor, int>();
    ::test_device_attribute<attributes::stream_priorities_supported, ::cudaDevAttrStreamPrioritiesSupported, bool>();
    ::test_device_attribute<attributes::global_l1_cache_supported, ::cudaDevAttrGlobalL1CacheSupported, bool>();
    ::test_device_attribute<attributes::local_l1_cache_supported, ::cudaDevAttrLocalL1CacheSupported, bool>();
    ::test_device_attribute<attributes::max_shared_memory_per_multiprocessor,
                            ::cudaDevAttrMaxSharedMemoryPerMultiprocessor,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::max_registers_per_multiprocessor,
                            ::cudaDevAttrMaxRegistersPerMultiprocessor,
                            int>();
    ::test_device_attribute<attributes::is_multi_gpu_board, ::cudaDevAttrIsMultiGpuBoard, bool>();
    ::test_device_attribute<attributes::multi_gpu_board_group_id, ::cudaDevAttrMultiGpuBoardGroupID, int>();
    ::test_device_attribute<attributes::host_native_atomic_supported, ::cudaDevAttrHostNativeAtomicSupported, bool>();
    ::test_device_attribute<attributes::single_to_double_precision_perf_ratio,
                            ::cudaDevAttrSingleToDoublePrecisionPerfRatio,
                            int>();
    ::test_device_attribute<attributes::pageable_memory_access, ::cudaDevAttrPageableMemoryAccess, bool>();
    ::test_device_attribute<attributes::concurrent_managed_access, ::cudaDevAttrConcurrentManagedAccess, bool>();
    ::test_device_attribute<attributes::compute_preemption_supported, ::cudaDevAttrComputePreemptionSupported, bool>();
    ::test_device_attribute<attributes::can_use_host_pointer_for_registered_mem,
                            ::cudaDevAttrCanUseHostPointerForRegisteredMem,
                            bool>();
    ::test_device_attribute<attributes::cooperative_launch, ::cudaDevAttrCooperativeLaunch, bool>();
    ::test_device_attribute<attributes::can_flush_remote_writes, ::cudaDevAttrCanFlushRemoteWrites, bool>();
    ::test_device_attribute<attributes::host_register_supported, ::cudaDevAttrHostRegisterSupported, bool>();
    ::test_device_attribute<attributes::pageable_memory_access_uses_host_page_tables,
                            ::cudaDevAttrPageableMemoryAccessUsesHostPageTables,
                            bool>();
    ::test_device_attribute<attributes::direct_managed_mem_access_from_host,
                            ::cudaDevAttrDirectManagedMemAccessFromHost,
                            bool>();
    ::test_device_attribute<attributes::max_shared_memory_per_block_optin,
                            ::cudaDevAttrMaxSharedMemoryPerBlockOptin,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::max_blocks_per_multiprocessor, ::cudaDevAttrMaxBlocksPerMultiprocessor, int>();
    ::test_device_attribute<attributes::max_persisting_l2_cache_size,
                            ::cudaDevAttrMaxPersistingL2CacheSize,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::max_access_policy_window_size,
                            ::cudaDevAttrMaxAccessPolicyWindowSize,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::reserved_shared_memory_per_block,
                            ::cudaDevAttrReservedSharedMemoryPerBlock,
                            cuda::std::size_t>();
    ::test_device_attribute<attributes::sparse_cuda_array_supported, ::cudaDevAttrSparseCudaArraySupported, bool>();
    ::test_device_attribute<attributes::host_register_read_only_supported,
                            ::cudaDevAttrHostRegisterReadOnlySupported,
                            bool>();
    ::test_device_attribute<attributes::memory_pools_supported, ::cudaDevAttrMemoryPoolsSupported, bool>();
    ::test_device_attribute<attributes::gpu_direct_rdma_supported, ::cudaDevAttrGPUDirectRDMASupported, bool>();
    ::test_device_attribute<attributes::gpu_direct_rdma_flush_writes_options,
                            ::cudaDevAttrGPUDirectRDMAFlushWritesOptions,
                            ::cudaFlushGPUDirectRDMAWritesOptions>();
    ::test_device_attribute<attributes::gpu_direct_rdma_writes_ordering,
                            ::cudaDevAttrGPUDirectRDMAWritesOrdering,
                            ::cudaGPUDirectRDMAWritesOrdering>();
    ::test_device_attribute<attributes::memory_pool_supported_handle_types,
                            ::cudaDevAttrMemoryPoolSupportedHandleTypes,
                            ::cudaMemAllocationHandleType>();
    ::test_device_attribute<attributes::deferred_mapping_cuda_array_supported,
                            ::cudaDevAttrDeferredMappingCudaArraySupported,
                            bool>();
    ::test_device_attribute<attributes::ipc_event_support, ::cudaDevAttrIpcEventSupport, bool>();

#if _CCCL_CTK_AT_LEAST(12, 2)
    ::test_device_attribute<attributes::numa_config, ::cudaDevAttrNumaConfig, ::cudaDeviceNumaConfig>();
    ::test_device_attribute<attributes::numa_id, ::cudaDevAttrNumaId, int>();
#endif // _CCCL_CTK_AT_LEAST(12, 2)

    SECTION("compute_mode")
    {
      STATIC_REQUIRE(::cudaComputeModeDefault == attributes::compute_mode.default_mode);
      STATIC_REQUIRE(::cudaComputeModeProhibited == attributes::compute_mode.prohibited_mode);
      STATIC_REQUIRE(::cudaComputeModeExclusiveProcess == attributes::compute_mode.exclusive_process_mode);

      auto mode = device_ref(0).attribute(attributes::compute_mode);
      CCCLRT_REQUIRE((mode == attributes::compute_mode.default_mode || //
                      mode == attributes::compute_mode.prohibited_mode || //
                      mode == attributes::compute_mode.exclusive_process_mode));
    }

    SECTION("gpu_direct_rdma_flush_writes_options")
    {
      STATIC_REQUIRE(::cudaFlushGPUDirectRDMAWritesOptionHost == attributes::gpu_direct_rdma_flush_writes_options.host);
      STATIC_REQUIRE(
        ::cudaFlushGPUDirectRDMAWritesOptionMemOps == attributes::gpu_direct_rdma_flush_writes_options.mem_ops);

      [[maybe_unused]] auto options = device_ref(0).attribute(attributes::gpu_direct_rdma_flush_writes_options);
#if !_CCCL_COMPILER(MSVC)
      CCCLRT_REQUIRE((options == attributes::gpu_direct_rdma_flush_writes_options.host || //
                      options == attributes::gpu_direct_rdma_flush_writes_options.mem_ops));
#endif
    }

    SECTION("gpu_direct_rdma_writes_ordering")
    {
      STATIC_REQUIRE(::cudaGPUDirectRDMAWritesOrderingNone == attributes::gpu_direct_rdma_writes_ordering.none);
      STATIC_REQUIRE(::cudaGPUDirectRDMAWritesOrderingOwner == attributes::gpu_direct_rdma_writes_ordering.owner);
      STATIC_REQUIRE(
        ::cudaGPUDirectRDMAWritesOrderingAllDevices == attributes::gpu_direct_rdma_writes_ordering.all_devices);

      auto ordering = device_ref(0).attribute(attributes::gpu_direct_rdma_writes_ordering);
      CCCLRT_REQUIRE((ordering == attributes::gpu_direct_rdma_writes_ordering.none || //
                      ordering == attributes::gpu_direct_rdma_writes_ordering.owner || //
                      ordering == attributes::gpu_direct_rdma_writes_ordering.all_devices));
    }

    SECTION("memory_pool_supported_handle_types")
    {
      STATIC_REQUIRE(::cudaMemHandleTypeNone == attributes::memory_pool_supported_handle_types.none);
      STATIC_REQUIRE(
        ::cudaMemHandleTypePosixFileDescriptor == attributes::memory_pool_supported_handle_types.posix_file_descriptor);
      STATIC_REQUIRE(::cudaMemHandleTypeWin32 == attributes::memory_pool_supported_handle_types.win32);
      STATIC_REQUIRE(::cudaMemHandleTypeWin32Kmt == attributes::memory_pool_supported_handle_types.win32_kmt);
#if _CCCL_CTK_AT_LEAST(12, 4)
      STATIC_REQUIRE(::cudaMemHandleTypeFabric == 0x8);
      STATIC_REQUIRE(::cudaMemHandleTypeFabric == attributes::memory_pool_supported_handle_types.fabric);
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 4) ^^^ / vvv _CCCL_CTK_BELOW(12, 4) vvv
      STATIC_REQUIRE(0x8 == attributes::memory_pool_supported_handle_types.fabric);
#endif // ^^^ _CCCL_CTK_BELOW(12, 4) ^^^

      constexpr int all_handle_types =
        attributes::memory_pool_supported_handle_types.none
        | attributes::memory_pool_supported_handle_types.posix_file_descriptor
        | attributes::memory_pool_supported_handle_types.win32
        | attributes::memory_pool_supported_handle_types.win32_kmt
        | attributes::memory_pool_supported_handle_types.fabric;
      auto handle_types = device_ref(0).attribute(attributes::memory_pool_supported_handle_types);
      CCCLRT_REQUIRE(static_cast<int>(handle_types) <= static_cast<int>(all_handle_types));
    }

#if _CCCL_CTK_AT_LEAST(12, 2)
    SECTION("numa_config")
    {
      STATIC_REQUIRE(::cudaDeviceNumaConfigNone == attributes::numa_config.none);
      STATIC_REQUIRE(::cudaDeviceNumaConfigNumaNode == attributes::numa_config.numa_node);

      auto config = device_ref(0).attribute(attributes::numa_config);
      CCCLRT_REQUIRE((config == attributes::numa_config.none || //
                      config == attributes::numa_config.numa_node));
    }
#endif // _CCCL_CTK_AT_LEAST(12, 2)

    SECTION("Compute capability")
    {
      cuda::compute_capability compute_cap = device_ref(0).attribute(attributes::compute_capability);
      int compute_cap_major                = device_ref(0).attribute(attributes::compute_capability_major);
      int compute_cap_minor                = device_ref(0).attribute(attributes::compute_capability_minor);
      CCCLRT_REQUIRE(compute_cap.get() == 10 * compute_cap_major + compute_cap_minor);
    }
  }
  SECTION("Name")
  {
    const auto name = device_ref(0).name();
    CCCLRT_REQUIRE(name.length() != 0);
    CCCLRT_REQUIRE(name[0] != 0);
  }
}

C2H_CCCLRT_TEST("global devices vector", "[device]")
{
  CCCLRT_REQUIRE(cuda::devices.size() > 0);
  CCCLRT_REQUIRE(cuda::devices.begin() != cuda::devices.end());
  CCCLRT_REQUIRE(cuda::devices.begin() == cuda::devices.begin());
  CCCLRT_REQUIRE(cuda::devices.end() == cuda::devices.end());
  CCCLRT_REQUIRE(cuda::devices.size() == static_cast<size_t>(cuda::devices.end() - cuda::devices.begin()));

  CCCLRT_REQUIRE(0 == cuda::devices[0].get());
  CCCLRT_REQUIRE(cuda::device_ref{0} == cuda::devices[0]);

  CCCLRT_REQUIRE(0 == (*cuda::devices.begin()).get());
  CCCLRT_REQUIRE(cuda::device_ref{0} == *cuda::devices.begin());

  CCCLRT_REQUIRE(0 == cuda::devices.begin()->get());
  CCCLRT_REQUIRE(0 == cuda::devices.begin()[0].get());

  if (cuda::devices.size() > 1)
  {
    CCCLRT_REQUIRE(1 == cuda::devices[1].get());
    CCCLRT_REQUIRE(cuda::device_ref{0} != cuda::devices[1].get());

    CCCLRT_REQUIRE(1 == (*std::next(cuda::devices.begin())).get());
    CCCLRT_REQUIRE(1 == std::next(cuda::devices.begin())->get());
    CCCLRT_REQUIRE(1 == cuda::devices.begin()[1].get());

    CCCLRT_REQUIRE(cuda::devices.size() - 1 == static_cast<std::size_t>((*std::prev(cuda::devices.end())).get()));
    CCCLRT_REQUIRE(cuda::devices.size() - 1 == static_cast<std::size_t>(std::prev(cuda::devices.end())->get()));
    CCCLRT_REQUIRE(cuda::devices.size() - 1 == static_cast<std::size_t>(cuda::devices.end()[-1].get()));

    auto peers = cuda::devices[0].peers();
    for (auto peer : peers)
    {
      CCCLRT_REQUIRE(cuda::devices[0].has_peer_access_to(peer));
      CCCLRT_REQUIRE(peer.has_peer_access_to(cuda::devices[0]));
    }
  }

#if _CCCL_HAS_EXCEPTIONS()
  try
  {
    [[maybe_unused]] const cuda::device_ref& dev = cuda::devices[cuda::devices.size()];
    CCCLRT_REQUIRE(false); // should not get here
  }
  catch (const std::out_of_range&)
  {
    CCCLRT_REQUIRE(true); // expected
  }
#endif // _CCCL_HAS_EXCEPTIONS()
}

C2H_CCCLRT_TEST("memory location", "[device]")
{
  cuda::memory_location loc = cuda::devices[0];
  CCCLRT_REQUIRE(loc.type == ::cudaMemLocationTypeDevice);
  CCCLRT_REQUIRE(loc.id == 0);

  if (cuda::devices.size() > 1)
  {
    loc = cuda::device_ref{1};
    CCCLRT_REQUIRE(loc.type == ::cudaMemLocationTypeDevice);
    CCCLRT_REQUIRE(loc.id == 1);
  }
}
