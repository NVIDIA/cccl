// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_cp_async_bulk_prefetch_tensor(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[1], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[2], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[3], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[4], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [tensorMap, tensorCoords],
        // cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4));));

#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[1], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[1], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[1])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[1],
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[1],
                               const cuda::std::int32_t (&)[1],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[1],
                               const cuda::std::int32_t (&)[1])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[1],
                               const cuda::std::int32_t (&)[1])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[2], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[2], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[2])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[2],
                               const cuda::std::int32_t (&)[1],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[2],
                               const cuda::std::int32_t (&)[1],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[2],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[2],
                               const cuda::std::int32_t (&)[1],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[2])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[2],
                               const cuda::std::int32_t (&)[1],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[2])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[3])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[3])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[3], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[3], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[3])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[3])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[3],
                               const cuda::std::int32_t (&)[2],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[3],
                               const cuda::std::int32_t (&)[2],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[3],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[3],
                               const cuda::std::int32_t (&)[2],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[3])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[3],
                               const cuda::std::int32_t (&)[2],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[3])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[4], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[4], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[4])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[4],
                               const cuda::std::int32_t (&)[3],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[4],
                               const cuda::std::int32_t (&)[3],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[4],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[4],
                               const cuda::std::int32_t (&)[3],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[4])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[4],
                               const cuda::std::int32_t (&)[3],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[4])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[5],
                               const cuda::std::int32_t (&)[4],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[5],
                               const cuda::std::int32_t (&)[4],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[5],
                               cuda::std::uint64_t)>(cuda::ptx::cp_async_bulk_prefetch_tensor_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[5],
                               const cuda::std::int32_t (&)[4],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_stride
        // [tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
        // tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               const void*,
                               const void*,
                               const cuda::std::int16_t (&)[5],
                               const cuda::std::int32_t (&)[4],
                               const cuda::std::int16_t&,
                               const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last [tensorMap, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords], cache_policy;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5], cuda::std::uint64_t)>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4_override));));

#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107a,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last_override));));

  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_107f,
    (
        // cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last.override::global_address [tensorMap,
        // gAddrToOverride, tensorCoords];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, const void*, const void*, const cuda::std::int32_t (&)[5])>(
            cuda::ptx::cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last_override));));

#endif // __cccl_ptx_isa >= 940
}
