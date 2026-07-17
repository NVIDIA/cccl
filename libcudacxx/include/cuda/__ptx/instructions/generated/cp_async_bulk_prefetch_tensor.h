// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_PREFETCH_TENSOR_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_PREFETCH_TENSOR_H_

/*
// cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80,
SM_90
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint [%0, {%1}], %2;"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80,
SM_90
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint [%0, {%1, %2}], %3;"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80,
SM_90
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3}], %4;"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__tensorCoords[2]), "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80,
SM_90
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3, %4}], %5;"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX ISA 80,
SM_90
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::cache_hint [tensorMap, tensorCoords], cache_policy; // PTX
ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_tile_gather4(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint [%0, {%1, %2, %3, %4, %5}], %6;"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[1]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  ::cuda::ptx::space_global_t, const void* __tensorMap, const ::cuda::std::int32_t (&__tensorCoords)[1])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last [%0, {%1}];"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[1],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address [%0, %1, {%2}], %3;"
      :
      : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[1]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[1])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address [%0, %1, {%2}];"
      :
      : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim [tensorMap,
gAddrToOverride, tensorSizeToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim "
      "[%0, %1, {%2}, {%3}], %4;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.1d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim [tensorMap,
gAddrToOverride, tensorSizeToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[1],
  const int32_t (&tensorCoords)[1]);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16, ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[1],
  const ::cuda::std::int32_t (&__tensorCoords)[1])
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.1d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim "
      "[%0, %1, {%2}, {%3}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "r"(__tensorCoords[0])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[2]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  ::cuda::ptx::space_global_t, const void* __tensorMap, const ::cuda::std::int32_t (&__tensorCoords)[2])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last [%0, {%1, %2}];"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[2],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address [%0, %1, {%2, %3}], %4;"
      :
      : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[2]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address [%0, %1, {%2, %3}];"
      :
      : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__tensorCoords[1])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}], %8;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[2],
  const B32 (&tensorLowerStrideToOverride)[1],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[2]);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[2],
  const _B32 (&__tensorLowerStrideToOverride)[1],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[2])
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3}, {%4}, %5, {%6, %7}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[3]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  ::cuda::ptx::space_global_t, const void* __tensorMap, const ::cuda::std::int32_t (&__tensorCoords)[3])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last [%0, {%1, %2, %3}];"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__tensorCoords[2])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[3],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address [%0, %1, {%2, %3, %4}], "
      "%5;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[3]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address [%0, %1, {%2, %3, %4}];"
      :
      : "l"(__tensorMap), "l"(__gAddrToOverride), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__tensorCoords[2])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}], %11;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.3d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[3],
  const B32 (&tensorLowerStrideToOverride)[2],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[3]);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[3],
  const _B32 (&__tensorLowerStrideToOverride)[2],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[3])
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.3d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3, %4}, {%5, %6}, %7, {%8, %9, %10}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[4]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  ::cuda::ptx::space_global_t, const void* __tensorMap, const ::cuda::std::int32_t (&__tensorCoords)[4])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last [%0, {%1, %2, %3, %4}];"
      :
      : "l"(__tensorMap), "r"(__tensorCoords[0]), "r"(__tensorCoords[1]), "r"(__tensorCoords[2]), "r"(__tensorCoords[3])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[4],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address [%0, %1, {%2, %3, %4, "
      "%5}], %6;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[4]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address [%0, %1, {%2, %3, %4, "
      "%5}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}], %14;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.4d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[4],
  const B32 (&tensorLowerStrideToOverride)[3],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[4]);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[4],
  const _B32 (&__tensorLowerStrideToOverride)[3],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[4])
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.4d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3, %4, %5}, {%6, %7, %8}, %9, {%10, %11, %12, %13}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94, SM_107a,
SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last(
  ::cuda::ptx::space_global_t, const void* __tensorMap, const ::cuda::std::int32_t (&__tensorCoords)[5])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last [%0, {%1, %2, %3, %4, %5}];"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::cache_hint.override::global_address [tensorMap, gAddrToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address [%0, %1, {%2, %3, %4, "
      "%5, %6}], %7;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::evict_last.override::global_address [tensorMap, gAddrToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address [%0, %1, {%2, %3, %4, "
      "%5, %6}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::cache_hint.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::cache_hint.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}], %17;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.5d.L2.src.tile.L2::evict_last.override::global_address.override::global_dim_stride
[tensorMap, gAddrToOverride, tensorSizeToOverride, tensorLowerStrideToOverride, tensorUpperStrideToOverride,
tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename B16, enable_if_t<sizeof(B16) == 2, bool> = true, typename B32, enable_if_t<sizeof(B32) == 4, bool> =
true>
__device__ static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const B16 (&tensorSizeToOverride)[5],
  const B32 (&tensorLowerStrideToOverride)[4],
  const B16& tensorUpperStrideToOverride,
  const int32_t (&tensorCoords)[5]);
*/
#if __cccl_ptx_isa >= 940
template <typename _B16,
          ::cuda::std::enable_if_t<sizeof(_B16) == 2, bool> = true,
          typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const _B16 (&__tensorSizeToOverride)[5],
  const _B32 (&__tensorLowerStrideToOverride)[4],
  const _B16& __tensorUpperStrideToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5])
{
  // __space == space_global (due to parameter type constraint)
  static_assert(sizeof(_B16) == 2, "");
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B16) == 2, "");
  asm("cp.async.bulk.prefetch.tensor.5d.L2.global.tile.L2::evict_last.override::global_address.override::global_dim_"
      "stride [%0, %1, {%2, %3, %4, %5, %6}, {%7, %8, %9, %10}, %11, {%12, %13, %14, %15, %16}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[0])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[1])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[2])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorSizeToOverride[4])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[0])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[1])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[2])),
        "r"(/*as_b32*/ *reinterpret_cast<const ::cuda::std::int32_t*>(&__tensorLowerStrideToOverride[3])),
        "h"(/*as_b16*/ *reinterpret_cast<const ::cuda::std::int16_t*>(&__tensorUpperStrideToOverride)),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::evict_last [tensorMap, tensorCoords]; // PTX ISA 94,
SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const int32_t (&tensorCoords)[5]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last(
  ::cuda::ptx::space_global_t, const void* __tensorMap, const ::cuda::std::int32_t (&__tensorCoords)[5])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last [%0, {%1, %2, %3, %4, %5}];"
      :
      : "l"(__tensorMap),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::cache_hint.override::global_address [tensorMap,
gAddrToOverride, tensorCoords], cache_policy; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5],
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_tile_gather4_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5],
  ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::cache_hint.override::global_address [%0, %1, {%2, "
      "%3, %4, %5, %6}], %7;"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4]),
        "l"(__cache_policy)
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// cp.async.bulk.prefetch.tensor.2d.L2.src.tile::gather4.L2::evict_last.override::global_address [tensorMap,
gAddrToOverride, tensorCoords]; // PTX ISA 94, SM_107a, SM_107f
// .src       = { .global }
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last_override(
  cuda::ptx::space_global_t,
  const void* tensorMap,
  const void* gAddrToOverride,
  const int32_t (&tensorCoords)[5]);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_tensor_tile_gather4_L2_evict_last_override(
  ::cuda::ptx::space_global_t,
  const void* __tensorMap,
  const void* __gAddrToOverride,
  const ::cuda::std::int32_t (&__tensorCoords)[5])
{
  // __space == space_global (due to parameter type constraint)
  asm("cp.async.bulk.prefetch.tensor.2d.L2.global.tile::gather4.L2::evict_last.override::global_address [%0, %1, {%2, "
      "%3, %4, %5, %6}];"
      :
      : "l"(__tensorMap),
        "l"(__gAddrToOverride),
        "r"(__tensorCoords[0]),
        "r"(__tensorCoords[1]),
        "r"(__tensorCoords[2]),
        "r"(__tensorCoords[3]),
        "r"(__tensorCoords[4])
      : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_PREFETCH_TENSOR_H_
