// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_BULK_PREFETCH_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_BULK_PREFETCH_H_

/*
// cp.async.bulk.prefetch.L2.global.L2::cache_hint [srcMem], size, cache_policy; // PTX ISA 80, SM_90
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch(
  const void* srcMem,
  uint32_t size,
  uint64_t cache_policy = 0x10F0000000000000);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch(
  const void* __srcMem, ::cuda::std::uint32_t __size, ::cuda::std::uint64_t __cache_policy = 0x10F0000000000000)
{
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::cache_hint [%0], %1, %2;"
               :
               : "l"(__as_ptr_gmem(__srcMem)), "r"(__size), "l"(__cache_policy)
               : "memory");
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.prefetch.L2.global.L2::evict_last [srcMem], size; // PTX ISA 94, SM_107a, SM_107f
template <typename = void>
__device__ static inline void cp_async_bulk_prefetch_L2_evict_last(
  const void* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void cp_async_bulk_prefetch_L2_evict_last(const void* __srcMem, ::cuda::std::uint32_t __size)
{
  asm volatile("cp.async.bulk.prefetch.L2.global.L2::evict_last [%0], %1;"
               :
               : "l"(__as_ptr_gmem(__srcMem)), "r"(__size)
               : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_BULK_PREFETCH_H_
