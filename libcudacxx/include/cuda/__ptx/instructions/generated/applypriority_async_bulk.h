// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_APPLYPRIORITY_ASYNC_BULK_H_
#define _CUDA_PTX_GENERATED_APPLYPRIORITY_ASYNC_BULK_H_

/*
// applypriority.async.bulk.global.bulk_group.L2::evict_normal [srcMem], size; // PTX ISA 94, SM_107a, SM_107f
template <typename = void>
__device__ static inline void applypriority_async_bulk_L2_evict_normal(
  void* srcMem,
  uint32_t size);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void applypriority_async_bulk_L2_evict_normal(void* __srcMem, ::cuda::std::uint32_t __size)
{
  asm volatile("applypriority.async.bulk.global.bulk_group.L2::evict_normal [%0], %1;"
               :
               : "l"(__as_ptr_gmem(__srcMem)), "r"(__size)
               : "memory");
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_APPLYPRIORITY_ASYNC_BULK_H_
