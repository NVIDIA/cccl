// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_PREFETCH_H_
#define _CUDA_PTX_GENERATED_PREFETCH_H_

/*
// prefetch.global.L1 [addr]; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline void prefetch_L1(
  const void* addr);
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L1(const void* __addr)
{
  asm volatile("prefetch.global.L1 [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 200

/*
// prefetch.global.L2 [addr]; // PTX ISA 20, SM_50
template <typename = void>
__device__ static inline void prefetch_L2(
  const void* addr);
*/
#if __cccl_ptx_isa >= 200
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L2(const void* __addr)
{
  asm volatile("prefetch.global.L2 [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 200

/*
// prefetch.global.L1::32B.valid_addr [addr]; // PTX ISA 94, SM_90
template <typename = void>
__device__ static inline void prefetch_L1_32B(
  const void* addr);
*/
#if __cccl_ptx_isa >= 940
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L1_32B(const void* __addr)
{
  asm volatile("prefetch.global.L1::32B.valid_addr [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 940

/*
// prefetch.global.L2::evict_last [addr]; // PTX ISA 74, SM_80
template <typename = void>
__device__ static inline void prefetch_L2_evict_last(
  const void* addr);
*/
#if __cccl_ptx_isa >= 740
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L2_evict_last(const void* __addr)
{
  asm volatile("prefetch.global.L2::evict_last [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 740

/*
// prefetch.global.L2::evict_normal [addr]; // PTX ISA 74, SM_80
template <typename = void>
__device__ static inline void prefetch_L2_evict_normal(
  const void* addr);
*/
#if __cccl_ptx_isa >= 740
template <typename = void>
_CCCL_DEVICE static inline void prefetch_L2_evict_normal(const void* __addr)
{
  asm volatile("prefetch.global.L2::evict_normal [%0];" : : "l"(__as_ptr_gmem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 740

/*
// prefetch.tensormap [addr]; // PTX ISA 80, SM_90
template <typename = void>
__device__ static inline void prefetch_tensormap(
  const void* addr);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE static inline void prefetch_tensormap(const void* __addr)
{
  asm volatile("prefetch.tensormap [%0];" : : "l"(__addr) : "memory");
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_PREFETCH_H_
