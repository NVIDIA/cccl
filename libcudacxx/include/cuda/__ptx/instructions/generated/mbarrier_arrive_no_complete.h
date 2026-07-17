// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_NO_COMPLETE_H_
#define _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_NO_COMPLETE_H_

/*
// mbarrier.arrive.noComplete.shared.b64                       state,  [addr], count;    // 5.  PTX ISA 70, SM_80
template <typename = void>
__device__ static inline uint64_t mbarrier_arrive_no_complete(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
template <typename = void>
_CCCL_DEVICE static inline ::cuda::std::uint64_t
mbarrier_arrive_no_complete(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __count)
{
  ::cuda::std::uint64_t __state;
  asm("mbarrier.arrive.noComplete.shared.b64                       %0,  [%1], %2;    // 5. "
      : "=l"(__state)
      : "r"(__as_ptr_smem(__addr)), "r"(__count)
      : "memory");
  return __state;
}
#endif // __cccl_ptx_isa >= 700

#endif // _CUDA_PTX_GENERATED_MBARRIER_ARRIVE_NO_COMPLETE_H_
