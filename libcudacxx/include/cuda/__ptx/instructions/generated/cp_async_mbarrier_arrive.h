// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_H_
#define _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_H_

/*
// cp.async.mbarrier.arrive.b64 [addr]; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline void cp_async_mbarrier_arrive(
  uint64_t* addr);
*/
#if __cccl_ptx_isa >= 700
template <typename = void>
_CCCL_DEVICE_API void cp_async_mbarrier_arrive(::cuda::std::uint64_t* __addr)
{
  asm("cp.async.mbarrier.arrive.b64 [%0];" : : "r"(__as_ptr_smem(__addr)) : "memory");
}
#endif // __cccl_ptx_isa >= 700

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_CP_ASYNC_MBARRIER_ARRIVE_H_
