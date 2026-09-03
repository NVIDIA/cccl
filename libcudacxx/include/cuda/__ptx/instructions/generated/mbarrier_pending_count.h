// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_MBARRIER_PENDING_COUNT_H_
#define _CUDA_PTX_GENERATED_MBARRIER_PENDING_COUNT_H_

/*
// mbarrier.pending_count.b64 count, state; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline uint32_t mbarrier_pending_count(
  uint64_t state);
*/
#if __cccl_ptx_isa >= 700
template <typename = void>
_CCCL_DEVICE_API ::cuda::std::uint32_t mbarrier_pending_count(::cuda::std::uint64_t __state)
{
  ::cuda::std::uint32_t __count;
  asm volatile("mbarrier.pending_count.b64 %0, %1;" : "=r"(__count) : "l"(__state) :);
  return __count;
}
#endif // __cccl_ptx_isa >= 700

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_MBARRIER_PENDING_COUNT_H_
