// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_
#define _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_

/*
// tcgen05.wait::ld.sync.aligned; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
template <typename = void>
__device__ static inline void tcgen05_wait_ld();
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE_API void tcgen05_wait_ld()
{
  asm volatile("tcgen05.wait::ld.sync.aligned;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// tcgen05.wait::st.sync.aligned; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
template <typename = void>
__device__ static inline void tcgen05_wait_st();
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE_API void tcgen05_wait_st()
{
  asm volatile("tcgen05.wait::st.sync.aligned;" : : : "memory");
}
#endif // __cccl_ptx_isa >= 860

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_TCGEN05_WAIT_H_
