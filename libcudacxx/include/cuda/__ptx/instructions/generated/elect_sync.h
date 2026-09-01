// This file was automatically generated. Do not edit.

// clang-tidy does not distinguish generated PTX constraints or inline-assembly branch bodies.
// NOLINTBEGIN(modernize-unary-static-assert, bugprone-branch-clone)

#ifndef _CUDA_PTX_GENERATED_ELECT_SYNC_H_
#define _CUDA_PTX_GENERATED_ELECT_SYNC_H_

/*
// elect.sync _|is_elected, membermask; // PTX ISA 80, SM_90
template <typename = void>
__device__ static inline bool elect_sync(
  const uint32_t& membermask);
*/
#if __cccl_ptx_isa >= 800
template <typename = void>
_CCCL_DEVICE_API bool elect_sync(const ::cuda::std::uint32_t& __membermask)
{
  ::cuda::std::uint32_t __is_elected;
  asm volatile(
    "{\n\t"
    ".reg .pred P_OUT; \n\t"
    "elect.sync _|P_OUT, %1; \n\t"
    "selp.b32 %0, 1, 0, P_OUT; \n"
    "}"
    : "=r"(__is_elected)
    : "r"(__membermask)
    :);
  return static_cast<bool>(__is_elected);
}
#endif // __cccl_ptx_isa >= 800

// NOLINTEND(modernize-unary-static-assert, bugprone-branch-clone)

#endif // _CUDA_PTX_GENERATED_ELECT_SYNC_H_
