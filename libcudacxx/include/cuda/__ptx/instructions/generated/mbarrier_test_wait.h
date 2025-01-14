// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_TEST_WAIT_H_
#define _CUDA_PTX_GENERATED_MBARRIER_TEST_WAIT_H_

/*
// mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1. PTX
ISA 70, SM_80 template <typename = void>
__device__ static inline bool mbarrier_test_wait(
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 700
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_80__();
template <typename = void>
_CCCL_DEVICE static inline bool mbarrier_test_wait(_CUDA_VSTD::uint64_t* __addr, const _CUDA_VSTD::uint64_t& __state)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_80,
    (_CUDA_VSTD::uint32_t __waitComplete;
     asm("{\n\t .reg .pred P_OUT; \n\t"
         "mbarrier.test_wait.shared.b64 P_OUT, [%1], %2;                                                  // 1. \n\t"
         "selp.b32 %0, 1, 0, P_OUT; \n"
         "}" : "=r"(__waitComplete) : "r"(__as_ptr_smem(__addr)),
         "l"(__state) : "memory");
     return static_cast<bool>(__waitComplete);),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_80__(); return false;));
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.test_wait{.sem}{.scope}.shared::cta.b64        waitComplete, [addr], state;                        // 2. PTX
ISA 80, SM_90
// .sem       = { .acquire }
// .scope     = { .cta, .cluster }
template <cuda::ptx::dot_scope Scope>
__device__ static inline bool mbarrier_test_wait(
  cuda::ptx::sem_acquire_t,
  cuda::ptx::scope_t<Scope> scope,
  uint64_t* addr,
  const uint64_t& state);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_90__();
template <dot_scope _Scope>
_CCCL_DEVICE static inline bool mbarrier_test_wait(
  sem_acquire_t, scope_t<_Scope> __scope, _CUDA_VSTD::uint64_t* __addr, const _CUDA_VSTD::uint64_t& __state)
{
  // __sem == sem_acquire (due to parameter type constraint)
  static_assert(__scope == scope_cta || __scope == scope_cluster, "");
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (
      _CUDA_VSTD::uint32_t __waitComplete; _CCCL_IF_CONSTEXPR (__scope == scope_cta) {
        asm("{\n\t .reg .pred P_OUT; \n\t"
            "mbarrier.test_wait.acquire.cta.shared::cta.b64        P_OUT, [%1], %2;                        // 2.  \n\t"
            "selp.b32 %0, 1, 0, P_OUT; \n"
            "}"
            : "=r"(__waitComplete)
            : "r"(__as_ptr_smem(__addr)), "l"(__state)
            : "memory");
      } else _CCCL_IF_CONSTEXPR (__scope == scope_cluster) {
        asm("{\n\t .reg .pred P_OUT; \n\t"
            "mbarrier.test_wait.acquire.cluster.shared::cta.b64        P_OUT, [%1], %2;                        // 2.  "
            "\n\t"
            "selp.b32 %0, 1, 0, P_OUT; \n"
            "}"
            : "=r"(__waitComplete)
            : "r"(__as_ptr_smem(__addr)), "l"(__state)
            : "memory");
      } return static_cast<bool>(__waitComplete);),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_90__(); return false;));
}
#endif // __cccl_ptx_isa >= 800

#endif // _CUDA_PTX_GENERATED_MBARRIER_TEST_WAIT_H_
