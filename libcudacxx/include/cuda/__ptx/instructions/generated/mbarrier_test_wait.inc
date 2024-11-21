/*
// mbarrier.test_wait.shared.b64 waitComplete, [addr], state;                                                  // 1. PTX
ISA 70, SM_80 template <typename=void>
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
         "}"
         : "=r"(__waitComplete)
         : "r"(__as_ptr_smem(__addr)), "l"(__state)
         : "memory");
     return static_cast<bool>(__waitComplete);),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_mbarrier_test_wait_is_not_supported_before_SM_80__(); return false;));
}
#endif // __cccl_ptx_isa >= 700
