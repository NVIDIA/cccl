/*
// fence{.sem}.scope; // 1. PTX ISA 60, SM_70
// .sem       = { .sc, .acq_rel }
// .scope     = { .cta, .gpu, .sys }
template <cuda::ptx::dot_sem Sem, cuda::ptx::dot_scope Scope>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_t<Scope> scope);
*/
#if __cccl_ptx_isa >= 600
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_70__();
template <dot_sem _Sem, dot_scope _Scope>
_CCCL_DEVICE static inline void fence(sem_t<_Sem> __sem, scope_t<_Scope> __scope)
{
  static_assert(__sem == sem_sc || __sem == sem_acq_rel, "");
  static_assert(__scope == scope_cta || __scope == scope_gpu || __scope == scope_sys, "");
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_70,
    (
      _CCCL_IF_CONSTEXPR (__sem == sem_sc && __scope == scope_cta) {
        asm volatile("fence.sc.cta; // 1." : : : "memory");
      } _CCCL_ELSE_IF_CONSTEXPR (__sem == sem_sc && __scope == scope_gpu) {
        asm volatile("fence.sc.gpu; // 1." : : : "memory");
      } _CCCL_ELSE_IF_CONSTEXPR (__sem == sem_sc && __scope == scope_sys) {
        asm volatile("fence.sc.sys; // 1." : : : "memory");
      } _CCCL_ELSE_IF_CONSTEXPR (__sem == sem_acq_rel && __scope == scope_cta) {
        asm volatile("fence.acq_rel.cta; // 1." : : : "memory");
      } _CCCL_ELSE_IF_CONSTEXPR (__sem == sem_acq_rel && __scope == scope_gpu) {
        asm volatile("fence.acq_rel.gpu; // 1." : : : "memory");
      } _CCCL_ELSE_IF_CONSTEXPR (__sem == sem_acq_rel && __scope == scope_sys) {
        asm volatile("fence.acq_rel.sys; // 1." : : : "memory");
      }),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_fence_is_not_supported_before_SM_70__();));
}
#endif // __cccl_ptx_isa >= 600

/*
// fence{.sem}.scope; // 2. PTX ISA 78, SM_90
// .sem       = { .sc, .acq_rel }
// .scope     = { .cluster }
template <cuda::ptx::dot_sem Sem>
__device__ static inline void fence(
  cuda::ptx::sem_t<Sem> sem,
  cuda::ptx::scope_cluster_t);
*/
#if __cccl_ptx_isa >= 780
extern "C" _CCCL_DEVICE void __cuda_ptx_fence_is_not_supported_before_SM_90__();
template <dot_sem _Sem>
_CCCL_DEVICE static inline void fence(sem_t<_Sem> __sem, scope_cluster_t)
{
  static_assert(__sem == sem_sc || __sem == sem_acq_rel, "");
  // __scope == scope_cluster (due to parameter type constraint)
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (
      _CCCL_IF_CONSTEXPR (__sem == sem_sc) {
        asm volatile("fence.sc.cluster; // 2." : : : "memory");
      } _CCCL_ELSE_IF_CONSTEXPR (__sem == sem_acq_rel) {
        asm volatile("fence.acq_rel.cluster; // 2." : : : "memory");
      }),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_fence_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 780
