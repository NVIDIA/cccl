/*
// cp.async.bulk.wait_group N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group(
  cuda::ptx::n32_t<N32> N);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_wait_group_is_not_supported_before_SM_90__();
template <int _N32>
_CCCL_DEVICE static inline void cp_async_bulk_wait_group(n32_t<_N32> __n)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("cp.async.bulk.wait_group %0;"
                  :
                  : "n"(__n.value)
                  : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_wait_group_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800

/*
// cp.async.bulk.wait_group.read N; // PTX ISA 80, SM_90
template <int N32>
__device__ static inline void cp_async_bulk_wait_group_read(
  cuda::ptx::n32_t<N32> N);
*/
#if __cccl_ptx_isa >= 800
extern "C" _CCCL_DEVICE void __cuda_ptx_cp_async_bulk_wait_group_read_is_not_supported_before_SM_90__();
template <int _N32>
_CCCL_DEVICE static inline void cp_async_bulk_wait_group_read(n32_t<_N32> __n)
{
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_90,
    (asm volatile("cp.async.bulk.wait_group.read %0;"
                  :
                  : "n"(__n.value)
                  : "memory");),
    (
      // Unsupported architectures will have a linker error with a semi-decent error message
      __cuda_ptx_cp_async_bulk_wait_group_read_is_not_supported_before_SM_90__();));
}
#endif // __cccl_ptx_isa >= 800
