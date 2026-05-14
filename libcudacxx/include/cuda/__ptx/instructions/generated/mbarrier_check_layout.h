// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_CHECK_LAYOUT_H_
#define _CUDA_PTX_GENERATED_MBARRIER_CHECK_LAYOUT_H_

/*
// mbarrier.check_layout.layout.shared::cta.b64 p, [addr]; // PTX ISA 94, SM_90
// .layout    = { .layout::v0, .layout::v1 }
template <cuda::ptx::dot_layout Layout>
__device__ static inline bool mbarrier_check_layout(
  cuda::ptx::layout_t<Layout> layout,
  const uint64_t* addr);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_layout _Layout>
_CCCL_DEVICE static inline bool
mbarrier_check_layout(::cuda::ptx::layout_t<_Layout> __layout, const ::cuda::std::uint64_t* __addr)
{
  static_assert(__layout == layout_v0 || __layout == layout_v1, "");
  ::cuda::std::uint32_t __p;
  if constexpr (__layout == layout_v0)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.check_layout.layout::v0.shared::cta.b64 P_OUT, [%1]; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__p)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  else if constexpr (__layout == layout_v1)
  {
    asm("{\n\t"
        ".reg .pred P_OUT; \n\t"
        "mbarrier.check_layout.layout::v1.shared::cta.b64 P_OUT, [%1]; \n\t"
        "selp.b32 %0, 1, 0, P_OUT; \n"
        "}"
        : "=r"(__p)
        : "r"(__as_ptr_smem(__addr))
        : "memory");
  }
  return static_cast<bool>(__p);
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_MBARRIER_CHECK_LAYOUT_H_
