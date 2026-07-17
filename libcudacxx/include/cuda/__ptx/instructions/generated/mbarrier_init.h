// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_MBARRIER_INIT_H_
#define _CUDA_PTX_GENERATED_MBARRIER_INIT_H_

/*
// mbarrier.init.shared.b64 [addr], count; // PTX ISA 70, SM_80
template <typename = void>
__device__ static inline void mbarrier_init(
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 700
template <typename = void>
_CCCL_DEVICE static inline void mbarrier_init(::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __count)
{
  asm("mbarrier.init.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(__addr)), "r"(__count) : "memory");
}
#endif // __cccl_ptx_isa >= 700

/*
// mbarrier.init.layout.shared.b64 [addr], count; // PTX ISA 94, SM_90
// .layout    = { .layout::v0, .layout::v1 }
template <cuda::ptx::dot_layout Layout>
__device__ static inline void mbarrier_init(
  cuda::ptx::layout_t<Layout> layout,
  uint64_t* addr,
  const uint32_t& count);
*/
#if __cccl_ptx_isa >= 940
template <::cuda::ptx::dot_layout _Layout>
_CCCL_DEVICE static inline void mbarrier_init(
  ::cuda::ptx::layout_t<_Layout> __layout, ::cuda::std::uint64_t* __addr, const ::cuda::std::uint32_t& __count)
{
  static_assert(__layout == layout_v0 || __layout == layout_v1, "");
  if constexpr (__layout == layout_v0)
  {
    asm("mbarrier.init.layout::v0.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(__addr)), "r"(__count) : "memory");
  }
  else if constexpr (__layout == layout_v1)
  {
    asm("mbarrier.init.layout::v1.shared.b64 [%0], %1;" : : "r"(__as_ptr_smem(__addr)), "r"(__count) : "memory");
  }
}
#endif // __cccl_ptx_isa >= 940

#endif // _CUDA_PTX_GENERATED_MBARRIER_INIT_H_
