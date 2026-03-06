// This file was automatically generated. Do not edit.

// We use a special strategy to force the generation of the PTX. This is mainly
// a fight against dead-code-elimination in the NVVM layer.
//
// The reason we need this strategy is because certain older versions of ptxas
// segfault when a non-sensical sequence of PTX is generated. So instead, we try
// to force the instantiation and compilation to PTX of all the overloads of the
// PTX wrapping functions.
//
// We do this by writing a function pointer of each overload to the kernel
// parameter `fn_ptr`.
//
// Because `fn_ptr` is possibly visible outside this translation unit, the
// compiler must compile all the functions which are stored.

__global__ void test_ldmatrix_m16n16_trans(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const cuda::std::int8_t*)>(
            cuda::ptx::ldmatrix_m16n16_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const cuda::std::int8_t*)>(
            cuda::ptx::ldmatrix_m16n16_trans));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b6x16_p32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b4x16_p64));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const cuda::std::int8_t*)>(
            cuda::ptx::ldmatrix_m16n16_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const cuda::std::int8_t*)>(
            cuda::ptx::ldmatrix_m16n16_trans));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b6x16_p32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b4x16_p64));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m16n16_trans_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860
}
