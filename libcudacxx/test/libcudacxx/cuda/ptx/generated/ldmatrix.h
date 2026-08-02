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

__global__ void test_ldmatrix(void** fn_ptr)
{
#if __cccl_ptx_isa >= 650
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ldmatrix.sync.aligned.m8n8.x1.shared.b16 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const cuda::std::int16_t*)>(
            cuda::ptx::ldmatrix_m8n8));));
#endif // __cccl_ptx_isa >= 650

#if __cccl_ptx_isa >= 650
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ldmatrix.sync.aligned.m8n8.x2.shared.b16 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const cuda::std::int16_t*)>(
            cuda::ptx::ldmatrix_m8n8));));
#endif // __cccl_ptx_isa >= 650

#if __cccl_ptx_isa >= 650
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ldmatrix.sync.aligned.m8n8.x4.shared.b16 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const cuda::std::int16_t*)>(
            cuda::ptx::ldmatrix_m8n8));));
#endif // __cccl_ptx_isa >= 650

#if __cccl_ptx_isa >= 650
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const cuda::std::int16_t*)>(
            cuda::ptx::ldmatrix_m8n8_trans));));
#endif // __cccl_ptx_isa >= 650

#if __cccl_ptx_isa >= 650
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const cuda::std::int16_t*)>(
            cuda::ptx::ldmatrix_m8n8_trans));));
#endif // __cccl_ptx_isa >= 650

#if __cccl_ptx_isa >= 650
  NV_IF_TARGET(
    NV_PROVIDES_SM_75,
    (
        // ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const cuda::std::int16_t*)>(
            cuda::ptx::ldmatrix_m8n8_trans));));
#endif // __cccl_ptx_isa >= 650

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b6x16_p32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b4x16_p64));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m8n16.x1.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[1], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b6x16_p32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b4x16_p64));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m8n16.x2.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[2], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b6x16_p32));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b6x16_p32 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b6x16_p32));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b4x16_p64));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // ldmatrix.sync.aligned.m8n16.x4.shared.b8x16.b4x16_p64 out, [smem_ptr];
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, cuda::std::uint32_t (&out)[4], const void*)>(
            cuda::ptx::ldmatrix_m8n16_b8x16_b4x16_p64));));
#endif // __cccl_ptx_isa >= 860
}
