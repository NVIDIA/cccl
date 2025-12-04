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

__global__ void test_stmatrix(void** fn_ptr)
{
#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // stmatrix.sync.aligned.m8n8.x1.shared.b16 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int16_t*, const cuda::std::uint32_t (&)[1])>(cuda::ptx::stmatrix_m8n8));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // stmatrix.sync.aligned.m8n8.x2.shared.b16 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int16_t*, const cuda::std::uint32_t (&)[2])>(cuda::ptx::stmatrix_m8n8));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // stmatrix.sync.aligned.m8n8.x4.shared.b16 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int16_t*, const cuda::std::uint32_t (&)[4])>(cuda::ptx::stmatrix_m8n8));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [gmem_ptr], input;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int16_t*, const cuda::std::uint32_t (&)[1])>(
            cuda::ptx::stmatrix_m8n8_trans));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [gmem_ptr], input;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int16_t*, const cuda::std::uint32_t (&)[2])>(
            cuda::ptx::stmatrix_m8n8_trans));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 780
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [gmem_ptr], input;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int16_t*, const cuda::std::uint32_t (&)[4])>(
            cuda::ptx::stmatrix_m8n8_trans));));
#endif // __cccl_ptx_isa >= 780

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[1])>(
          cuda::ptx::stmatrix_m16n8_trans));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[2])>(
          cuda::ptx::stmatrix_m16n8_trans));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103a,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110a,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120a,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121a,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_100f,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_103f,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_110f,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_120f,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_121f,
    (
        // stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [gmem_ptr], input;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::std::int8_t*, const cuda::std::uint32_t (&)[4])>(
          cuda::ptx::stmatrix_m16n8_trans));));
#endif // __cccl_ptx_isa >= 860
}
