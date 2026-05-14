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

__global__ void test_tensormap_query(void** fn_ptr)
{
#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(NV_HAS_FEATURE_SM_90a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_100a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_103a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));

  NV_IF_TARGET(NV_HAS_FEATURE_SM_107a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));

  NV_IF_TARGET(NV_HAS_FEATURE_SM_110a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_121a,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));

  NV_IF_TARGET(NV_HAS_FEATURE_SM_100f,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_103f,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));

  NV_IF_TARGET(NV_HAS_FEATURE_SM_107f,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));

  NV_IF_TARGET(NV_HAS_FEATURE_SM_110f,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_120f,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));
  NV_IF_TARGET(NV_HAS_FEATURE_SM_121f,
               (
                   // tensormap.query.is_large.u32 outPred, tensorSize;
                   * fn_ptr++ = reinterpret_cast<void*>(
                     static_cast<bool (*)(cuda::std::uint32_t)>(cuda::ptx::tensormap_query_is_large));));

#endif // __cccl_ptx_isa >= 940
}
