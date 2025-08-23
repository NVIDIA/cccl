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

__global__ void test_st_async(void** fn_ptr)
{
#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b32 [addr], value, [remote_bar];    // 1.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t*, const cuda::std::int32_t&, cuda::std::uint64_t*)>(
            cuda::ptx::st_async));
          // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.b64 [addr], value, [remote_bar];    // 1.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::std::int64_t*, const cuda::std::int64_t&, cuda::std::uint64_t*)>(
                cuda::ptx::st_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b32 [addr], value, [remote_bar]; // 2.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t*, const cuda::std::int32_t (&)[2], cuda::std::uint64_t*)>(
            cuda::ptx::st_async));
          // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v2.b64 [addr], value, [remote_bar]; // 2.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::std::int64_t*, const cuda::std::int64_t (&)[2], cuda::std::uint64_t*)>(
                cuda::ptx::st_async));));
#endif // __cccl_ptx_isa >= 810

#if __cccl_ptx_isa >= 810
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [addr], value, [remote_bar];    // 3.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::std::int32_t*, const cuda::std::int32_t (&)[4], cuda::std::uint64_t*)>(
            cuda::ptx::st_async));));
#endif // __cccl_ptx_isa >= 810
}
