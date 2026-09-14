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

__global__ void test_applypriority_async_bulk(void** fn_ptr)
{
#if __cccl_ptx_isa >= 940

  NV_IF_TARGET(NV_HAS_FEATURE_SM_107a,
               (
                   // applypriority.async.bulk.global.bulk_group.L2::evict_normal [srcMem], size;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(void*, cuda::std::uint32_t)>(
                     cuda::ptx::applypriority_async_bulk_L2_evict_normal));));

  NV_IF_TARGET(NV_HAS_FEATURE_SM_107f,
               (
                   // applypriority.async.bulk.global.bulk_group.L2::evict_normal [srcMem], size;
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(void*, cuda::std::uint32_t)>(
                     cuda::ptx::applypriority_async_bulk_L2_evict_normal));));

#endif // __cccl_ptx_isa >= 940
}
