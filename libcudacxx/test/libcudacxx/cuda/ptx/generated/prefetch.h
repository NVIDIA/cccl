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

__global__ void test_prefetch(void** fn_ptr)
{
#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // prefetch.global.L1 [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(const void*)>(cuda::ptx::prefetch_L1));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 200
  NV_IF_TARGET(NV_PROVIDES_SM_50,
               (
                   // prefetch.global.L2 [addr];
                   * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(const void*)>(cuda::ptx::prefetch_L2));));
#endif // __cccl_ptx_isa >= 200

#if __cccl_ptx_isa >= 940
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // prefetch.global.L1::32B.valid_addr [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(const void*)>(cuda::ptx::prefetch_L1_32B));));
#endif // __cccl_ptx_isa >= 940

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // prefetch.global.L2::evict_last [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(const void*)>(cuda::ptx::prefetch_L2_evict_last));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 740
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (
        // prefetch.global.L2::evict_normal [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(const void*)>(cuda::ptx::prefetch_L2_evict_normal));));
#endif // __cccl_ptx_isa >= 740

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // prefetch.tensormap [addr];
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(const void*)>(cuda::ptx::prefetch_tensormap));));
#endif // __cccl_ptx_isa >= 800
}
