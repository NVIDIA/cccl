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

__global__ void test_tcgen05_fence(void** fn_ptr)
{
#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.fence::before_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_before_thread_sync));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.fence::before_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_before_thread_sync));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.fence::before_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_before_thread_sync));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.fence::before_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_before_thread_sync));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.fence::before_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_before_thread_sync));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.fence::before_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_before_thread_sync));));
#endif // __cccl_ptx_isa >= 860

#if __cccl_ptx_isa >= 860
  NV_DISPATCH_TARGET(
    NV_HAS_FEATURE_SM_100a,
    (
        // tcgen05.fence::after_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_after_thread_sync));),
    NV_HAS_FEATURE_SM_103a,
    (
        // tcgen05.fence::after_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_after_thread_sync));),
    NV_HAS_FEATURE_SM_110a,
    (
        // tcgen05.fence::after_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_after_thread_sync));),
    NV_HAS_FEATURE_SM_100f,
    (
        // tcgen05.fence::after_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_after_thread_sync));),
    NV_HAS_FEATURE_SM_103f,
    (
        // tcgen05.fence::after_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_after_thread_sync));),
    NV_HAS_FEATURE_SM_110f,
    (
        // tcgen05.fence::after_thread_sync;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)()>(cuda::ptx::tcgen05_fence_after_thread_sync));));
#endif // __cccl_ptx_isa >= 860
}
