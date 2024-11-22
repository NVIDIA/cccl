__global__ void test_tensormap_cp_fenceproxy(void** fn_ptr)
{
#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cta.sync.aligned  [dst], [src], size;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(
            cuda::ptx::sem_release_t, cuda::ptx::scope_cta_t, void*, const void*, cuda::ptx::n32_t<128>)>(
            cuda::ptx::tensormap_cp_fenceproxy));
          // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.cluster.sync.aligned  [dst], [src],
          // size;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_cluster_t, void*, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::tensormap_cp_fenceproxy));
          // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.gpu.sync.aligned  [dst], [src], size;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_gpu_t, void*, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::tensormap_cp_fenceproxy));
          // tensormap.cp_fenceproxy.global.shared::cta.tensormap::generic.release.sys.sync.aligned  [dst], [src], size;
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(
                cuda::ptx::sem_release_t, cuda::ptx::scope_sys_t, void*, const void*, cuda::ptx::n32_t<128>)>(
                cuda::ptx::tensormap_cp_fenceproxy));));
#endif // __cccl_ptx_isa >= 830
}
