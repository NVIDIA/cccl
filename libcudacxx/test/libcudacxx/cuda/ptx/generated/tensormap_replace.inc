__global__ void test_tensormap_replace(void** fn_ptr)
{
#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_address.global.b1024.b64    [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, int64_t)>(
          cuda::ptx::tensormap_replace_global_address));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_address.shared::cta.b1024.b64    [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, int64_t)>(
          cuda::ptx::tensormap_replace_global_address));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.rank.global.b1024.b32              [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t, void*, int32_t)>(cuda::ptx::tensormap_replace_rank));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.rank.shared::cta.b1024.b32              [tm_addr], new_val;
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_shared_t, void*, int32_t)>(cuda::ptx::tensormap_replace_rank));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.box_dim.global.b1024.b32           [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_box_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.box_dim.shared::cta.b1024.b32           [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_box_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_dim.global.b1024.b32        [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_global_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_dim.shared::cta.b1024.b32        [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_global_dim));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_stride.global.b1024.b64     [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int64_t)>(
            cuda::ptx::tensormap_replace_global_stride));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.global_stride.shared::cta.b1024.b64     [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int64_t)>(
            cuda::ptx::tensormap_replace_global_stride));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.element_stride.global.b1024.b32    [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_element_size));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.element_stride.shared::cta.b1024.b32    [tm_addr], ord, new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>, int32_t)>(
            cuda::ptx::tensormap_replace_element_size));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.elemtype.global.b1024.b32          [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_elemtype));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.elemtype.shared::cta.b1024.b32          [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_elemtype));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.interleave_layout.global.b1024.b32 [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_interleave_layout));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.interleave_layout.shared::cta.b1024.b32 [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_interleave_layout));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.swizzle_mode.global.b1024.b32      [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_swizzle_mode));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.swizzle_mode.shared::cta.b1024.b32      [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_swizzle_mode));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.fill_mode.global.b1024.b32         [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_global_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_fill_mode));));
#endif // __cccl_ptx_isa >= 830

#if __cccl_ptx_isa >= 830
  NV_IF_TARGET(
    NV_HAS_FEATURE_SM_90a,
    (
        // tensormap.replace.tile.fill_mode.shared::cta.b1024.b32         [tm_addr], new_val;
        * fn_ptr++ =
          reinterpret_cast<void*>(static_cast<void (*)(cuda::ptx::space_shared_t, void*, cuda::ptx::n32_t<0>)>(
            cuda::ptx::tensormap_replace_fill_mode));));
#endif // __cccl_ptx_isa >= 830
}
