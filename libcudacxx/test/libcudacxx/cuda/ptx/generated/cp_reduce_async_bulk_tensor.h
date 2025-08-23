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

__global__ void test_cp_reduce_async_bulk_tensor(void** fn_ptr)
{
#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.tensor.1d.global.shared::cta.add.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; //
        // 1a.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               const void*,
                               const cuda::std::int32_t (&)[1],
                               const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.min.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_min_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.max.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_max_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.inc.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_inc_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.dec.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_dec_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.and.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.or.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
          // // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.1d.global.shared::cta.xor.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1a.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[1],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; //
        // 1b.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               const void*,
                               const cuda::std::int32_t (&)[2],
                               const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.min.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_min_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.max.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_max_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.inc.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_inc_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.dec.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_dec_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.and.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.or.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
          // // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.2d.global.shared::cta.xor.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1b.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[2],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; //
        // 1c.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               const void*,
                               const cuda::std::int32_t (&)[3],
                               const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.min.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_min_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.max.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_max_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.inc.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_inc_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.dec.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_dec_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.and.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.or.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
          // // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.3d.global.shared::cta.xor.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1c.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[3],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.tensor.4d.global.shared::cta.add.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; //
        // 1d.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               const void*,
                               const cuda::std::int32_t (&)[4],
                               const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.min.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_min_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.max.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_max_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.inc.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_inc_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.dec.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_dec_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.and.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.or.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
          // // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.4d.global.shared::cta.xor.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1d.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[4],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800

#if __cccl_ptx_isa >= 800
  NV_IF_TARGET(
    NV_PROVIDES_SM_90,
    (
        // cp.reduce.async.bulk.tensor.5d.global.shared::cta.add.tile.bulk_group [tensorMap, tensorCoords], [srcMem]; //
        // 1e.
        * fn_ptr++ = reinterpret_cast<void*>(
          static_cast<void (*)(cuda::ptx::space_global_t,
                               cuda::ptx::space_shared_t,
                               cuda::ptx::op_add_t,
                               const void*,
                               const cuda::std::int32_t (&)[5],
                               const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.min.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_min_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.max.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_max_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.inc.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_inc_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.dec.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_dec_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.and.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_and_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.or.tile.bulk_group [tensorMap, tensorCoords], [srcMem];
          // // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_or_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));
          // cp.reduce.async.bulk.tensor.5d.global.shared::cta.xor.tile.bulk_group [tensorMap, tensorCoords],
          // [srcMem]; // 1e.
            * fn_ptr++ = reinterpret_cast<void*>(
              static_cast<void (*)(cuda::ptx::space_global_t,
                                   cuda::ptx::space_shared_t,
                                   cuda::ptx::op_xor_op_t,
                                   const void*,
                                   const cuda::std::int32_t (&)[5],
                                   const void*)>(cuda::ptx::cp_reduce_async_bulk_tensor));));
#endif // __cccl_ptx_isa >= 800
}
