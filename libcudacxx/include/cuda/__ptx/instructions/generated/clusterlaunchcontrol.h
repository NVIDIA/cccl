// This file was automatically generated. Do not edit.

#ifndef _CUDA_PTX_GENERATED_CLUSTERLAUNCHCONTROL_H_
#define _CUDA_PTX_GENERATED_CLUSTERLAUNCHCONTROL_H_

/*
// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [addr], [smem_bar]; // PTX ISA
86, SM_100 template <typename = void>
__device__ static inline void clusterlaunchcontrol_try_cancel(
  void* addr,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void clusterlaunchcontrol_try_cancel(void* __addr, ::cuda::std::uint64_t* __smem_bar)
{
  asm("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [%0], [%1];"
      :
      : "r"(__as_ptr_smem(__addr)), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 [addr],
[smem_bar]; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f, SM_120a, SM_120f,
SM_121a, SM_121f template <typename = void>
__device__ static inline void clusterlaunchcontrol_try_cancel_multicast(
  void* addr,
  uint64_t* smem_bar);
*/
#if __cccl_ptx_isa >= 860
template <typename = void>
_CCCL_DEVICE static inline void
clusterlaunchcontrol_try_cancel_multicast(void* __addr, ::cuda::std::uint64_t* __smem_bar)
{
  asm("clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.multicast::cluster::all.b128 "
      "[%0], [%1];"
      :
      : "r"(__as_ptr_smem(__addr)), "r"(__as_ptr_smem(__smem_bar))
      : "memory");
}
#endif // __cccl_ptx_isa >= 860

/*
// clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 pred_is_canceled, try_cancel_response; // PTX ISA 86, SM_100
template <typename B128, enable_if_t<sizeof(B128) == 16, bool> = true>
__device__ static inline bool clusterlaunchcontrol_query_cancel_is_canceled(
  B128 try_cancel_response);
*/
#if __cccl_ptx_isa >= 860
template <typename _B128, ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline bool clusterlaunchcontrol_query_cancel_is_canceled(_B128 __try_cancel_response)
{
  static_assert(sizeof(_B128) == 16, "");
  ::cuda::std::uint32_t __pred_is_canceled;
  asm("{\n\t"
      ".reg .pred P_OUT; \n\t"
      ".reg .b128 B128_try_cancel_response; \n\t"
      "mov.b128 B128_try_cancel_response, {%1, %2}; \n\t"
      "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 P_OUT, B128_try_cancel_response; \n\t"
      "selp.b32 %0, 1, 0, P_OUT; \n"
      "}"
      : "=r"(__pred_is_canceled)
      : "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).x),
        "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).y)
      :);
  return static_cast<bool>(__pred_is_canceled);
}
#endif // __cccl_ptx_isa >= 860

/*
// clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 ret_dim, try_cancel_response; // PTX ISA 86, SM_100
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool>
= true>
__device__ static inline B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_x(
  B128 try_cancel_response);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          typename _B128,
          ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_x(_B128 __try_cancel_response)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B128) == 16, "");
  ::cuda::std::uint32_t __ret_dim;
  asm("{\n\t"
      ".reg .b128 B128_try_cancel_response; \n\t"
      "mov.b128 B128_try_cancel_response, {%1, %2}; \n\t"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, B128_try_cancel_response; \n"
      "}"
      : "=r"(__ret_dim)
      : "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).x),
        "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).y)
      :);
  return *reinterpret_cast<_B32*>(&__ret_dim);
}
#endif // __cccl_ptx_isa >= 860

/*
// clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 ret_dim, try_cancel_response; // PTX ISA 86, SM_100
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool>
= true>
__device__ static inline B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_y(
  B128 try_cancel_response);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          typename _B128,
          ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_y(_B128 __try_cancel_response)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B128) == 16, "");
  ::cuda::std::uint32_t __ret_dim;
  asm("{\n\t"
      ".reg .b128 B128_try_cancel_response; \n\t"
      "mov.b128 B128_try_cancel_response, {%1, %2}; \n\t"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, B128_try_cancel_response; \n"
      "}"
      : "=r"(__ret_dim)
      : "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).x),
        "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).y)
      :);
  return *reinterpret_cast<_B32*>(&__ret_dim);
}
#endif // __cccl_ptx_isa >= 860

/*
// clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 ret_dim, try_cancel_response; // PTX ISA 86, SM_100
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool>
= true>
__device__ static inline B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_z(
  B128 try_cancel_response);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          typename _B128,
          ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline _B32 clusterlaunchcontrol_query_cancel_get_first_ctaid_z(_B128 __try_cancel_response)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B128) == 16, "");
  ::cuda::std::uint32_t __ret_dim;
  asm("{\n\t"
      ".reg .b128 B128_try_cancel_response; \n\t"
      "mov.b128 B128_try_cancel_response, {%1, %2}; \n\t"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, B128_try_cancel_response; \n"
      "}"
      : "=r"(__ret_dim)
      : "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).x),
        "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).y)
      :);
  return *reinterpret_cast<_B32*>(&__ret_dim);
}
#endif // __cccl_ptx_isa >= 860

/*
// clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 block_dim, try_cancel_response; // PTX ISA 86, SM_100
template <typename B32, enable_if_t<sizeof(B32) == 4, bool> = true, typename B128, enable_if_t<sizeof(B128) == 16, bool>
= true>
__device__ static inline void clusterlaunchcontrol_query_cancel_get_first_ctaid(
  B32 (&block_dim)[4],
  B128 try_cancel_response);
*/
#if __cccl_ptx_isa >= 860
template <typename _B32,
          ::cuda::std::enable_if_t<sizeof(_B32) == 4, bool> = true,
          typename _B128,
          ::cuda::std::enable_if_t<sizeof(_B128) == 16, bool> = true>
_CCCL_DEVICE static inline void
clusterlaunchcontrol_query_cancel_get_first_ctaid(_B32 (&__block_dim)[4], _B128 __try_cancel_response)
{
  static_assert(sizeof(_B32) == 4, "");
  static_assert(sizeof(_B128) == 16, "");
  asm("{\n\t"
      ".reg .b128 B128_try_cancel_response; \n\t"
      "mov.b128 B128_try_cancel_response, {%4, %5}; \n\t"
      "clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, %3}, B128_try_cancel_response; \n"
      "}"
      : "=r"(__block_dim[0]), "=r"(__block_dim[1]), "=r"(__block_dim[2]), "=r"(__block_dim[3])
      : "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).x),
        "l"((*reinterpret_cast<longlong2*>(&__try_cancel_response)).y)
      :);
}
#endif // __cccl_ptx_isa >= 860

#endif // _CUDA_PTX_GENERATED_CLUSTERLAUNCHCONTROL_H_
