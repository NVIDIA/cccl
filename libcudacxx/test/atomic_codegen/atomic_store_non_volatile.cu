#include <cuda/atomic>

__global__ void store_relaxed_device_non_volatile(int* data, int in)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  ref.store(in, cuda::std::memory_order_relaxed);
}

/*

; SM8X-LABEL: .target sm_80
; SM8X:      .visible .entry [[FUNCTION:_.*store_relaxed_device_non_volatile.*]](
; SM8X-DAG:  ld.param.u64 %rd[[#ATOM:]], [[[FUNCTION]]_param_0];
; SM8X-DAG:  ld.param.u32 %r[[#INPUT:]], [[[FUNCTION]]_param_1];
; SM8X-NEXT: //
; SM8X-NEXT: st.relaxed.gpu.b32 [%rd[[#ATOM]]],%r[[#INPUT]];
; SM8X-NEXT: //
; SM8X-NEXT: ret;

*/
