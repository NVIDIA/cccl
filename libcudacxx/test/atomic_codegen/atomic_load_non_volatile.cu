#include <cuda/atomic>

__global__ void load_relaxed_device_non_volatile(int* data, int* out)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  *out     = ref.load(cuda::std::memory_order_relaxed);
}

/*

; SM8X-LABEL: .target sm_80
; SM8X:      .visible .entry [[FUNCTION:_.*load_relaxed_device_non_volatile.*]](
; SM8X-DAG:  ld.param.u64 %rd[[#ATOM:]], [[[FUNCTION]]_param_0];
; SM8X-DAG:  ld.param.u64 %rd[[#EXPECTED:]], [[[FUNCTION]]_param_1];
; SM8X-NEXT: cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#EXPECTED]];
; SM8X-NEXT: //
; SM8X-NEXT: ld.relaxed.gpu.b32 %r[[#DEST:]],[%rd[[#ATOM]]];
; SM8X-NEXT: //
; SM8X-NEXT: st.global.u32 [%rd[[#GOUT]]], %r[[#DEST]];
; SM8X-NEXT: ret;

*/
