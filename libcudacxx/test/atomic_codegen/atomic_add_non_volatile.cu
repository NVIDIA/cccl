#include <cuda/atomic>

__global__ void add_relaxed_device_non_volatile(int* data, int* out, int n)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  *out     = ref.fetch_add(n, cuda::std::memory_order_relaxed);
}

/*

; SM8X-LABEL: .target sm_80
; SM8X:      .visible .entry [[FUNCTION:_.*add_relaxed_device_non_volatile.*]](
; SM8X-DAG:  ld.param.u64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SM8X-DAG:  ld.param.u64 %rd[[#RESULT:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; SM8X-DAG:  ld.param.u32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_2{{.*}}
; SM8X-NEXT: cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#RESULT]];
; SM8X-NEXT: {{/*[[:space:]] *}}atom.add.relaxed.gpu.s32 %r[[#DEST:]],[%rd[[#ATOM]]],%r[[#INPUT]];{{[[:space:]]/*}}
; SM8X-NEXT: st.global.u32 [%rd[[#GOUT]]], %r[[#DEST]];
; SM8X-NEXT: ret;

*/
