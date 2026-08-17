#include <cuda/atomic>

__global__ void cas_device_relaxed_non_volatile(int* data, int* out, int n)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  ref.compare_exchange_strong(*out, n, cuda::std::memory_order_relaxed);
}

// clang-format off
/*

; SMXX-LABEL: .target sm_{{[0-9]+}}
; SMXX:      .visible .entry [[FUNCTION:_.*cas_device_relaxed_non_volatile.*]](
; SMXX-DAG:  ld.param.{{b|u}}64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SMXX-DAG:  ld.param.{{b|u}}64 %rd[[#EXPECTED:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; SMXX-DAG:  ld.param.{{b|u}}32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_2{{.*}}
; SMXX-DAG:  cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#EXPECTED]];
; SMXX-DAG:  ld.global.{{b|u}}32 %r[[#LOCALEXP:]], [%rd[[#INPUT]]];
; SMXX-NEXT: {{/*[[:space:]] *}}atom.cas.relaxed.gpu.b32 %r[[#DEST:]],[%rd[[#ATOM]]],%r[[#LOCALEXP]],%r[[#INPUT]];{{[[:space:]]/*}}
; SMXX-NEXT: st.global.{{b|u}}32 [%rd[[#GOUT]]], %r[[#DEST]];
; SMXX-NEXT: ret;

*/
