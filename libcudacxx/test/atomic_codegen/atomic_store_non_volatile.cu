#include <cuda/atomic>

__global__ void store_relaxed_device_non_volatile(int* data, int in)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  ref.store(in, cuda::std::memory_order_relaxed);
}

__global__ void store_relaxed_cluster_non_volatile(int* data, int in)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_cluster>{*(data)};
  ref.store(in, cuda::std::memory_order_relaxed);
}

/*

; SMXX-LABEL: .target sm_{{[0-9]+[af]?}}
; SMXX:      .visible .entry [[FUNCTION:_.*store_relaxed_device_non_volatile.*]](
; SMXX-DAG:  ld.param.{{b|u}}64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SMXX-DAG:  ld.param.{{b|u}}32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; GLOBAL-DAG: cvta.to.global.u64 %rd[[#GATOM:]], %rd[[#ATOM]];
; GENERIC-NEXT: {{/*[[:space:]] *}}st.relaxed.gpu.b32 [%rd[[#ATOM]]],{{ ?}}%r[[#INPUT]];{{[[:space:]]/*}}
; GLOBAL-NEXT: st.relaxed.gpu.global.b32 [%rd[[#GATOM]]], %r[[#INPUT]];
; SMXX-NEXT: ret;

; SMXX-LABEL: .visible .entry {{_.*store_relaxed_cluster_non_volatile.*}}(
; SMXX-NOT: {{.*}}st.relaxed.{{.*}}
; GENERIC: {{.*}}st.relaxed.[[CLUSTER_SCOPE]].b32 {{.*}}
; GLOBAL: st.relaxed.[[CLUSTER_SCOPE]].global.b32 {{.*}}
; SMXX-NOT: {{.*}}st.relaxed.{{.*}}
; SMXX: ret;

*/
