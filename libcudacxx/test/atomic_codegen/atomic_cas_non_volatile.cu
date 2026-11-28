#include <cuda/atomic>

// Export the observed value on both outcomes; CAS itself only updates expected on failure.
__global__ void cas_device_relaxed_non_volatile(int* data, int* out, int n)
{
  auto ref     = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  int expected = *out;
  ref.compare_exchange_strong(expected, n, cuda::std::memory_order_relaxed);
  *out = expected;
}

__global__ void cas_cluster_relaxed_non_volatile(int* data, int* out, int n)
{
  auto ref     = cuda::atomic_ref<int, cuda::thread_scope_cluster>{*(data)};
  int expected = *out;
  ref.compare_exchange_strong(expected, n, cuda::std::memory_order_relaxed);
  *out = expected;
}

// clang-format off
/*

; SMXX-LABEL: .target sm_{{[0-9]+[af]?}}
; SMXX:      .visible .entry [[FUNCTION:_.*cas_device_relaxed_non_volatile.*]](
; SMXX-DAG:  ld.param.{{b|u}}64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SMXX-DAG:  ld.param.{{b|u}}64 %rd[[#EXPECTED:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; SMXX-DAG:  ld.param.{{b|u}}32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_2{{.*}}
; GLOBAL-DAG: cvta.to.global.u64 %rd[[#GATOM:]], %rd[[#ATOM]];
; SMXX-DAG:  cvta.to.global.u64 %rd[[#GOUT:]], %rd[[#EXPECTED]];
; SMXX-DAG:  ld.global.{{b|u}}32 %r[[#LOCALEXP:]], [%rd[[#GOUT]]];
; GENERIC-NEXT: {{/*[[:space:]] *}}atom.cas.relaxed.gpu.b32 %r[[#DEST:]],[%rd[[#ATOM]]],%r[[#LOCALEXP]],%r[[#INPUT]];{{[[:space:]]/*}}
; GLOBAL-NEXT: atom.relaxed.gpu.global.cas.b32 %r[[#DEST:]], [%rd[[#GATOM]]], %r[[#LOCALEXP]], %r[[#INPUT]];
; SMXX-NEXT: st.global.{{b|u}}32 [%rd[[#GOUT]]], %r[[#DEST]];
; SMXX-NEXT: ret;

; SMXX-LABEL: .visible .entry {{_.*cas_cluster_relaxed_non_volatile.*}}(
; SMXX-NOT: {{.*}}atom.{{.*}}
; GENERIC: {{.*}}atom.cas.relaxed.[[CLUSTER_SCOPE]].b32 {{.*}}
; GLOBAL: atom.relaxed.[[CLUSTER_SCOPE]].global.cas.b32 {{.*}}
; SMXX-NOT: {{.*}}atom.{{.*}}
; SMXX: ret;

*/
