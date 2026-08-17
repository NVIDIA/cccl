#include <cuda/atomic>

__global__ void store_relaxed_device_non_volatile(int* data, int in)
{
  auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
  ref.store(in, cuda::std::memory_order_relaxed);
}

/*

; SMXX-LABEL: .target sm_{{[0-9]+}}
; SMXX:      .visible .entry [[FUNCTION:_.*store_relaxed_device_non_volatile.*]](
; SMXX-DAG:  ld.param.{{b|u}}64 %rd[[#ATOM:]], {{.*}}[[FUNCTION]]_param_0{{.*}}
; SMXX-DAG:  ld.param.{{b|u}}32 %r[[#INPUT:]], {{.*}}[[FUNCTION]]_param_1{{.*}}
; SMXX-NEXT: {{/*[[:space:]] *}}st.relaxed.gpu.b32 [%rd[[#ATOM]]],%r[[#INPUT]];{{[[:space:]]/*}}
; SMXX-NEXT: ret;

*/
