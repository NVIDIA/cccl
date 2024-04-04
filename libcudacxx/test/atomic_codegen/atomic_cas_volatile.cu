#include <cuda/atomic>

__global__ void cas_device_relaxed_volatile(int* data, int n) {
    if (n) {
        volatile auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
        ref.compare_exchange_strong(n, n, cuda::std::memory_order_relaxed);
    }
}

/*
## SM80 checks
; SM8X: Fatbin elf code
; SM8X: code for sm_8{{[0-9]}}
; SM8X-DAG:  {{^.*}}Function : {{.*}}cas_device_relaxed_volatile{{.*$}}
; SM8X:      {{^.*}}STL.{{[0-9]*}}{{.*$}}
; SM8X:      {{^.*}}ATOM.E.CAS.STRONG.GPU{{.*$}}
; SM8X-DAG:  {{^.*}}EXIT{{.*$}}
*/
