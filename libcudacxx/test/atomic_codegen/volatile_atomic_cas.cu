#include <cuda/atomic>

__global__ void test_volatile(int* data, int n) {
    if (n) {
        volatile auto ref = cuda::atomic_ref<int, cuda::thread_scope_device>{*(data)};
        ref.compare_exchange_strong(n, n, cuda::std::memory_order_relaxed);
    }
}

/*
; CHECK:     {{Function.*test_volatile.*}}
; CHECK-NOT: {{.*}}STL.64{{.*}}
; CHECK:     {{.*}}ATOM.E.CAS.STRONG.GPU{{.*}}
*/

int main() {
    return 0;
}
