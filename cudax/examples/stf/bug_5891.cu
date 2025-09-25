#include <cuda/experimental/stf.cuh>
#include <cuda/experimental/__stf/places/blocked_partition.cuh>

using namespace cuda::experimental::stf;

struct foo {
    char bar[2];
};

int main() {
    constexpr size_t N = 1ULL << 32;  // 4,294,967,296 elements
    std::vector<foo> X(N);
    foo* data_X = X.data();
    exec_place_grid where;
    where = exec_place::repeat(exec_place::current_device(), 2);
    auto cdp = data_place::composite(blocked_partition(), where);
    context ctx;
    auto lX = ctx.logical_data(data_X, {N});
    ctx.launch(where, lX.rw(cdp))->*[] __device__(auto t, auto dX) {};
    ctx.finalize();
    return 0;
}
