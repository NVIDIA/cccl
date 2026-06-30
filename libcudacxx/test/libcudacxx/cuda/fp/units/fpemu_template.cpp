#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

/**
 * @brief Example demonstrating how to pass template accuracy parameters to the fp64emu_t template type
 *
 * This example shows how to explicitly specify accuracy template parameters
 * when constructing fp64emu emulated floating-point types. By using the fp64emu_t<m>
 * template, users can select the desired accuracy level at compile time for each variable or operation.
 *
 * The run_mode template function illustrates this by instantiating fp64emu_t with different
 * accuracy levels (high, mid, low), enabling control over
 * floating-point emulation behavior. This approach is useful for applications requiring
 * precise selection of floating-point semantics for correctness, performance, or compatibility.
 *
 * All builtin functions (__dadd_rn, __dmul_rn, fma, sqrt, etc.) are template functions that
 * deduce the accuracy level from their argument types, dispatching to the correct accuracy-specific implementation.
 */

//  Define the macros for the host and device runs
#if __CUDACC__
    #define MALLOC(x,s) cudaMallocManaged(&x,s)
    #define FREE(x) cudaFree(x)
    #define RUN_EXAMPLE(x,y) run_example<<<1, 1>>>(x,y)
    #define DEVICE_SYNCHRONIZE() cudaDeviceSynchronize()
    #define TARGET_DEVICE __global__
#else
    #define MALLOC(x,s) x = (double*)malloc(s)
    #define FREE(x) free(x)
    #define RUN_EXAMPLE(x,y) run_example(x,y)
    #define DEVICE_SYNCHRONIZE()
    #define TARGET_DEVICE
    #define __device__ 
    #define __host__ 
#endif

// Define the run_mode template function
// m: accuracy level (high, mid, low; def == high)
template<fp64emu_accuracy m = fp64emu_accuracy::def>
__device__ void run_mode(double *inp, double *out)
{
    fp64emu_t<m> x = inp[0];
    fp64emu_t<m> c = 0.001;
    // Compute: ((x + x) * x - x) / (x + c) using builtins
    // Accuracy m is deduced from argument types - dispatches to correct accuracy builtin
    out[0] = __ddiv_rn(__dsub_rn(__dmul_rn(__dadd_rn(x, x), x), x), __dadd_rn(x, c));
    return;
}

// Native hardware version using standard operations
// This matches the computation in run_mode exactly
__device__ void run_native_hw(double *inp, double *out)
{
    double x = inp[0];
    double c = 0.001;
    #if defined(__CUDA_ARCH__)
        out[0] = __ddiv_rn(__dsub_rn(__dmul_rn(__dadd_rn(x, x), x), x), __dadd_rn(x, c));
    #else
        out[0] = ((x + x) * x - x) / (x + c);
    #endif
}

// Define the run_example function
// inp: input array
// out: output array
// Runs the run_mode template function for different accuracy levels
TARGET_DEVICE void run_example(double *inp, double *out) 
{
    run_native_hw(&inp[0], &out[0]);  // Native hardware
    run_mode<fp64emu_accuracy::high>(&inp[1], &out[1]);  // Emulated high
    run_mode<fp64emu_accuracy::def>(&inp[2], &out[2]);      // Emulated def
    run_mode<fp64emu_accuracy::low>(&inp[3], &out[3]);     // Emulated low
}

int main(int argc, char** argv) 
{
    (void)argv; // Suppress unused parameter warning
    // Declare pointers for input, output and fp64emu results
    double* inp;
    double* out;

    // Allocate Unified Memory – accessible from CPU or GPU 
    MALLOC(inp, 5 * sizeof(double));
    MALLOC(out, 5 * sizeof(double));

    // Initialize input values
    inp[0] = -0x1.57f1782782a8ap-1 * (double)argc;
    inp[1] = inp[0];
    inp[2] = inp[0];
    inp[3] = inp[0];
    inp[4] = inp[0];

    // Launch CUDA kernel
    RUN_EXAMPLE(inp, out);

    // Wait for GPU to finish before accessing results on CPU
    DEVICE_SYNCHRONIZE();

    // Print results
    printf("  ** This example demonstrates the usage of the fp64emu library template\n     controlling the accuracy level\n\n");
    printf("Operation: ((x + x) * x - x) / (x + 0.001)\n\n");
    printf("Input: %la,    Native HW output: %la\n", inp[0], out[0]);
    printf("Input: %la,   Emulated high output: %la\n", inp[1], out[1]);
    printf("Input: %la,   Emulated def  output: %la\n", inp[2], out[2]); 
    printf("Input: %la,   Emulated low  output: %la\n", inp[3], out[3]);

    // Basic verification: emulated results should match native HW within tolerance
    int errors = 0;
    double tol_acc  = 1e-12;  // high accuracy: tight tolerance
    double tol_def  = 1e-10;  // def accuracy
    double tol_fast = 1e-4;   // low accuracy: relaxed tolerance

    double err_acc  = fabs(out[1] - out[0]);
    double err_def  = fabs(out[2] - out[0]);
    double err_fast = fabs(out[3] - out[0]);

    if (err_acc  > tol_acc)  { printf("FAIL: high err=%.2e\n", err_acc);  errors++; }
    if (err_def  > tol_def)  { printf("FAIL: def err=%.2e\n",  err_def);  errors++; }
    if (err_fast > tol_fast) { printf("FAIL: low err=%.2e\n",  err_fast); errors++; }

    printf("fpemu_template: %s (%d errors)\n", errors ? "FAIL" : "PASS", errors);

    // Free Unified Memory
    FREE(inp);
    FREE(out);

    return errors ? 1 : 0;
}
