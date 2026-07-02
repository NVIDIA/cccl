/*
    fpemu.cpp - FP64 Emulation Extended Precision Demo
    ======================================================================================================

    This example demonstrates the usage and basic operations of emulated double-precision
    floating-point arithmetic on both CPU and GPU (CUDA).

    What is FP64 Emulation?
    -------------------------------------------------------------------------
    FP64 emulation represents a double-precision (64-bit) IEEE-754 floating-point number
    using single-precision (32-bit) float operations. This is useful when native FP64
    hardware is unavailable or slow, providing configurable accuracy/performance trade-offs
    through the accuracy template parameter.

    Accuracy Levels:
    -------------------------------------------------------------------------
    - fp64emu_accuracy::high — correctly rounded, full IEEE-754 range (INF, NaN, subnormals)
    - fp64emu_accuracy::mid  — up to 1-2 LSB error, limited special value support
    - fp64emu_accuracy::low  — up to half mantissa bits lost, limited special value support
    - fp64emu_accuracy::def  — default selector (== high, IEEE-correct)

    Features Demonstrated:
    -------------------------------------------------------------------------
    - Construction and assignment of fp64emu values
    - Using high accuracy for addition
    - Using low accuracy for multiplication
    - Basic arithmetic: subtraction, division
    - Compound operations: sqrt, fma
    - Comparison operators
    - CPU/GPU host/device compatibility

    Build Instructions:
    -------------------------------------------------------------------------
    Using the provided Makefile (recommended):
        make EXAMPLE=fpemu                # Build for GPU (CUDA)
        make TARGET=host EXAMPLE=fpemu    # Build for CPU only
        make run EXAMPLE=fpemu            # Build and run

    Manual compilation on CPU:
        g++ -std=c++17 -O2 -I../include fpemu.cpp -o fpemu.exe

    Manual compilation with CUDA:
        nvcc -std=c++17 -O2 -I../include fpemu.cpp -o fpemu.exe

    Output:
    -------------------------------------------------------------------------
    The program performs emulated double-precision calculations and compares results with
    IEEE-754 double precision, displaying absolute errors to demonstrate the behavior of
    different accuracy levels.

    Types Used:
    -------------------------------------------------------------------------
    - fp64emu      : Default emulation type (fp64emu_accuracy::def, == high)
    - fp64emu_high : High-accuracy emulation type (correctly rounded, full IEEE range)
    - fp64emu_low  : Low-accuracy emulation type (reduced precision, higher performance)
*/
#include <stdio.h>
#include <stdlib.h>
#include <cmath>

// FPEMU library header
#include <cuda/fpemu>

using namespace cuda::experimental; // FP SDK lives in cuda::experimental (later cuda::)

// Macros for host/device compatibility
#if defined(__CUDACC__)
    #define HOST_DEVICE __host__ __device__
    #define KERNEL      __global__
#else
    #define HOST_DEVICE
    #define KERNEL
#endif

#define VALUE1 1.234567890123456789
#define VALUE2 9.876543210987654321
#define VALUE3 2.324354600000000000
#define VALUE4 5.0

// Simple kernel/function to test fp64 emulation operations
HOST_DEVICE void fpemu_operations(double* results)
{
    // Default accuracy for general use
    fp64emu a = VALUE1;
    fp64emu b = VALUE2;
    fp64emu c = VALUE3;
    fp64emu d = VALUE4;

    // --- Addition using high accuracy ---
    // Demonstrates correctly rounded addition with full IEEE-754 range
    fp64emu_high a_acc = VALUE1;
    fp64emu_high b_acc = VALUE2;
    fp64emu_high sum_acc = a_acc + b_acc;
    results[0] = sum_acc;

    // --- Multiplication using low accuracy ---
    // Demonstrates fast multiplication with reduced precision
    fp64emu_low a_fast = VALUE1;
    fp64emu_low b_fast = VALUE2;
    fp64emu_low prod_fast = a_fast * b_fast;
    results[1] = prod_fast;

    // --- Other operations using default accuracy ---
    fp64emu diff = a - b;
    fp64emu quot = a / b;
    results[2] = diff;
    results[3] = quot;

    // Square root
    fp64emu sqrt_x = sqrt(d);
    results[4] = sqrt_x;

    // FMA (fused multiply-add)
    fp64emu fma_result = fma(a, b, c);
    results[5] = fma_result;

    // Comparisons (store as 0.0 or 1.0)
    results[6] = (a > b)  ? 1.0 : 0.0;
    results[7] = (a < b)  ? 1.0 : 0.0;
    results[8] = (a == a) ? 1.0 : 0.0;

    // Compound operations
    fp64emu temp = a;
    temp += b;
    results[9] = temp;

    temp = a;
    temp *= b;
    results[10] = temp;
} // fpemu_operations

#if defined(__CUDACC__)
// CUDA kernel wrapper
KERNEL void fpemu_kernel(double* results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        fpemu_operations(results);
    }
}
#endif

void run_comparison_test()
{
    printf("\n");
    printf("================================================================================\n");
    printf("  FP64 EMULATION STANDALONE DEMO\n");
    printf("  Comparing Emulated FP64 vs Native Double Precision\n");
    printf("================================================================================\n\n");

    // Input values
    double a_d = VALUE1;
    double b_d = VALUE2;
    double c_d = VALUE3;
    double d_d = VALUE4;

    // Double precision reference
    double ref_sum  = a_d + b_d;
    double ref_prod = a_d * b_d;
    double ref_diff = a_d - b_d;
    double ref_quot = a_d / b_d;
    double ref_sqrt = std::sqrt(d_d);
    double ref_fma  = std::fma(a_d, b_d, c_d);

    printf("Input values:\n");
    printf("  a = %.17f\n", a_d);
    printf("  b = %.17f\n", b_d);
    printf("  c = %.17f\n", c_d);
    printf("  d = %.17f\n\n", d_d);

    // Allocate memory for results
    const int NUM_RESULTS = 11;
    double* results;

#if defined(__CUDACC__)
    // CUDA path
    cudaMallocManaged(&results, NUM_RESULTS * sizeof(double));

    // Increase stack size for fpemu template instantiations which are stack-heavy
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);

    printf("Running on GPU...\n");
    fpemu_kernel<<<1, 1>>>(results);

    // Check for errors (capture sync return value)
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA Error: %s\n", cudaGetErrorString(err));
        cudaFree(results);
        return;
    }
    printf("GPU execution completed.\n\n");
#else
    // CPU path
    results = (double*)malloc(NUM_RESULTS * sizeof(double));
    printf("Running on CPU...\n");
    fpemu_operations(results);
    printf("CPU execution completed.\n\n");
#endif

    // Display results
    printf("Results:\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("Operation          | Double Reference   | Emulated Result    | Abs Error\n");
    printf("--------------------------------------------------------------------------------\n");

    printf("a + b (high)       | %18.15f | %18.15f | %.2e\n", ref_sum,  results[0], fabs(results[0] - ref_sum));
    printf("a * b (low)        | %18.15f | %18.15f | %.2e\n", ref_prod, results[1], fabs(results[1] - ref_prod));
    printf("a - b (def)        | %18.15f | %18.15f | %.2e\n", ref_diff, results[2], fabs(results[2] - ref_diff));
    printf("a / b (def)        | %18.15f | %18.15f | %.2e\n", ref_quot, results[3], fabs(results[3] - ref_quot));
    printf("sqrt(d) (def)      | %18.15f | %18.15f | %.2e\n", ref_sqrt, results[4], fabs(results[4] - ref_sqrt));
    printf("fma(a, b, c) (def) | %18.15f | %18.15f | %.2e\n", ref_fma,  results[5], fabs(results[5] - ref_fma));

    printf("\nComparisons (1.0 = true, 0.0 = false):\n");
    printf("  a > b:  %.1f\n", results[6]);
    printf("  a < b:  %.1f\n", results[7]);
    printf("  a == a: %.1f\n", results[8]);

    printf("\nCompound operations (def accuracy):\n");
    printf("  a += b: %.15f\n", results[9]);
    printf("  a *= b: %.15f\n", results[10]);

    printf("\n");
    printf("================================================================================\n");
    printf("  DEMO COMPLETED SUCCESSFULLY\n");
    printf("================================================================================\n");

    // Cleanup
#if defined(__CUDACC__)
    cudaFree(results);
#else
    free(results);
#endif
} // run_comparison_test

int main()
{
    run_comparison_test();

    printf("\nThis standalone demo demonstrated:\n");
    printf("  FP64 emulation construction from double values\n");
    printf("  Accuracy-specific operations:\n");
    printf("    - high: correctly rounded addition (full IEEE-754)\n");
    printf("    - low: reduced precision multiplication (higher performance)\n");
    printf("    - def: default accuracy for subtraction, division, sqrt, fma\n");
    printf("  Comparison operators\n");
    printf("  Compound assignments\n");
    printf("  Accuracy comparison with native double precision\n");
#if defined(__CUDACC__)
    printf("  GPU/CUDA execution\n");
#else
    printf("  CPU execution\n");
#endif
    printf("\n");

    return 0;
} // main
