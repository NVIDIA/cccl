# C++ stdBLAS Library with Accelerated and Optimized Backends

## C++ Standard Libraries for Linear Algebra

### C++ mdspan Library
This library provides views to multi-dimensional arrays.

The most recent version of the mdspan proposal can be found at https://wg21.link/p0009.

A reference implementation of mdspan can be found at https://github.com/kokkos/mdspan.

A copy of the reference implementation up to Pull Request 151, https://github.com/kokkos/mdspan/pull/151, is shipped with the HPC SDK.

### C++ stdBLAS Library
This library provides linear algebra operations similar to the BLAS library, but with simplified and flexible C++ APIs including support for matrices with both column-major and row-major memory layouts, and in-place transformations.

The most recent version of the stdBLAS proposal can be found at https://wg21.link/p1673/.

A reference implementation of stdBLAS can be found at https://github.com/kokkos/stdBLAS.

A copy of the reference implmentation up to Pull Request 241, https://github.com/kokkos/stdBLAS/pull/241, is shipped with the HPC SDK.

### Accelerated and Optimized Backends
Like all `stdpar` algorithms, stdBLAS provides possibilities to run the computation with parallelized or accelerated algorithms by calling the API with a parallel execution policy. Accelerated backends based on cuBLAS and optimized backend based on BLAS/OpenBLAS are available.

## Requirements
The library has been tested to work with the following environment:
- GCC toolchain 7.2 or higher
- C++17 or higher
- CUDA runtime 11.2 or higher
- CUDA Toolkit 11.4.2 or higher

## Usage

### stdBLAS Sequential Implementation

One can include the linear algebra header file to access the sequential implementation of all stdBLAS functions.
```
#include <experimental/linalg>
```
Currently, all stdBLAS functions are under namespace `std::experimental::linalg`. A namespace alias `stdex` is introduced for `std::experimental` for convenience purposes.
```
stdex::linalg::matrix_product( std::execution::seq, A, B, C );
```
Each function provides an overloaded API without an execution policy.  A call to this API is directed to the sequential implementation as well.
```
stdex::linalg::matrix_product( A, B, C );
```
Since the stdBLAS reference implementation requires some C++17 features, we compile with the `--c++17` flag:
```
nvc++ -c --c++17 -o test_stdblas.o test_stdblas.cpp
nvc++ -o test_stdblas.out test_stdblas.o
```

### Use Accelerated/Optimized Backends
Like other C++ standard parallel algorithms, parallel backends for stdBLAS can be enabled by compiling with the `-stdpar` flag.  However, we need to make sure that the cuBLAS/BLAS libraries are accessible at compile time and link time.  If optimized libraries are not accessible, sequential fallback is available.

#### Use the cuBLAS Backend
To offload computation to the GPU using cuBLAS as the backend, add `-stdpar[=gpu] -cudalib=cublas` to the compile line and the link line.  For example:
```
nvc++ -c --c++17 -stdpar -cudalib=cublas -o test_stdblas.o test_stdblas.cpp
nvc++ -o test_stdblas.out -stdpar -cudalib=cublas test_stdblas.o
```
In this case, calls with the `std::execution::par` execution policy are directed to the cuBLAS backend.
```
stdex::linalg::matrix_product( std::execution::par, A, B, C );
```
Data movements between the host and the device is taken care of by managed memory, which is enabled by the `-stdpar` flag.

If the cuBLAS library is not accessible at compile time (for example, if the `-cudalib=cublas` flag is not present), the compiler will attempt to fall back to the sequential algorithm for calls with `std::execution::par` and report a warning.

If the cuBLAS library is not provided at link time, there are "undefined reference" error messages.

#### Use the `no_sync` cuBLAS Backend
By default, the cuBLAS backend is synchronous, with a `cudaStreamSynchronize` call after cuBLAS calls. This ensures compatibility with the `std::par` and `std::par_unseq` standard C++ execution policies. However, it may lead to performance loss, especially for iterative algorithms or for repeated calls to small problems, when those synchronizations are not necessary.

One can call the cuBLAS backend without the synchronization by calling stdBLAS APIs with execution policy `no_sync(std::execution::par)` instead of `std::execution::par`. When needed, a synchronization can be done by either making a stdBLAS API call with `std::execution::par` or calling `cudaDeviceSynchronize()`. For example:

```
for ( int iter = 0; iter < niter; ++iter )
{
    stdex::linalg::matrix_product( no_sync( std::execution::par ), A, B, C );
}

cudaDeviceSynchronize();
```
This is similar to the `thrust::cuda::par_nosync` execution policy in `thrust`.

> Neither `no_sync` nor `cudaDeviceSynchronize()` is available when the BLAS backend is used (when compiling with `-stdpar=multicore -lblas`).

#### Use the BLAS Backend
To execute on the CPU using optimized algorithms in OpenBLAS as the backend, add `-stdpar=multicore` to the compile line and add `-lblas` to the link line.
```
nvc++ --c++17 -stdpar=multicore -o test_stdblas.o test_stdblas.cpp
nvc++ -o test_stdblas.out -lblas test_stdblas.o
```
In this case, calls with the `std::execution::par` execution policy are directed to the OpenBLAS backend.
```
stdex::linalg::matrix_product( std::execution::par, A, B, C );
```
If the BLAS library is not provided at link time, there are "undefined reference" error messages.

> The OpenBLAS backend uses OpenMP for multi-threading.

> With either of the backends enabled, one can still choose to run the sequential algorithm by specifying the `std::execution::seq` execution policy, or by calling the API without the execution policy argument.
> ```
> stdex::linalg::matrix_product( std::execution::seq, A, B, C );
> stdex::linalg::matrix_product(                      A, B, C );
> ```

## Limitations, Error Handling and Fallback

### Limitations
The mdspan and stdBLAS libraries are flexible in support of memory layouts and in-place transformations. However, there are limitations in cases that can be accelerated using the cuBLAS/BLAS backends.

#### Data Types
The BLAS and most of the cuBLAS APIs only support four data types: `float`, `double`, `std::complex<float>` and `std::complex<double>`. These are all data types that are supported for acceleration, except for `matrix_product` that supports limited  mixed-precision scenarios.

#### Size Types
The mdspan library supports user-specified size types, including both 32-bit and 64-bit integral types. On the other hand, the cuBLAS library only supports the 32-bit size type of `int`, while the BLAS library provides APIs with sizes of both 32-bit and 64-bit integers.

Currently, both cuBLAS and BLAS backends of the stdBLAS library are written with an `int` size type, and call cuBLAS and BLAS APIs with 32-bit integers.

In the meantime, they have been tested to work with mdspan's with 64-bit integer size types (for example, `std::size_t` and `std::int64_t`), as long as the extents of the mdspan do not exceed the range of `int`.

#### Memory Layouts
The mdspan library provides support to multiple pre-defined memory layouts as well as user-defined memory layouts. On the other hand, cuBLAS/BLAS APIs only take matrices in the column-major layout (`layout_left`). The stdBLAS accelerated backends are written to support both row-major and column-major layouts. Below is a list of all memory layouts supported for acceleration:
- `layout_left`
- `layout_right`
- `layout_stride` (for sub-matrices of matrices in `layout_left` or `layout_right`)

> To work with matrices in `layout_right`, the matrix is treated as the transpose of a matrix in `layout_left`.
>
> For `layout_stride`, it is assumed that either `stride(0) == 1` or `stride(1) == 1`.

#### In-place Transformations
The stdBLAS library provides support to in-place transformations `scaled`, `conjugated`, `transposed`, and their combinations.

The accelerated backends support the following nesting of in-place transformations
- `scaled(A)`
- `conjugated(A)`
- `transposed(A)`
- `scaled(conjugated(A))`
- `conjugated(scaled(A))`
- `scaled(transposed(A))`
- `transposed(scaled(A))`
- `conjugate_transposed(A)`
- `transposed(conjugated(A))`
- `conjugate_transposed(scaled(A))`
- `scaled(conjugate_transposed(A))`

> In real applications, there could be `mdspan<...> A1 = scaled(alpha, A0)` in one place, and `mdspan<...> A2 = scaled(beta, A1)` in another place. This is effectively a nesting in-place transformation of `scaled(beta, scaled(alpha, A))`. Cases like this are not supported for acceleration.
>
> cuBLAS/BLAS APIs support in-place transformations `transposed` and `conjugate_transposed` with the `trans` argument. However, this argument doesn't support the `conjugated` transformation, which is not needed by a linear algebra algorithm. In stdBLAS, an extra `transposed` transformation is introduced when we handle matrices in `layout_right`, which may turn `conjugate_transposed` into `conjugated`. This situation is taken care of by performing an extra element-wise conjugate after the cuBLAS/BLAS API call.

### Error Handling
Below are scenarios where an error could happen:
- Inputs/operations are not supported by the cuBLAS/BLAS backend:
  - Unsupported memory layouts.
  - Unsupported combination of data types.
  - Unsupported combination of in-place transformations on input vectors/matrices.
  - In-place transformation on output or input/output vectors/matrices.
- A cuBLAS or a CUDA API call returns a non-zero status.

For unsupported cases that can be caught at compile-time (for example, unsupported combination of data types), we issue a compile-time error.

For unsupported cases that can only be detected at run-time, as well as cuBLAS/CUDA errors, we throw an exception with a `system_error`.  There are three error categories: `stdblas_category`, `cublas_category`, and `cuda_category`. The application is responsible for catching the exception and handling it appropriately.

> The stdBLAS proposal lists _preconditions_ and _constraints_ for each function. It is assumed that all those conditions are met. The library doesn't check those conditions. A call to a stdBLAS API when those preconditions are not met (for example, when matrix dimensionalities are not compatible for a `matrix_product` call) leads to undefined behavior which may cause termination of the application.

### Sequential Fallback Options
For unsupported cases, options are provided to allow a fallback to the sequential implementation instead of compilation errors or exceptions.
- To enable sequential fallback instead of compilation errors for unsupported cases, compile with macro `-DSTDBLAS_FALLBACK`.
- To enable sequential fallback instead of run-time exceptions, set environment variable `NV_STDBLAS_RUNTIME_FALLBACK=1` before running the application.

No sequential fallback is provided for cuBLAS/CUDA errors.

The stdBLAS proposal defines preconditions for each function.  For example, it is the application's responsibility to make sure that the dimensions of input vectors/matrices are compatible for the operation to be performed.  To be consistent with the proposal, the parallel backend wrappers do not perform any checks on a listed precondition.

## Examples
Examples are provided in the `examples/stdpar/stdblas` directory under the HPC SDK directory.

## List of Functions
The following functions are supported for acceleration in this release.

### BLAS Level-1 Functions
Historically, BLAS Level-1 functions operate on vectors.  In stdBLAS, some of the corresponding functions can operate on matrices as well.  These include `add`, `scale` and `copy`.

> To support matrices: In the cuBLAS backend, we call the API for matrix-matrix addition/transposition, `cublas<t>geam`, instead; In the BLAS backend, we write additional wrapper functions to handle this scenario by making one BLAS API for each column/row of the matrix.

#### Add Elementwise (`add`)
Supported interfaces:
- `add( exec, in x, in y, out z )`

#### Multiply Elements by Scalar (`scale`)
Supported interfaces:
- `scale( exec, in alpha, inout a )`

#### Copy Elements (`copy`)
Supported interfaces:
- `copy( exec, in x, out y )`

> Element-wise conjugate, when needed, is done by additional function calls/kernel launches.

#### Dot Product (`dot`, `dotc`)
Supported interfaces:
- `dot ( exec, in x, in y, in init )`
- `dot ( exec, in x, in y )`
- `dotc( exec, in x, in y, in init )`
- `dotc( exec, in x, in y )`

> The cuBLAS/BLAS API call computes the dot product of the raw vectors. Scaling and conjugate, when needed, are both performed afterwards.

#### Euclidean Norm of a Vector (`vector_norm2`)
Supported interfaces:
- `vector_norm2( exec, in x, in init )`
- `vector_norm2( exec, in x )`

> The cuBLAS/BLAS API call computes the vector norm of the raw vector; Scaling, when needed, is performed afterwards.

### BLAS Level-2 Functions

#### General Matrix-Vector Product (`matrix_vector_product`)
Supported interfaces:
- `matrix_vector_product( exec, in A, in x, out y )`
- `matrix_vector_product( exec, in A, in x, in y0, out y )`

> Conjugate-only transformations are done with additional function calls/kernel launches.
> - For matrix A in `layout_right`, this happens for `conjugate_transposed( A )`.

#### Triangular Matrix-Vector Solve (`triangular_matrix_vector_solve`)
Supported interfaces:
-   `triangular_matrix_vector_solve( exec, in A, t, d, in b, out x )`

> The in-place API `triangular_matrix_vector_solve( exec, in A, t, d, inout b )` is not implemented in the reference implementation yet.

> Scaling and conjugate are done with additional function calls/kernel launches.
> - For matrix A in `layout_right`, this happens for `conjugate_transposed( A )`.

### BLAS Level-3 Functions

#### General Matrix-Matrix Product (`matrix_product`)
Supported interfaces:
- `matrix_product( exec, in A, in B, out C )`
- `matrix_product( exec, in A, in B, in E, out C )`

Mixed-precision support:

In addition to the four data types, `matrix_product` also supports the following scenario with mixed-precision:
- `A` and `B` are in `std::int8_t` (or `signed char`), `C` and `E` are in `float`

> Conjugate-only operations are done with additional function calls/kernel launches.
> - For matrices A and B in `layout_right`, this happens for `conjugate_transposed( A )`.

#### Rank-k Symmetric Matrix Update (`symmetric_matrix_rank_k_update`)
Supported interfaces:
- `symmetric_matrix_rank_k_update( exec, in alpha, in A, inout C, t )`
- `symmetric_matrix_rank_k_update( exec,           in A, inout C, t )`

> Conjugate-only operations are done with additional function calls/kernel launches.
> - For matrix A in `layout_right`, this happens for `conjugate_transposed( A )`.

#### Rank-2k Symmetric Matrix Update (`symmetric_matrix_rank_2k_update`)
Supported interfaces:
- `symmetric_matrix_rank_2k_update( in A, in B, inout C, t )`

Limitation on supported in-place transformations:
- If `transposed` is applied on `A `or `B`, both matrices must have the same transformation.

> Conjugate-only operations are done with additional function calls/kernel launches.
> - For matrices A and B in `layout_right`, this happens for `conjugate_transposed( A )`.

#### Solve Multiple Triangular Linear Systems (`triangular_matrix_matrix_[left|right]_solve`)
Supported interfaces:
-   `triangular_matrix_matrix_left_solve ( exec, in A, t, d, in B, out X )`
-   `triangular_matrix_matrix_right_solve( exec, in A, t, d, in B, out X )`

> The in-place APIs `triangular_matrix_matrix_left_solve ( exec, in A, t, d, inout B )`  and `triangular_matrix_matrix_right_solve( exec, in A, t, d, inout B )` are not implemented in the reference implementation yet

> Conjugate-only operations are done with additional function calls/kernel launches.
> - For matrices A and B in `layout_right`, this happens for `conjugate_transposed( A )`.

## Disclaimers

This is a prototype implementation that is still in the standardization process. It is likely to change at some time in the future.
