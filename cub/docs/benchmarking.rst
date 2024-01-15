CUB Benchmarks
*************************************

This file contains instrutions on how to run all CUB benchmarks using CUB tuning infrastructure.

.. code-block:: bash

    pip3 install --user fpzip pandas scipy
    git clone https://github.com/NVIDIA/cccl.git
    cmake -B build -DCCCL_ENABLE_THRUST=OFF\
             -DCCCL_ENABLE_LIBCUDACXX=OFF\
             -DCCCL_ENABLE_CUB=ON\
             -DCUB_ENABLE_DIALECT_CPP11=OFF\
             -DCUB_ENABLE_DIALECT_CPP14=OFF\
             -DCUB_ENABLE_DIALECT_CPP17=ON\
             -DCUB_ENABLE_DIALECT_CPP20=OFF\
             -DCUB_ENABLE_RDC_TESTS=OFF\
             -DCUB_ENABLE_BENCHMARKS=YES\
             -DCUB_ENABLE_TUNING=YES\
             -DCMAKE_BUILD_TYPE=Release\
             -DCMAKE_CUDA_ARCHITECTURES="89;90"
    cd build
    ../cub/benchmarks/scripts/run.py


Expected output for the command above is:


.. code-block:: bash

    ../cub/benchmarks/scripts/run.py
    &&&& RUNNING bench
    ctk:  12.2.140
    cub:  812ba98d1
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__16 4.095999884157209e-06 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__20 1.2288000107218977e-05 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__24 0.00016998399223666638 -sec
    &&&& PERF cub_bench_adjacent_difference_subtract_left_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__28 0.002673664130270481 -sec
    ...


It's also possible to benchmark a subset of algorithms and workloads:

.. code-block:: bash

    ../cub/benchmarks/scripts/run.py -R '.*scan.exclusive.sum.*' -a 'Elements{io}[pow2]=[24,28]' -a 'T{ct}=I32'
    &&&& RUNNING bench
    ctk:  12.2.140
    cub:  812ba98d1
    &&&& PERF cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__24 0.00016899200272746384 -sec
    &&&& PERF cub_bench_scan_exclusive_sum_base_T_ct__I32___OffsetT_ct__I32___Elements_io__pow2__28 0.002696000039577484 -sec
    &&&& PASSED bench

