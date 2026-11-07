.. _libcudacxx-ptx-instructions:

PTX Instructions
================

.. toctree::
   :maxdepth: 1

   instructions/ld
   instructions/st
   instructions/shr
   instructions/shl
   instructions/bmsk
   instructions/elect_sync
   instructions/prmt
   instructions/barrier_cluster
   instructions/bfind
   instructions/clusterlaunchcontrol
   instructions/cp_async_bulk
   instructions/cp_async_bulk_commit_group
   instructions/cp_async_bulk_wait_group
   instructions/cp_async_bulk_tensor
   instructions/cp_async_mbarrier_arrive
   instructions/cp_reduce_async_bulk
   instructions/cp_reduce_async_bulk_tensor
   instructions/exit
   instructions/fence
   instructions/getctarank
   instructions/mapa
   instructions/mbarrier_init
   instructions/mbarrier_inval
   instructions/mbarrier_arrive
   instructions/mbarrier_expect_tx
   instructions/mbarrier_wait
   instructions/multimem_ld_reduce
   instructions/multimem_red
   instructions/multimem_st
   instructions/red_async
   instructions/shfl_sync
   instructions/st_async
   instructions/st_bulk
   instructions/tcgen05_alloc
   instructions/tcgen05_commit
   instructions/tcgen05_cp
   instructions/tcgen05_fence
   instructions/tcgen05_ld
   instructions/tcgen05_mma
   instructions/tcgen05_mma_ws
   instructions/tcgen05_shift
   instructions/tcgen05_st
   instructions/tcgen05_wait
   instructions/tensormap_replace
   instructions/tensormap_cp_fenceproxy
   instructions/trap
   instructions/setmaxnreg
   instructions/special_registers
   instructions/applypriority_async_bulk
   instructions/cp_async_bulk_prefetch
   instructions/cp_async_bulk_prefetch_tensor
   instructions/cp_async_mbarrier_arrive_noinc
   instructions/fabric_submit
   instructions/fabric_try_get
   instructions/fabric_try_pullred
   instructions/fabric_try_put
   instructions/fabric_try_red
   instructions/fabric_wait
   instructions/ldmatrix
   instructions/mbarrier_check_layout
   instructions/mbarrier_complete_tx
   instructions/mbarrier_pending_count
   instructions/prefetch
   instructions/stmatrix


Instructions by section
-----------------------

.. list-table:: `Integer Arithmetic Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `sad <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad>`__
     - No
     -
   * - `div <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div>`__
     - No
     -
   * - `rem <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem>`__
     - No
     -
   * - `abs <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs>`__
     - No
     -
   * - `neg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg>`__
     - No
     -
   * - `min <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-min>`__
     - No
     -
   * - `max <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max>`__
     - No
     -
   * - `popc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc>`__
     - No
     -
   * - `clz <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz>`__
     - No
     -
   * - :ref:`bfind <libcudacxx-ptx-instructions-bfind>`
     - CCCL 3.0.0
     - ``<cuda/ptxs/bfind.h>``
   * - `fns <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns>`__
     - No
     -
   * - `brev <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev>`__
     - No
     -
   * - `bfe <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe>`__
     - No
     -
   * - `bfi <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi>`__
     - No
     -
   * - `szext <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext>`__
     - No
     -
   * - :ref:`bmsk <libcudacxx-ptx-instructions-bmsk>`
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/bmsk.h>``
   * - `dp4a <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a>`__
     - No
     -
   * - `dp2a <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a>`__
     - No
     -

.. list-table:: `Extended-Precision Integer Arithmetic Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `add.cc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-add-cc>`__
     - No
     -
   * - `addc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc>`__
     - No
     -
   * - `sub.cc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-sub-cc>`__
     - No
     -
   * - `subc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc>`__
     - No
     -
   * - `mad.cc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-mad-cc>`__
     - No
     -
   * - `madc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc>`__
     - No
     -

.. list-table:: `Floating-Point Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `testp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-testp>`__
     - No
     -
   * - `copysign <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign>`__
     - No
     -
   * - `add <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add>`__
     - No
     -
   * - `sub <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sub>`__
     - No
     -
   * - `mul <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul>`__
     - No
     -
   * - `fma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma>`__
     - No
     -
   * - `mad <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mad>`__
     - No
     -
   * - `div <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div>`__
     - No
     -
   * - `abs <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-abs>`__
     - No
     -
   * - `neg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-neg>`__
     - No
     -
   * - `min <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-min>`__
     - No
     -
   * - `max <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>`__
     - No
     -
   * - `rcp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp>`__
     - No
     -
   * - `rcp.approx.ftz.f64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp-approx-ftz-f64>`__
     - No
     -
   * - `sqrt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt>`__
     - No
     -
   * - `rsqrt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt>`__
     - No
     -
   * - `rsqrt.approx.ftz.f64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt-approx-ftz-f64>`__
     - No
     -
   * - `sin <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin>`__
     - No
     -
   * - `cos <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos>`__
     - No
     -
   * - `lg2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2>`__
     - No
     -
   * - `ex2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2>`__
     - No
     -
   * - `tanh <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh>`__
     - No
     -

.. list-table:: `Half Precision Floating-Point Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `add <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-add>`__
     - No
     -
   * - `sub <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-sub>`__
     - No
     -
   * - `mul <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul>`__
     - No
     -
   * - `fma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma>`__
     - No
     -
   * - `neg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-neg>`__
     - No
     -
   * - `abs <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-abs>`__
     - No
     -
   * - `min <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-min>`__
     - No
     -
   * - `max <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-max>`__
     - No
     -
   * - `tanh <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-tanh>`__
     - No
     -
   * - `ex2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-ex2>`__
     - No
     -

.. list-table:: `Comparison and Selection Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `set <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-set>`__
     - No
     -
   * - `setp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp>`__
     - No
     -
   * - `selp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp>`__
     - No
     -
   * - `slct <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-slct>`__
     - No
     -

.. list-table:: `Half Precision Comparison Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `set <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-set>`__
     - No
     -
   * - `setp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp>`__
     - No
     -

.. list-table:: `Logic and Shift Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `and <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and>`__
     - No
     -
   * - `or <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-or>`__
     - No
     -
   * - `xor <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-xor>`__
     - No
     -
   * - `not <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-not>`__
     - No
     -
   * - `cnot <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-cnot>`__
     - No
     -
   * - `lop3 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3>`__
     - No
     -
   * - `shf <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf>`__
     - No
     -
   * - :ref:`shl <libcudacxx-ptx-instructions-shl>`
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/shl.h>``
   * - :ref:`shr <libcudacxx-ptx-instructions-shr>`
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/shr.h>``

.. list-table:: `Data Movement and Conversion Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `mov <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov-2>`__
     - No
     -
   * - `shfl <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-deprecated>`__
     - No
     -
   * - :ref:`shfl.sync <libcudacxx-ptx-instructions-shfl_sync>`
     - Yes, CCCL 2.9.0 / CUDA 12.9
     - ``<cuda/ptxs/shfl_sync.h>``
   * - :ref:`prmt <libcudacxx-ptx-instructions-prmt>`
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/prmt.h>``
   * - :ref:`ld <libcudacxx-ptx-instructions-ld>`
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/ld.h>``
   * - `ld.global.nc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/ld.h>``
   * - `ldu <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu>`__
     - No
     -
   * - :ref:`st <libcudacxx-ptx-instructions-st>`
     - Yes, CCCL 3.0.0 / CUDA 13.0
     - ``<cuda/ptxs/st.h>``
   * - :ref:`st.async <libcudacxx-ptx-instructions-st-async>`
     - CCCL 2.3.0 / CUDA 12.4
     - ``<cuda/ptxs/st_async.h>``
   * - :ref:`st.bulk <libcudacxx-ptx-instructions-st-bulk>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/st_bulk.h>``
   * - :ref:`multimem.ld_reduce <libcudacxx-ptx-instructions-multimem-ld_reduce>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/multimem_ld_reduce.h>``
   * - :ref:`multimem.st <libcudacxx-ptx-instructions-multimem-st>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/multimem_st.h>``
   * - :ref:`multimem.red <libcudacxx-ptx-instructions-multimem-red>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/multimem_red.h>``
   * - :ref:`prefetch, prefetchu <libcudacxx-ptx-instructions-prefetch>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/prefetch.h>``
   * - `applypriority <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority>`__
     - No
     -
   * - `discard <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard>`__
     - No
     -
   * - `createpolicy <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy>`__
     - No
     -
   * - `isspacep <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep>`__
     - No
     -
   * - `cvta <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta>`__
     - No
     -
   * - `cvt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt>`__
     - No
     -
   * - `cvt.pack <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt-pack>`__
     - No
     -
   * - :ref:`mapa <libcudacxx-ptx-instructions-mapa>`
     - No
     -
   * - :ref:`getctarank <libcudacxx-ptx-instructions-getctarank>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/getctarank.h>``

.. list-table:: `Data Movement and Conversion Instructions: Asynchronous copy <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `cp.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async>`__
     - No
     -
   * - `cp.async.commit_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group>`__
     - No
     -
   * - `cp.async.wait_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all>`__
     - No
     -
   * - :ref:`cp.async.bulk <libcudacxx-ptx-instructions-cp-async-bulk>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/cp_async_bulk.h>``
   * - :ref:`cp.reduce.async.bulk <libcudacxx-ptx-instructions-cp-reduce-async-bulk>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/cp_reduce_async_bulk.h>``
   * - :ref:`cp.async.bulk.prefetch <libcudacxx-ptx-instructions-cp-async-bulk-prefetch>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/cp_async_bulk_prefetch.h>``
   * - :ref:`cp.async.bulk.tensor <libcudacxx-ptx-instructions-cp-async-bulk-tensor>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/cp_async_bulk_tensor.h>``
   * - :ref:`cp.reduce.async.bulk.tensor <libcudacxx-ptx-instructions-cp-reduce-async-bulk-tensor>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/cp_reduce_async_bulk_tensor.h>``
   * - :ref:`cp.async.bulk.prefetch.tensor <libcudacxx-ptx-instructions-cp-async-bulk-prefetch-tensor>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/cp_async_bulk_prefetch_tensor.h>``
   * - :ref:`cp.async.bulk.commit_group <libcudacxx-ptx-instructions-cp-async-bulk-commit_group>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/cp_async_bulk_commit_group.h>``
   * - :ref:`cp.async.bulk.wait_group <libcudacxx-ptx-instructions-cp-async-bulk-wait_group>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/cp_async_bulk_wait_group.h>``
   * - :ref:`applypriority.async.bulk <libcudacxx-ptx-instructions-applypriority-async-bulk>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/applypriority_async_bulk.h>``
   * - :ref:`tensormap.replace <libcudacxx-ptx-instructions-tensormap-replace>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/tensormap_replace.h>``

.. list-table:: `Texture Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `tex <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex>`__
     - No
     -
   * - `tld4 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tld4>`__
     - No
     -
   * - `txq <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-txq>`__
     - No
     -
   * - `istypep <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-istypep>`__
     - No
     -

.. list-table:: `Surface Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `suld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suld>`__
     - No
     -
   * - `sust <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust>`__
     - No
     -
   * - `sured <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sured>`__
     - No
     -
   * - `suq <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suq>`__
     - No
     -

.. list-table:: `Control Flow Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `{} <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-curly-braces>`__
     - No
     -
   * - `@ <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at>`__
     - No
     -
   * - `bra <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra>`__
     - No
     -
   * - `bra <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra>`__
     - No
     -
   * - `call <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-call>`__
     - No
     -
   * - `ret <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret>`__
     - No
     -
   * - :ref:`exit <libcudacxx-ptx-instructions-exit>`
     - CCCL 3.0.0
     - ``<cuda/ptxs/exit.h>``

.. list-table:: `Parallel Synchronization and Communication Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `bar, barrier <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier>`__
     - No
     -
   * - `bar.warp.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync>`__
     - No
     -
   * - :ref:`barrier.cluster <libcudacxx-ptx-instructions-barrier-cluster>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/barrier_cluster.h>``
   * - `membar <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence>`__
     - No
     -
   * - :ref:`fence <libcudacxx-ptx-instructions-fence>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/fence.h>``
   * - `atom <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom>`__
     - No
     -
   * - `red <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red>`__
     - No
     -
   * - :ref:`red.async <libcudacxx-ptx-instructions-mbarrier-red-async>`
     - CCCL 2.3.0 / CUDA 12.4
     - ``<cuda/ptxs/red_async.h>``
   * - `vote <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-deprecated>`__
     - No
     -
   * - `vote.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync>`__
     - No
     -
   * - `match.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync>`__
     - No
     -
   * - `activemask <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask>`__
     - No
     -
   * - `redux.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync>`__
     - No
     -
   * - `griddepcontrol <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol>`__
     - No
     -
   * - :ref:`elect.sync <libcudacxx-ptx-instructions-elect_sync>`
     - CCCL 3.1.0 / CUDA 13.1
     - ``<cuda/ptxs/elect_sync.h>``
   * - :ref:`fabric.submit <libcudacxx-ptx-instructions-fabric-submit>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/fabric_submit.h>``
   * - :ref:`fabric.try_get <libcudacxx-ptx-instructions-fabric-try-get>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/fabric_try_get.h>``
   * - :ref:`fabric.try_pullred <libcudacxx-ptx-instructions-fabric-try-pullred>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/fabric_try_pullred.h>``
   * - :ref:`fabric.try_put <libcudacxx-ptx-instructions-fabric-try-put>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/fabric_try_put.h>``
   * - :ref:`fabric.try_red <libcudacxx-ptx-instructions-fabric-try-red>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/fabric_try_red.h>``
   * - :ref:`fabric.wait <libcudacxx-ptx-instructions-fabric-wait>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/fabric_wait.h>``

.. list-table:: `Parallel Synchronization and Communication Instructions: mbarrier <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - :ref:`mbarrier.init <libcudacxx-ptx-instructions-mbarrier-init>`
     - CCCL 2.5.0 / CUDA Future
     - ``<cuda/ptxs/mbarrier_init.h>``
   * - :ref:`mbarrier.inval <libcudacxx-ptx-instructions-mbarrier_inval>`
     - CCCL 3.2.0 / CUDA 13.2
     - ``<cuda/ptxs/mbarrier_inval.h>``
   * - :ref:`mbarrier.complete_tx <libcudacxx-ptx-instructions-mbarrier-complete-tx>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_complete_tx.h>``
   * - :ref:`mbarrier.arrive <libcudacxx-ptx-instructions-mbarrier-arrive>`
     - CCCL 2.3.0 / CUDA 12.4
     - ``<cuda/ptxs/mbarrier_arrive.h>``
   * - :ref:`mbarrier.arrive_drop <libcudacxx-ptx-instructions-mbarrier-arrive-drop>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_arrive.h>``
   * - :ref:`mbarrier.arrive.expect_tx <libcudacxx-ptx-instructions-mbarrier-arrive-expect-tx>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_arrive.h>``
   * - :ref:`mbarrier.arrive.noComplete <libcudacxx-ptx-instructions-mbarrier-arrive-no-complete>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_arrive.h>``
   * - :ref:`cp.async.mbarrier.arrive <libcudacxx-ptx-instructions-cp-async-mbarrier-arrive>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/cp_async_mbarrier_arrive.h>``
   * - :ref:`cp.async.mbarrier.arrive.noinc <libcudacxx-ptx-instructions-cp-async-mbarrier-arrive-noinc>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/cp_async_mbarrier_arrive.h>``
   * - :ref:`mbarrier.check_layout <libcudacxx-ptx-instructions-mbarrier-check-layout>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_check_layout.h>``
   * - :ref:`mbarrier.expect_tx <libcudacxx-ptx-instructions-mbarrier-expect_tx>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/mbarrier_expect_tx.h>``
   * - :ref:`mbarrier.test_wait <libcudacxx-ptx-instructions-mbarrier-test_wait>`
     - CCCL 2.3.0 / CUDA 12.4
     - ``<cuda/ptxs/mbarrier_wait.h>``
   * - :ref:`mbarrier.test_wait.parity <libcudacxx-ptx-instructions-mbarrier-test-wait-parity>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_wait.h>``
   * - :ref:`mbarrier.try_wait <libcudacxx-ptx-instructions-mbarrier-try_wait>`
     - CCCL 2.3.0 / CUDA 12.4
     - ``<cuda/ptxs/mbarrier_wait.h>``
   * - :ref:`mbarrier.try_wait.parity <libcudacxx-ptx-instructions-mbarrier-try-wait-parity>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_wait.h>``
   * - :ref:`mbarrier.pending_count <libcudacxx-ptx-instructions-mbarrier-pending-count>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/mbarrier_pending_count.h>``
   * - :ref:`tensormap.cp_fenceproxy <libcudacxx-ptx-instructions-tensormap-cp_fenceproxy>`
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/tensormap_cp_fenceproxy.h>``
   * - :ref:`clusterlaunchcontrol.try_cancel <libcudacxx-ptx-instructions-clusterlaunchcontrol>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/clusterlaunchcontrol.h>``
   * - :ref:`clusterlaunchcontrol.query_cancel <libcudacxx-ptx-instructions-clusterlaunchcontrol>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/clusterlaunchcontrol.h>``

.. list-table:: `Warp Level Matrix Multiply-Accumulate Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `wmma.load <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-wmma-load>`__
     - No
     -
   * - `wmma.store <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-wmma-store>`__
     - No
     -
   * - `wmma.mma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma>`__
     - No
     -
   * - `mma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma>`__
     - No
     -
   * - :ref:`ldmatrix <libcudacxx-ptx-instructions-ldmatrix>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/ldmatrix.h>``
   * - :ref:`stmatrix <libcudacxx-ptx-instructions-stmatrix>`
     - CCCL 3.4.0 / CUDA 13.4
     - ``<cuda/ptxs/stmatrix.h>``
   * - `movmatrix <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-transpose-instruction-movmatrix>`__
     - No
     -
   * - `mma.sp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma>`__
     - No
     -

.. list-table:: `Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `wgmma.mma_async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async>`__
     - No
     -
   * - `wgmma.mma_async.sp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async-sp>`__
     - No
     -
   * - `wgmma.fence <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-fence>`__
     - No
     -
   * - `wgmma.commit_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-commit-group>`__
     - No
     -
   * - `wgmma.wait_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group>`__
     - No
     -

.. list-table:: `TensorCore 5th Generation Family Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - :ref:`tcgen05.alloc <libcudacxx-ptx-instructions-tcgen05-alloc>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_alloc.h>``
   * - :ref:`tcgen05.commit <libcudacxx-ptx-instructions-tcgen05-commit>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_commit.h>``
   * - :ref:`tcgen05.cp <libcudacxx-ptx-instructions-tcgen05-cp>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_cp.h>``
   * - :ref:`tcgen05.fence <libcudacxx-ptx-instructions-tcgen05-fence>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_fence.h>``
   * - :ref:`tcgen05.ld <libcudacxx-ptx-instructions-tcgen05-ld>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_ld.h>``
   * - :ref:`tcgen05.mma <libcudacxx-ptx-instructions-tcgen05-mma>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_mma.h>``
   * - :ref:`tcgen05.mma.ws <libcudacxx-ptx-instructions-tcgen05-mma-ws>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_mma_ws.h>``
   * - :ref:`tcgen05.shift <libcudacxx-ptx-instructions-tcgen05-shift>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_shift.h>``
   * - :ref:`tcgen05.st <libcudacxx-ptx-instructions-tcgen05-st>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_st.h>``
   * - :ref:`tcgen05.wait <libcudacxx-ptx-instructions-tcgen05-wait>`
     - CCCL 2.8 / CUDA 12.9
     - ``<cuda/ptxs/tcgen05_wait.h>``


.. list-table:: `Stack Manipulation Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `stacksave <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave>`__
     - No
     -
   * - `stackrestore <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore>`__
     - No
     -
   * - `alloca <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca>`__
     - No
     -

.. list-table:: `Video Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#video-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `vadd, vsub, vabsdiff, vmin, vmax <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax>`__
     - No
     -
   * - `vshl, vshr <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vshl-vshr>`__
     - No
     -
   * - `vmad <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vmad>`__
     - No
     -
   * - `vset <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vset>`__
     - No
     -

.. list-table:: `SIMD Video Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `vadd2, vsub2, vavrg2, vabsdiff2, vmin2, vmax2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2>`__
     - No
     -
   * - `vset2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset2>`__
     - No
     -
   * - `vadd4, vsub4, vavrg4, vabsdiff4, vmin4, vmax4 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4>`__
     - No
     -
   * - `vset4 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset4>`__
     - No
     -

.. list-table:: `Miscellaneous Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions>`__
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `brkpt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt>`__
     - No
     -
   * - `nanosleep <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep>`__
     - No
     -
   * - `pmevent <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent>`__
     - No
     -
   * - :ref:`trap <libcudacxx-ptx-instructions-trap>`
     - CCCL 3.0.0
     - ``<cuda/ptxs/trap.h>``
   * - :ref:`setmaxnreg <libcudacxx-ptx-instructions-setmaxnreg>`
     - CCCL 3.2.0 / CUDA 13.2
     - ``<cuda/ptxs/setmaxnreg.h>``

.. list-table:: `Special registers <libcudacxx-ptx-instructions-special-registers>`
   :widths: 40 30 30
   :header-rows: 1

   * - Instruction
     - Available in libcu++
     - Header
   * - `tid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-tid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `ntid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ntid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `laneid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-laneid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `warpid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-warpid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `nwarpid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nwarpid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `ctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ctaid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `nctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nctaid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `smid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-smid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `nsmid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nsmid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `gridid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-gridid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `is_explicit_cluster <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-is-explicit-cluster>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `clusterid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clusterid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `nclusterid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nclusterid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `cluster_ctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctaid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `cluster_nctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-nctaid>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `cluster_ctarank <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctarank>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `cluster_nctarank <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-nctarank>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `lanemask_eq <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-eq>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `lanemask_le <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-le>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `lanemask_lt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-lt>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `lanemask_ge <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-ge>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `lanemask_gt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-gt>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `clock, clock_hi <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock-clock-hi>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `clock64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock64>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `pm0 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-pm0-pm7>`__
     - No
     -
   * - `pm0_64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-pm0-64-pm7-64>`__
     - No
     -
   * - `envreg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-envreg-32>`__
     - No
     -
   * - `globaltimer, globaltimer_lo, globaltimer_hi <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `reserved_smem_offset_begin, reserved_smem_offset_end, reserved_smem_offset_cap, reserved_smem_offset_2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-reserved-smem-offset-begin-reserved-smem-offset-end-reserved-smem-offset-cap-reserved-smem-offset-2>`__
     - No
     -
   * - `total_smem_size <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-total-smem-size>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `aggr_smem_size <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-aggr-smem-size>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `dynamic_smem_size <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-dynamic-smem-size>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
   * - `current_graph_exec <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-current-graph-exec>`__
     - CCCL 2.4.0 / CUDA 12.5
     - ``<cuda/ptxs/get_sreg.h>``
