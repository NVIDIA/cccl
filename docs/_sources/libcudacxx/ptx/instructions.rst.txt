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
   instructions/mbarrier_test_wait
   instructions/mbarrier_try_wait
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


Instructions by section
-----------------------

.. list-table:: `Integer Arithmetic Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `sad <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-sad>`__
     - No
   * - `div <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-div>`__
     - No
   * - `rem <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-rem>`__
     - No
   * - `abs <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-abs>`__
     - No
   * - `neg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-neg>`__
     - No
   * - `min <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-min>`__
     - No
   * - `max <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-max>`__
     - No
   * - `popc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-popc>`__
     - No
   * - `clz <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-clz>`__
     - No
   * - `bfind <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfind>`__
     - CCCL 3.0.0
   * - `fns <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-fns>`__
     - No
   * - `brev <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-brev>`__
     - No
   * - `bfe <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfe>`__
     - No
   * - `bfi <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bfi>`__
     - No
   * - `szext <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-szext>`__
     - No
   * - `bmsk <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-bmsk>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
   * - `dp4a <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp4a>`__
     - No
   * - `dp2a <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#integer-arithmetic-instructions-dp2a>`__
     - No

.. list-table:: `Extended-Precision Integer Arithmetic Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-integer-arithmetic-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `add.cc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-add-cc>`__
     - No
   * - `addc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-addc>`__
     - No
   * - `sub.cc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-sub-cc>`__
     - No
   * - `subc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-subc>`__
     - No
   * - `mad.cc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-mad-cc>`__
     - No
   * - `madc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#extended-precision-arithmetic-instructions-madc>`__
     - No

.. list-table:: `Floating-Point Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `testp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-testp>`__
     - No
   * - `copysign <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-copysign>`__
     - No
   * - `add <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-add>`__
     - No
   * - `sub <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sub>`__
     - No
   * - `mul <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mul>`__
     - No
   * - `fma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-fma>`__
     - No
   * - `mad <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-mad>`__
     - No
   * - `div <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-div>`__
     - No
   * - `abs <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-abs>`__
     - No
   * - `neg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-neg>`__
     - No
   * - `min <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-min>`__
     - No
   * - `max <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-max>`__
     - No
   * - `rcp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp>`__
     - No
   * - `rcp.approx.ftz.f64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rcp-approx-ftz-f64>`__
     - No
   * - `sqrt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sqrt>`__
     - No
   * - `rsqrt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt>`__
     - No
   * - `rsqrt.approx.ftz.f64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-rsqrt-approx-ftz-f64>`__
     - No
   * - `sin <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-sin>`__
     - No
   * - `cos <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-cos>`__
     - No
   * - `lg2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-lg2>`__
     - No
   * - `ex2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-ex2>`__
     - No
   * - `tanh <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#floating-point-instructions-tanh>`__
     - No

.. list-table:: `Half Precision Floating-Point Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `add <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-add>`__
     - No
   * - `sub <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-sub>`__
     - No
   * - `mul <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul>`__
     - No
   * - `fma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-fma>`__
     - No
   * - `neg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-neg>`__
     - No
   * - `abs <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-abs>`__
     - No
   * - `min <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-min>`__
     - No
   * - `max <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-max>`__
     - No
   * - `tanh <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-tanh>`__
     - No
   * - `ex2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-ex2>`__
     - No

.. list-table:: `Comparison and Selection Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `set <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-set>`__
     - No
   * - `setp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-setp>`__
     - No
   * - `selp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-selp>`__
     - No
   * - `slct <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#comparison-and-selection-instructions-slct>`__
     - No

.. list-table:: `Half Precision Comparison Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `set <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-set>`__
     - No
   * - `setp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-comparison-instructions-setp>`__
     - No

.. list-table:: `Logic and Shift Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `and <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-and>`__
     - No
   * - `or <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-or>`__
     - No
   * - `xor <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-xor>`__
     - No
   * - `not <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-not>`__
     - No
   * - `cnot <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-cnot>`__
     - No
   * - `lop3 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3>`__
     - No
   * - `shf <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shf>`__
     - No
   * - `shl <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shl>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
   * - `shr <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-shr>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0

.. list-table:: `Data Movement and Conversion Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `mov <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-mov-2>`__
     - No
   * - `shfl <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-deprecated>`__
     - No
   * - `shfl.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-shfl-sync>`__
     - Yes, CCCL 2.9.0 / CUDA 12.9
   * - `prmt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prmt>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
   * - `ld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
   * - `ld.global.nc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ld-global-nc>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
   * - `ldu <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-ldu>`__
     - No
   * - `st <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-st>`__
     - Yes, CCCL 3.0.0 / CUDA 13.0
   * - :ref:`st.async <libcudacxx-ptx-instructions-st-async>`
     - CCCL 2.3.0 / CUDA 12.4
   * - :ref:`st.bulk <libcudacxx-ptx-instructions-st-bulk>`
     - CCCL 2.8 / CUDA 12.9
   * - `multimem.ld_reduce, multimem.st, multimem.red <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-multimem-ld-reduce-multimem-st-multimem-red>`__
     - CCCL 2.8 / CUDA 12.9
   * - `prefetch, prefetchu <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-prefetch-prefetchu>`__
     - No
   * - `applypriority <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-applypriority>`__
     - No
   * - `discard <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-discard>`__
     - No
   * - `createpolicy <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-createpolicy>`__
     - No
   * - `isspacep <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep>`__
     - No
   * - `cvta <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvta>`__
     - No
   * - `cvt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt>`__
     - No
   * - `cvt.pack <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt-pack>`__
     - No
   * - :ref:`mapa <libcudacxx-ptx-instructions-mapa>`
     - No
   * - :ref:`getctarank <libcudacxx-ptx-instructions-getctarank>`
     - CCCL 2.4.0 / CUDA 12.5

.. list-table:: `Data Movement and Conversion Instructions: Asynchronous copy <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-asynchronous-copy>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `cp.async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async>`__
     - No
   * - `cp.async.commit_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-commit-group>`__
     - No
   * - `cp.async.wait_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-wait-group-cp-async-wait-all>`__
     - No
   * - :ref:`cp.async.bulk <libcudacxx-ptx-instructions-cp-async-bulk>`
     - CCCL 2.4.0 / CUDA 12.5
   * - :ref:`cp.reduce.async.bulk <libcudacxx-ptx-instructions-cp-reduce-async-bulk>`
     - CCCL 2.4.0 / CUDA 12.5
   * - `cp.async.bulk.prefetch <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch>`__
     - No
   * - :ref:`cp.reduce.async.bulk <libcudacxx-ptx-instructions-cp-async-bulk-tensor>`
     - CCCL 2.4.0 / CUDA 12.5
   * - :ref:`cp.reduce.async.bulk.tensor <libcudacxx-ptx-instructions-cp-reduce-async-bulk-tensor>`
     - CCCL 2.4.0 / CUDA 12.5
   * - `cp.async.bulk.prefetch.tensor <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk-prefetch-tensor>`__
     - No
   * - :ref:`cp.async.bulk.commit_group <libcudacxx-ptx-instructions-cp-async-bulk-commit_group>`
     - CCCL 2.4.0 / CUDA 12.5
   * - :ref:`cp.async.bulk.wait_group <libcudacxx-ptx-instructions-cp-async-bulk-wait_group>`
     - CCCL 2.4.0 / CUDA 12.5
   * - :ref:`tensormap.replace <libcudacxx-ptx-instructions-tensormap-replace>`
     - CCCL 2.4.0 / CUDA 12.5

.. list-table:: `Texture Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `tex <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tex>`__
     - No
   * - `tld4 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-tld4>`__
     - No
   * - `txq <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-txq>`__
     - No
   * - `istypep <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#texture-instructions-istypep>`__
     - No

.. list-table:: `Surface Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `suld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suld>`__
     - No
   * - `sust <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sust>`__
     - No
   * - `sured <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-sured>`__
     - No
   * - `suq <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#surface-instructions-suq>`__
     - No

.. list-table:: `Control Flow Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `{} <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-curly-braces>`__
     - No
   * - `@ <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-at>`__
     - No
   * - `bra <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra>`__
     - No
   * - `bra <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-bra>`__
     - No
   * - `call <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-call>`__
     - No
   * - `ret <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-ret>`__
     - No
   * - `exit <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#control-flow-instructions-exit>`__
     - CCCL 3.0.0

.. list-table:: `Parallel Synchronization and Communication Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `bar, barrier <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-barrier>`__
     - No
   * - `bar.warp.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-bar-warp-sync>`__
     - No
   * - :ref:`barrier.cluster <libcudacxx-ptx-instructions-barrier-cluster>`
     - CCCL 2.4.0 / CUDA 12.5
   * - `membar <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-membar-fence>`__
     - No
   * - :ref:`fence <libcudacxx-ptx-instructions-fence>`
     - CCCL 2.4.0 / CUDA 12.5
   * - `atom <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-atom>`__
     - No
   * - `red <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-red>`__
     - No
   * - :ref:`red.async <libcudacxx-ptx-instructions-mbarrier-red-async>`
     - CCCL 2.3.0 / CUDA 12.4
   * - `vote <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-deprecated>`__
     - No
   * - `vote.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-vote-sync>`__
     - No
   * - `match.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-match-sync>`__
     - No
   * - `activemask <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-activemask>`__
     - No
   * - `redux.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-redux-sync>`__
     - No
   * - `griddepcontrol <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-griddepcontrol>`__
     - No
   * - `elect.sync <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-elect-sync>`__
     - CCCL 3.1.0 / CUDA 13.1

.. list-table:: `Parallel Synchronization and Communication Instructions: mbarrier <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - :ref:`mbarrier.init <libcudacxx-ptx-instructions-mbarrier-init>`
     - CCCL 2.5.0 / CUDA Future
   * - `mbarrier.inval <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-inval>`__
     - CCCL 3.2.0 / CUDA 13.2
   * - `mbarrier.complete_tx <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-complete-tx>`__
     - No
   * - :ref:`mbarrier.arrive <libcudacxx-ptx-instructions-mbarrier-arrive>`
     - CCCL 2.3.0 / CUDA 12.4
   * - `mbarrier.arrive_drop <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-arrive-drop>`__
     - No
   * - `cp.async.mbarrier.arrive <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-cp-async-mbarrier-arrive>`__
     - CCCL 2.8 / CUDA 12.9
   * - `mbarrier.expect_tx <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-expect-tx>`__
     - CCCL 2.8 / CUDA 12.9
   * - :ref:`mbarrier.test_wait <libcudacxx-ptx-instructions-mbarrier-test_wait>`
     - CCCL 2.3.0 / CUDA 12.4
   * - :ref:`mbarrier.try_wait <libcudacxx-ptx-instructions-mbarrier-try_wait>`
     - CCCL 2.3.0 / CUDA 12.4
   * - `mbarrier.pending_count <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-mbarrier-pending-count>`__
     - No
   * - :ref:`tensormap.cp_fenceproxy <libcudacxx-ptx-instructions-tensormap-cp_fenceproxy>`
     - CCCL 2.4.0 / CUDA 12.5
   * - :ref:`clusterlaunchcontrol.try_cancel <libcudacxx-ptx-instructions-clusterlaunchcontrol>`
     - CCCL 2.8 / CUDA 12.9
   * - :ref:`clusterlaunchcontrol.query_cancel <libcudacxx-ptx-instructions-clusterlaunchcontrol>`
     - CCCL 2.8 / CUDA 12.9

.. list-table:: `Warp Level Matrix Multiply-Accumulate Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-multiply-accumulate-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `wmma.load <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-wmma-load>`__
     - No
   * - `wmma.store <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-wmma-store>`__
     - No
   * - `wmma.mma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-wmma-mma>`__
     - No
   * - `mma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma>`__
     - No
   * - `ldmatrix <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix>`__
     - No
   * - `stmatrix <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-store-instruction-stmatrix>`__
     - No
   * - `movmatrix <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-transpose-instruction-movmatrix>`__
     - No
   * - `mma.sp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma>`__
     - No

.. list-table:: `Asynchronous Warpgroup Level Matrix Multiply-Accumulate Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `wgmma.mma_async <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async>`__
     - No
   * - `wgmma.mma_async.sp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async-sp>`__
     - No
   * - `wgmma.fence <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-fence>`__
     - No
   * - `wgmma.commit_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-commit-group>`__
     - No
   * - `wgmma.wait_group <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-wait-group>`__
     - No

.. list-table:: `TensorCore 5th Generation Family Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-family-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `tcgen05.alloc <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-dealloc-tcgen05-relinquish-alloc-permit>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.commit <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-commit>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.cp <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-cp>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.fence <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-fence>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.ld <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-ld>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.mma <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-mma>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.mma.ws <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-mma-ws>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.shift <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-shift>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.st <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-st>`__
     - CCCL 2.8 / CUDA 12.9
   * - `tcgen05.wait <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tensorcore-5th-generation-instructions-tcgen05-alloc-tcgen05-wait>`__
     - CCCL 2.8 / CUDA 12.9


.. list-table:: `Stack Manipulation Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `stacksave <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stacksave>`__
     - No
   * - `stackrestore <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-stackrestore>`__
     - No
   * - `alloca <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#stack-manipulation-instructions-alloca>`__
     - No

.. list-table:: `Video Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#video-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `vadd, vsub, vabsdiff, vmin, vmax <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vadd-vsub-vabsdiff-vmin-vmax>`__
     - No
   * - `vshl, vshr <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vshl-vshr>`__
     - No
   * - `vmad <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vmad>`__
     - No
   * - `vset <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#scalar-video-instructions-vset>`__
     - No

.. list-table:: `SIMD Video Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `vadd2, vsub2, vavrg2, vabsdiff2, vmin2, vmax2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd2-vsub2-vavrg2-vabsdiff2-vmin2-vmax2>`__
     - No
   * - `vset2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset2>`__
     - No
   * - `vadd4, vsub4, vavrg4, vabsdiff4, vmin4, vmax4 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vadd4-vsub4-vavrg4-vabsdiff4-vmin4-vmax4>`__
     - No
   * - `vset4 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#simd-video-instructions-vset4>`__
     - No

.. list-table:: `Miscellaneous Instructions <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions>`__
   :widths: 50 50
   :header-rows: 1

   * - Instruction
     - Available in libcu++
   * - `brkpt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-brkpt>`__
     - No
   * - `nanosleep <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-nanosleep>`__
     - No
   * - `pmevent <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-pmevent>`__
     - No
   * - `trap <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-trap>`__
     - CCCL 3.0.0
   * - `setmaxnreg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#miscellaneous-instructions-setmaxnreg>`__
     - CCCL 3.2.0 / CUDA 13.2

.. list-table:: `Special registers <libcudacxx-ptx-instructions-special-registers>`
   :widths: 25 25 25 25
   :header-rows: 1

   * - Instruction
     - PTX ISA
     - SM Version
     - Available in libcu++
   * - `tid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-tid>`__
     - 20
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `ntid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ntid>`__
     - 20
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `laneid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-laneid>`__
     - 13
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `warpid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-warpid>`__
     - 13
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `nwarpid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nwarpid>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `ctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-ctaid>`__
     - 20
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `nctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nctaid>`__
     - 20
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `smid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-smid>`__
     - 13
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `nsmid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nsmid>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `gridid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-gridid>`__
     - 30
     - 30
     - CCCL 2.4.0 / CUDA 12.5
   * - `is_explicit_cluster <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-is-explicit-cluster>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `clusterid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clusterid>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `nclusterid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-nclusterid>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `cluster_ctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctaid>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `cluster_nctaid <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-nctaid>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `cluster_ctarank <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-ctarank>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `cluster_nctarank <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-cluster-nctarank>`__
     - 78
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `lanemask_eq <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-eq>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `lanemask_le <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-le>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `lanemask_lt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-lt>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `lanemask_ge <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-ge>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `lanemask_gt <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-lanemask-gt>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `clock, clock_hi <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock-clock-hi>`__
     - 10
     - All
     - CCCL 2.4.0 / CUDA 12.5
   * - `clock64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-clock64>`__
     - 20
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `pm0 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-pm0-pm7>`__
     -
     -
     - No
   * - `pm0_64 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-pm0-64-pm7-64>`__
     -
     -
     - No
   * - `envreg <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-envreg-32>`__
     -
     -
     - No
   * - `globaltimer, globaltimer_lo, globaltimer_hi <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-globaltimer-globaltimer-lo-globaltimer-hi>`__
     - 31
     - 31
     - CCCL 2.4.0 / CUDA 12.5
   * - `reserved_smem_offset_begin, reserved_smem_offset_end, reserved_smem_offset_cap, reserved_smem_offset_2 <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-reserved-smem-offset-begin-reserved-smem-offset-end-reserved-smem-offset-cap-reserved-smem-offset-2>`__
     -
     -
     - No
   * - `total_smem_size <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-total-smem-size>`__
     - 41
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `aggr_smem_size <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-aggr-smem-size>`__
     - 81
     - 90
     - CCCL 2.4.0 / CUDA 12.5
   * - `dynamic_smem_size <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-dynamic-smem-size>`__
     - 41
     - 20
     - CCCL 2.4.0 / CUDA 12.5
   * - `current_graph_exec <https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#special-registers-current-graph-exec>`__
     - 80
     - 50
     - CCCL 2.4.0 / CUDA 12.5
