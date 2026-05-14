..
   This file was automatically generated. Do not edit.

tcgen05.mma.sp.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::f8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32 [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::fill [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::use [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::lastuse [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], a_desc, b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block16.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block16.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block16.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_103a, SM_107a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block16_tmem_a_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf8f6f4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf8f6f4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::1.kind::mxf4nvf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.sp.cta_group::2.kind::mxf4nvf4.block_scale.block32.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.sp.cta_group.kind.block_scale.block32.collector::a::discard [d_tmem], [a_tmem], b_desc, [sp_info_tmem], idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 88, SM_100a, SM_100f, SM_103a, SM_103f, SM_107a, SM_107f, SM_110a, SM_110f
   // .kind      = { .kind::mxf8f6f4, .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_sp_block_scale_block32_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t sp_info_tmem,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);
