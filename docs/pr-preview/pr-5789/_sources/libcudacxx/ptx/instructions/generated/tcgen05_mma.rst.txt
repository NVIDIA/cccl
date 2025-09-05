..
   This file was automatically generated. Do not edit.

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], a_desc, b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_1_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[4],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, disable_output_lane, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::2 }
   template <cuda::ptx::dot_kind Kind>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_2_t,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     const uint32_t (&disable_output_lane)[8],
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d, scale_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f
   // .kind      = { .kind::f16, .kind::tf32 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <int N32, cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d,
     cuda::ptx::n32_t<N32> scale_input_d);

tcgen05.mma.cta_group::1.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f16
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::tf32
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::f8f6f4
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::i8
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind [d_tmem], [a_tmem], b_desc, idesc, enable_input_d; // PTX ISA 86, SM_100a, SM_100f, SM_103a, SM_103f, SM_110a, SM_110f
   // .kind      = { .kind::f16, .kind::tf32, .kind::f8f6f4, .kind::i8 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint32_t a_tmem,
     uint64_t b_desc,
     uint32_t idesc,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_fill(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_fill(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_fill(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_fill(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::fill
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::fill [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_fill(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_use(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_use(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_use(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_use(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::use
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::use [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_use(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_lastuse(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_lastuse(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::lastuse
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::lastuse [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_lastuse(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_discard(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_collector_a_discard(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2x_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale.scale_vec::1X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::1X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf8f6f4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_1x_tmem_a_collector_a_discard(
     cuda::ptx::kind_mxf8f6f4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::2X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::2X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4, .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_kind Kind, cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_2_tmem_a_collector_a_discard(
     cuda::ptx::kind_t<Kind> kind,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);

tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X.collector::a::discard
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
.. code-block:: cuda

   // tcgen05.mma.cta_group.kind.block_scale.scale_vec::4X.collector::a::discard [d_tmem], a_desc, b_desc, idesc, [scale_A_tmem], [scale_B_tmem], enable_input_d; // PTX ISA 86, SM_100a, SM_103a, SM_110a
   // .kind      = { .kind::mxf4nvf4 }
   // .cta_group = { .cta_group::1, .cta_group::2 }
   template <cuda::ptx::dot_cta_group Cta_Group>
   __device__ static inline void tcgen05_mma_block_scale_vec_4x_tmem_a_collector_a_discard(
     cuda::ptx::kind_mxf4nvf4_t,
     cuda::ptx::cta_group_t<Cta_Group> cta_group,
     uint32_t d_tmem,
     uint64_t a_desc,
     uint64_t b_desc,
     uint32_t idesc,
     uint32_t scale_A_tmem,
     uint32_t scale_B_tmem,
     bool enable_input_d);
