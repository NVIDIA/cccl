# barrier.cluster

- PTX ISA: [`barrier.cluster`](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#parallel-synchronization-and-communication-instructions-barrier-cluster)

Similar functionality is provided through the builtins
`__cluster_barrier_arrive(), __cluster_barrier_arrive_relaxed(),
__cluster_barrier_wait()`, as well as the `cooperative_groups::cluster_group`
[API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cluster-group).

The `.aligned` variants of the instructions are not exposed.

**barrier_cluster**:
```cuda
// barrier.cluster.arrive; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_arrive();

// barrier.cluster.wait; // PTX ISA 78, SM_90
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_wait();

// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .release }
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_release_t);

// barrier.cluster.arrive.sem; // PTX ISA 80, SM_90
// .sem       = { .relaxed }
// Marked volatile
template <typename=void>
__device__ static inline void barrier_cluster_arrive(
  cuda::ptx::sem_relaxed_t);

// barrier.cluster.wait.sem; // PTX ISA 80, SM_90
// .sem       = { .acquire }
// Marked volatile and as clobbering memory
template <typename=void>
__device__ static inline void barrier_cluster_wait(
  cuda::ptx::sem_acquire_t);
```
