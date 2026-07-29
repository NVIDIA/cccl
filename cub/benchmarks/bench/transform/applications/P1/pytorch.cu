// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// ============================================================================
// NVBench benchmarks for chained elementwise operations was put together by Matthias Jouanneaux (DevTech). It mimics
// how pytorch uses element-wise kernels (i.e. cub::DeviceTransform) and also tries to preserve pytorch's operators and
// utility types. The main difference to ordinary CCCL benchmarks is the chaining of several operations in the
// benchmark's critical section. Furthermore, there are 4 types of work loads covering the combinations of few vs. many
// input buffers and few vs. many instructions in the kernel. For more discussion see:
// https://github.com/NVIDIA-dev/cccl_private/issues/639
// ============================================================================

// %RANGE% TUNE_BIF_BIAS bif -16:16:4
// %RANGE% TUNE_ALGORITHM alg 0:4:1
// %RANGE% TUNE_THREADS tpb 128:1024:128

// for TUNE_ALGORITHM == 1 (vectorized), this is the number of vectors per thread, which is similar in spirit
// %RANGE% TUNE_UNROLL_FACTOR unrl 1:4:1

// those parameters only apply if TUNE_ALGORITHM == 0 (prefetch)
// %RANGE% TUNE_PREFETCH_MULT pref 1:3:1

// those parameters only apply if TUNE_ALGORITHM == 1 (vectorized)
// %RANGE% TUNE_VEC_SIZE_POW2 vsp2 1:6:1

#if !TUNE_BASE && TUNE_ALGORITHM != 0 && (TUNE_PREFETCH_MULT != 1)
#  error "Non-prefetch algorithms require prefetch multiple to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 0 && (TUNE_PREFETCH_MULT != 1)

#if !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1)
#  error "Non-vectorized algorithms require vector size to be 1 since they ignore the parameters"
#endif // !TUNE_BASE && TUNE_ALGORITHM != 1 && (TUNE_VEC_SIZE_POW2 != 1)

#include <thrust/device_vector.h>

#include <cuda/iterator>
#include <cuda/random>
#include <cuda/std/algorithm.max.h>
#include <cuda/std/algorithm.min.h>
#include <cuda/std/algorithm.transform.h>
#include <cuda/std/cmath>
#include <cuda/std/execution>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include "../../common.h"
#include "bfloat16.h"

// ============================================================================
// at::opmath_type<T> — the compute type for intermediate math
// float for both float and BFloat16 (ATen/OpMathType.h)
// ============================================================================

template <typename T>
struct opmath_type_impl
{
  using type = T;
};
template <>
struct opmath_type_impl<BFloat16>
{
  using type = float;
};
template <typename T>
using opmath_type = typename opmath_type_impl<T>::type;

// ============================================================================
// Replicate ATen/c10 helpers without ATen dependencies.
// Each wrapper is annotated with the ATen source it replicates.
// ============================================================================

// c10::div_floor_floating (c10/util/generic_math.h:34)
template <typename scalar_t>
__device__ __forceinline__ scalar_t div_floor_floating(scalar_t a, scalar_t b)
{
  if (b == 0)
  {
    return a / b;
  }

  auto mod = std::fmod(a, b);
  auto div = (a - mod) / b;
  if ((mod != 0) && (b < 0) != (mod < 0))
  {
    div -= scalar_t(1);
  }

  scalar_t floordiv;
  if (div != 0)
  {
    floordiv = std::floor(div);
    if (div - floordiv > scalar_t(0.5))
    {
      floordiv += scalar_t(1.0);
    }
  }
  else
  {
    floordiv = ::copysignf(scalar_t(0), a / b);
  }
  return floordiv;
}

// is_lerp_weight_small + lerp (native/Lerp.h:11,21)
template <typename scalar_t>
__device__ __forceinline__ bool is_lerp_weight_small(scalar_t weight)
{
  return std::abs(weight) < scalar_t(0.5);
}

template <typename scalar_t, typename weight_t>
__device__ __forceinline__ scalar_t aten_lerp(scalar_t self_, scalar_t end_, weight_t weight_)
{
  using opmath_t        = opmath_type<scalar_t>;
  using opmath_weight_t = opmath_type<weight_t>;

  opmath_t self          = self_;
  opmath_t end           = end_;
  opmath_weight_t weight = weight_;

  return is_lerp_weight_small(weight) ? self + weight * (end - self) : end - (end - self) * (opmath_t(1) - weight);
}

// pointwise_op_impl (native/cuda/DeviceAddCmulCdiv.cuh:9)
template <typename opmath_t, typename Op>
__device__ __forceinline__ opmath_t
pointwise_op_impl(opmath_t input, opmath_t tensor1, opmath_t tensor2, opmath_t alpha, Op op)
{
  if (alpha == opmath_t(1))
  {
    if constexpr (std::is_same_v<Op, std::multiplies<opmath_t>> && std::is_floating_point_v<opmath_t>)
    {
      return std::fma(tensor1, tensor2, input);
    }
    else
    {
      return input + op(tensor1, tensor2);
    }
  }
  if constexpr (std::is_floating_point_v<opmath_t>)
  {
    return std::fma(alpha, op(tensor1, tensor2), input);
  }
  else
  {
    return input + alpha * op(tensor1, tensor2);
  }
}

// DivFunctor (native/cuda/BinaryInternal.h:20)
template <typename scalar_t>
struct DivFunctor
{
  __device__ scalar_t operator()(scalar_t a, scalar_t b) const
  {
    return a / b;
  }
};

// MulFunctor (native/cuda/BinaryInternal.h:27)
template <typename T>
struct MulFunctor
{
  __device__ T operator()(T a, T b) const
  {
    return a * b;
  }
};

// CUDAFunctorOnSelf_add — torchgen-generated ufunc functor for add(tensor, scalar)
// (torchgen/dest/ufunc.py, native/ufunc/add.h:14)
template <typename scalar_t>
struct CUDAFunctorOnSelf_add
{
  using opmath_t = opmath_type<scalar_t>;
  opmath_t other_;
  opmath_t alpha_;
  CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha)
      : other_(other)
      , alpha_(alpha)
  {}
  __device__ scalar_t operator()(scalar_t self) const
  {
    return static_cast<opmath_t>(self) + alpha_ * other_;
  }
};

// CUDAFunctor_add — torchgen-generated ufunc functor for add(tensor, tensor)
// (torchgen/dest/ufunc.py, native/ufunc/add.h:14)
template <typename scalar_t>
struct CUDAFunctor_add
{
  using opmath_t = opmath_type<scalar_t>;
  opmath_t alpha_;
  CUDAFunctor_add(opmath_t alpha)
      : alpha_(alpha)
  {}
  __device__ scalar_t operator()(scalar_t self, scalar_t other) const
  {
    return static_cast<opmath_t>(self) + alpha_ * static_cast<opmath_t>(other);
  }
};

// AbsFunctor (native/cuda/AbsKernel.cu:11)
template <typename scalar_t>
struct AbsFunctor
{
  __device__ __forceinline__ scalar_t operator()(const scalar_t a) const
  {
    return std::abs(a);
  }
};

// CompareFunctor (native/cuda/CompareKernels.cu:14 / 17)
enum class OpType
{
  GE,
  GT,
  LE,
  LT
};

template <typename scalar_t>
struct CompareFunctor
{
  constexpr CompareFunctor(OpType op)
      : op_(op) {};
  OpType op_;
  __device__ __forceinline__ bool operator()(scalar_t a, scalar_t b) const
  {
    if (op_ == OpType::GE)
    {
      return a >= b;
    }
    else if (op_ == OpType::GT)
    {
      return a > b;
    }
    else if (op_ == OpType::LE)
    {
      return a <= b;
    }
    else
    { // LT
      return a < b;
    }
  }
};

// ============================================================================
// RNG helpers
// ============================================================================

template <typename T>
struct normal_gen
{
  float mean, stddev;
  int64_t offset;
  __host__ __device__ T operator()(int64_t idx) const
  {
    cuda::pcg64 rng(42);
    rng.discard(offset + idx);
    cuda::std::normal_distribution<float> dist(mean, stddev);
    return T(dist(rng));
  }
};

template <typename T>
void fill_normal(thrust::device_vector<T>& v, int64_t n, int buf_idx)
{
  v.resize(n);
  cuda::std::transform(
    cuda::execution::gpu,
    cuda::counting_iterator<int64_t, int64_t>(0),
    cuda::counting_iterator<int64_t, int64_t>(n),
    v.begin(),
    normal_gen<T>{0.0f, 1.0f, buf_idx * n});
}

// ============================================================================
// Helper to call DeviceTransform::Transform with the tuning policy
// ============================================================================

template <typename... Inputs, typename Output, typename TransformOp>
void transform(cuda::std::tuple<Inputs...> inputs, Output output, int64_t n, TransformOp op, cudaStream_t stream)
{
  auto env = cuda::std::execution::env{
    cuda::stream_ref{stream}
#if !TUNE_BASE
    ,
    cuda::execution::tune(policy_selector{})
#endif // !TUNE_BASE
  };
  cub::DeviceTransform::Transform(inputs, output, n, op, env);
}

template <typename Input, typename Output, typename TransformOp>
void transform(Input input, Output output, int64_t n, TransformOp op, cudaStream_t stream)
{
  transform(cuda::std::make_tuple(input), output, n, op, stream);
}

// ============================================================================
// Element types
// ============================================================================

#ifdef TUNE_T
using element_types = nvbench::type_list<TUNE_T>;
#else
using element_types = nvbench::type_list<float, BFloat16>;
#endif

// ============================================================================
// many_inputs_many_instructions
//
// div_floor -> div_trunc -> div -> atan2 -> hypot ->
// xlogy -> xlog1py -> logaddexp -> logaddexp2 -> pow
// 11 inputs, 10 binary ops
// ============================================================================

template <typename T>
static void many_inputs_many_instructions(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");

  constexpr int num_in = 11;
  thrust::device_vector<T> in[num_in];
  for (int i = 0; i < num_in; i++)
  {
    fill_normal(in[i], n, i);
  }
  thrust::device_vector<T> tmpA(n, thrust::no_init), tmpB(n, thrust::no_init);

  T* d_in[num_in];
  for (int i = 0; i < num_in; i++)
  {
    d_in[i] = thrust::raw_pointer_cast(in[i].data());
  }
  T* d_a = thrust::raw_pointer_cast(tmpA.data());
  T* d_b = thrust::raw_pointer_cast(tmpB.data());

  state.add_element_count(n);
  state.add_global_memory_reads<T>(20L * n);
  state.add_global_memory_writes<T>(10L * n);

  // logaddexp2 captures inv_log_2 — native/cuda/LogAddExpKernel.cu:272
  using opmath_t       = opmath_type<T>;
  const auto inv_log_2 = static_cast<opmath_t>(1.0 / 0.693147180559945309417232121458176);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](const nvbench::launch& launch) {
    const auto s = launch.get_stream().get_stream();

    // div_floor: native/cuda/BinaryDivFloorKernel.cu:72, helper c10/util/generic_math.h:34
    transform(
      cuda::std::make_tuple(d_in[0], d_in[1]),
      d_a,
      n,
      [] __device__(T a, T b) -> T {
        return div_floor_floating(a, b);
      },
      s);

    // div_trunc: native/cuda/BinaryDivTruncKernel.cu:42
    transform(
      cuda::std::make_tuple(d_a, d_in[2]),
      d_b,
      n,
      [] __device__(T a, T b) -> T {
        return std::trunc(a / b);
      },
      s);

    // div: native/cuda/BinaryDivTrueKernel.cu:54, DivFunctor in native/cuda/BinaryInternal.h:20
    transform(cuda::std::make_tuple(d_b, d_in[3]), d_a, n, DivFunctor<T>(), s);

    // atan2: native/cuda/BinaryGeometricKernels.cu:18
    transform(
      cuda::std::make_tuple(d_a, d_in[4]),
      d_b,
      n,
      [] __device__(T a, T b) -> T {
        return ::atan2(a, b);
      },
      s);

    // hypot: native/cuda/BinaryGeometricKernels.cu:29
    transform(
      cuda::std::make_tuple(d_b, d_in[5]),
      d_a,
      n,
      [] __device__(T a, T b) -> T {
        return ::hypot(a, b);
      },
      s);

    // xlogy: native/cuda/BinaryMiscOpsKernels.cu:46
    transform(
      cuda::std::make_tuple(d_a, d_in[6]),
      d_b,
      n,
      [] __device__(T x, T y) -> T {
        if (::isnan(static_cast<float>(y)))
        {
          return NAN;
        }
        if (x == 0)
        {
          return 0;
        }
        return x * std::log(y);
      },
      s);

    // xlog1py: native/cuda/BinaryMiscOpsKernels.cu:60
    transform(
      cuda::std::make_tuple(d_b, d_in[7]),
      d_a,
      n,
      [] __device__(T x, T y) -> T {
        if (::isnan(static_cast<float>(y)))
        {
          return NAN;
        }
        if (x == 0)
        {
          return 0;
        }
        return x * std::log1p(y);
      },
      s);

    // logaddexp: native/cuda/LogAddExpKernel.cu:253
    transform(
      cuda::std::make_tuple(d_a, d_in[8]),
      d_b,
      n,
      [] __device__(T a_, T b_) -> T {
        using opmath_t = opmath_type<T>;
        const auto a   = static_cast<opmath_t>(a_);
        const auto b   = static_cast<opmath_t>(b_);
        if (::isinf(a) && a == b)
        {
          return a;
        }
        else
        {
          const auto m = ::max(a, b);
          return m + ::log1p(::exp(-::abs(a - b)));
        }
      },
      s);

    // logaddexp2: native/cuda/LogAddExpKernel.cu:272
    transform(
      cuda::std::make_tuple(d_b, d_in[9]),
      d_a,
      n,
      [inv_log_2] __device__(T a_, T b_) -> T {
        using opmath_t = opmath_type<T>;
        const auto a   = static_cast<opmath_t>(a_);
        const auto b   = static_cast<opmath_t>(b_);
        if (::isinf(a) && a == b)
        {
          return a;
        }
        else
        {
          const auto m = ::max(a, b);
          return m + ::log1p(::exp2(-::abs(a - b))) * inv_log_2;
        }
      },
      s);

    // pow (tensor,tensor): native/cuda/PowKernel.cu:136, helper native/cuda/Pow.cuh:40
    transform(
      cuda::std::make_tuple(d_a, d_in[10]),
      d_b,
      n,
      [] __device__(T base, T exp) -> T {
        return cuda::std::pow(base, exp);
      },
      s);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// ============================================================================
// many_inputs_few_instructions
//
// mse_loss -> smooth_l1_loss -> huber_loss -> clamp_min ->
// mul -> add -> addcmul -> lerp(scalar) -> lerp(tensor) -> greater
// 13 inputs, 8 binary ops + 2 ternary ops
// ============================================================================

template <typename T>
static void many_inputs_few_instructions(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");

  constexpr int num_in = 13;
  thrust::device_vector<T> in[num_in];
  for (int i = 0; i < num_in; i++)
  {
    fill_normal(in[i], n, i);
  }
  thrust::device_vector<T> tmpA(n, thrust::no_init), tmpB(n, thrust::no_init);

  T* d_in[num_in];
  for (int i = 0; i < num_in; i++)
  {
    d_in[i] = thrust::raw_pointer_cast(in[i].data());
  }
  T* d_a = thrust::raw_pointer_cast(tmpA.data());
  T* d_b = thrust::raw_pointer_cast(tmpB.data());

  // 8 binary (16 reads) + 2 ternary (6 reads) = 22 reads, 10 writes
  state.add_element_count(n);
  state.add_global_memory_reads<T>(22L * n);
  state.add_global_memory_writes<T>(10L * n);

  // Captured scalar parameters, matching how ATen sets them up before gpu_kernel
  using opmath_t = opmath_type<T>;
  T beta_val(1.0); // smooth_l1: scalar_t beta_val(beta)
  T delta_val(1.0); // huber: scalar_t delta_val(delta)
  // note: opmath_type is same as at::acc_type<scalar_t, true> here
  using accscalar_t     = opmath_type<T>; // addcmul: at::acc_type<scalar_t, true>
  const auto alpha      = accscalar_t(1); // addcmul: value.to<accscalar_t>()
  const auto weight_val = opmath_t(4.0); // lerp scalar: weight.to<opmath_t>()

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](const nvbench::launch& launch) {
    const auto s = launch.get_stream().get_stream();

    // mse_loss: native/cuda/BinaryMiscOpsKernels.cu:37
    transform(
      cuda::std::make_tuple(d_in[0], d_in[1]),
      d_a,
      n,
      [] __device__(T a, T b) -> T {
        auto diff = a - b;
        return diff * diff;
      },
      s);

    // smooth_l1_loss(beta=1.0): native/cuda/BinaryMiscOpsKernels.cu:19
    transform(
      cuda::std::make_tuple(d_a, d_in[2]),
      d_b,
      n,
      [beta_val] __device__(T a, T b) -> T {
        auto z = ::abs(a - b);
        return z < beta_val ? T(0.5) * z * z / beta_val : z - T(0.5) * beta_val;
      },
      s);

    // huber_loss(delta=1.0): native/cuda/BinaryMiscOpsKernels.cu:29
    transform(
      cuda::std::make_tuple(d_b, d_in[3]),
      d_a,
      n,
      [delta_val] __device__(T a, T b) -> T {
        auto z = ::abs(a - b);
        return z < delta_val ? T(0.5) * z * z : delta_val * (z - T(0.5) * delta_val);
      },
      s);

    // clamp(min=tensor) -> maximum: native/cuda/MaxMinElementwiseKernel.cu:28
    transform(
      cuda::std::make_tuple(d_a, d_in[4]),
      d_b,
      n,
      [] __device__(T a, T b) -> T {
        if (a != a)
        {
          return a;
        }
        else if (b != b)
        {
          return b;
        }
        else
        {
          return ::max(a, b);
        }
      },
      s);

    // mul: native/cuda/BinaryMulKernel.cu:39, MulFunctor in native/cuda/BinaryInternal.h:27
    using mul_opmath_t = opmath_type<T>;
    transform(cuda::std::make_tuple(d_b, d_in[5]), d_a, n, MulFunctor<mul_opmath_t>(), s);

    // add(alpha=1): native/ufunc/add.h:14, torchgen/dest/ufunc.py
    transform(cuda::std::make_tuple(d_a, d_in[6]), d_b, n, CUDAFunctor_add<T>(1.0), s);

    // addcmul(value=1): native/cuda/PointwiseOpsKernel.cu:87, native/cuda/DeviceAddCmulCdiv.cuh:9
    transform(
      cuda::std::make_tuple(d_b, d_in[7], d_in[8]),
      d_a,
      n,
      [alpha] __device__(T a, T b, T c) -> T {
        return pointwise_op_impl<accscalar_t>(a, b, c, alpha, cuda::std::multiplies<accscalar_t>());
      },
      s);

    // lerp(weight=4.0): native/cuda/Lerp.cu:130, native/Lerp.h:21
    transform(
      cuda::std::make_tuple(d_a, d_in[9]),
      d_b,
      n,
      [=] __device__(T self_val, T end_val) {
        return aten_lerp(self_val, end_val, weight_val);
      },
      s);

    // lerp(weight=tensor): native/cuda/Lerp.cu:76, native/Lerp.h:21
    transform(
      cuda::std::make_tuple(d_b, d_in[10], d_in[11]),
      d_a,
      n,
      [] __device__(T self_val, T end_val, T weight_val) -> T {
        return aten_lerp(self_val, end_val, weight_val);
      },
      s);

    // note: even though output is bool, we use d_b as output because
    // it must hold at least enough memory per element for bool (1 byte)
    // greater: native/cuda/CompareKernels.cu:69
    CompareFunctor<T> comp_f(OpType::GT);
    transform(cuda::std::make_tuple(d_a, d_in[12]), d_b, n, comp_f, s);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// ============================================================================
// few_inputs_many_instructions
//
// pow(2.5) -> tanh -> sin -> cos -> softplus ->
// silu -> mish -> elu -> gelu -> logsigmoid
// 1 input, 10 unary ops
// ============================================================================

template <typename T>
static void few_inputs_many_instructions(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> input(n, thrust::no_init);
  fill_normal(input, n, 0);
  thrust::device_vector<T> tmpA(n, thrust::no_init), tmpB(n, thrust::no_init);

  T* d_in = thrust::raw_pointer_cast(input.data());
  T* d_a  = thrust::raw_pointer_cast(tmpA.data());
  T* d_b  = thrust::raw_pointer_cast(tmpB.data());

  state.add_element_count(n);
  state.add_global_memory_reads<T>(10L * n);
  state.add_global_memory_writes<T>(10L * n);

  // Captured scalar parameters
  using opmath_t        = opmath_type<T>;
  const auto exp_val    = T(2.5); // pow: exp_scalar.to<scalar_t>()
  const auto beta       = opmath_t(1); // softplus: beta_.to<opmath_t>()
  const auto threshold  = opmath_t(20); // softplus: threshold_.to<opmath_t>()
  const auto negcoef    = opmath_t(1) * opmath_t(1); // elu: alpha * scale
  const auto poscoef    = opmath_t(1); // elu: scale
  const auto negiptcoef = opmath_t(1); // elu: input_scale

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](const nvbench::launch& launch) {
    const auto s = launch.get_stream().get_stream();

    // pow(scalar=2.5): native/cuda/PowKernel.cu:163, helper native/cuda/Pow.cuh:40
    transform(
      d_in,
      d_a,
      n,
      [=] __device__(T base) -> T {
        return cuda::std::pow(base, exp_val);
      },
      s);

    // tanh: native/cuda/UnaryGeometricTanhKernel.cu:50
    transform(
      d_a,
      d_b,
      n,
      [] __device__(T a) -> T {
        return ::tanh(a);
      },
      s);

    // sin: native/cuda/UnaryGeometricSinKernel.cu:50
    transform(
      d_b,
      d_a,
      n,
      [] __device__(T a) -> T {
        return ::sin(a);
      },
      s);

    // cos: native/cuda/UnaryGeometricCosKernel.cu:50
    transform(
      d_a,
      d_b,
      n,
      [] __device__(T a) -> T {
        return ::cos(a);
      },
      s);

    // softplus(beta=1, threshold=20): native/cuda/ActivationSoftplusKernel.cu:35
    transform(
      d_b,
      d_a,
      n,
      [beta, threshold] __device__(T a) -> T {
        using opmath_t = opmath_type<T>;
        opmath_t aop   = static_cast<opmath_t>(a);
        return (aop * beta) > threshold ? aop : (::log1p(std::exp(aop * beta))) / beta;
      },
      s);

    // silu: native/cuda/ActivationSiluKernel.cu:30
    transform(
      d_a,
      d_b,
      n,
      [] __device__(T x) -> T {
        using opmath_t       = opmath_type<T>;
        const opmath_t x_acc = static_cast<opmath_t>(x);
        return x_acc / (opmath_t(1) + ::exp(-x_acc));
      },
      s);

    // mish: native/cuda/ActivationMishKernel.cu:29
    transform(
      d_b,
      d_a,
      n,
      [] __device__(T x) -> T {
        using opmath_t       = opmath_type<T>;
        const opmath_t x_acc = static_cast<opmath_t>(x);
        return x_acc * ::tanhf(::log1pf(::expf(x_acc)));
      },
      s);

    // elu(alpha=1, scale=1, input_scale=1): native/cuda/ActivationEluKernel.cu:37
    transform(
      d_a,
      d_b,
      n,
      [negcoef, poscoef, negiptcoef] __device__(T a) -> T {
        using opmath_t = opmath_type<T>;
        opmath_t aop   = static_cast<opmath_t>(a);
        return aop > 0 ? aop * poscoef : std::expm1(aop * negiptcoef) * negcoef;
      },
      s);

    // gelu(approximate='none'): native/cuda/ActivationGeluKernel.cu:35
    transform(
      d_b,
      d_a,
      n,
      [] __device__(T x) -> T {
        using opmath_t            = opmath_type<T>;
        constexpr opmath_t kAlpha = M_SQRT1_2;
        return static_cast<opmath_t>(x) * opmath_t(0.5) * (opmath_t(1) + ::erf(static_cast<opmath_t>(x) * kAlpha));
      },
      s);

    // logsigmoid: native/cuda/ActivationLogSigmoidKernel.cu:30
    transform(
      d_a,
      d_b,
      n,
      [] __device__(T in_) -> T {
        using opmath_t    = opmath_type<T>;
        const opmath_t in = in_;
        const auto min    = cuda::std::min(opmath_t(0), in);
        const auto z      = std::exp(-std::abs(in));
        return min - std::log1p(z);
      },
      s);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

// ============================================================================
// few_inputs_few_instructions
//
// add(0.5) -> neg -> clamp(-2,1) -> abs -> mul(1.5) ->
// leaky_relu -> hardswish -> hardshrink -> hardsigmoid -> gt(0)
// 1 input, 10 unary ops
// ============================================================================

template <typename T>
static void few_inputs_few_instructions(nvbench::state& state, nvbench::type_list<T>)
try
{
  const auto n = state.get_int64("Elements{io}");

  thrust::device_vector<T> input(n, thrust::no_init);
  fill_normal(input, n, 0);
  thrust::device_vector<T> tmpA(n, thrust::no_init), tmpB(n, thrust::no_init);

  T* d_in = thrust::raw_pointer_cast(input.data());
  T* d_a  = thrust::raw_pointer_cast(tmpA.data());
  T* d_b  = thrust::raw_pointer_cast(tmpB.data());

  state.add_element_count(n);
  state.add_global_memory_reads<T>(10L * n);
  state.add_global_memory_writes<T>(10L * n);

  // Captured scalar parameters
  using opmath_t = opmath_type<T>;

  // clamp: native/cuda/TensorCompare.cu:58
  const auto lim0_val = opmath_t(-2);
  const auto lim1_val = opmath_t(1);
  const auto minmax   = 2; // 0=Min, 1=Max, 2=MinMax

  // mul scalar: MulFunctor via BUnaryFunctor with captured scalar
  const auto mul_scalar = opmath_t(1.5);

  // leaky_relu: native/cuda/ActivationLeakyReluKernel.cu:31
  const auto negval = opmath_t(0.01); // negval_.to<opmath_t>()

  // hardswish: native/cuda/ActivationHardswishKernel.cu:25
  const opmath_t zero(0.0f);
  const opmath_t one_sixth(1.0f / 6.0f);
  const opmath_t three(3.0f);
  const opmath_t six(6.0f);

  // hardshrink: native/cuda/ActivationHardshrinkKernel.cu:29
  const auto lambd = T(0.5); // value.to<scalar_t>()

  // gt scalar: native/cuda/CompareKernels.cu:47
  const T rhs(0);

  state.exec(nvbench::exec_tag::gpu | nvbench::exec_tag::no_batch, [&](const nvbench::launch& launch) {
    const auto s = launch.get_stream().get_stream();

    // add(scalar, alpha=1): native/ufunc/add.h:14
    transform(d_in, d_a, n, CUDAFunctorOnSelf_add<T>(T(0.5), T(1)), s);

    // neg: native/cuda/UnarySignKernels.cu:54
    transform(
      d_a,
      d_b,
      n,
      [] __device__(T a) -> T {
        return -a;
      },
      s);

    // clamp(min=-2, max=1): native/cuda/TensorCompare.cu:58 (MinMax branch)
    transform(
      d_b,
      d_a,
      n,
      [=] __device__(T v) -> T {
        using opmath_t = opmath_type<T>;
        if (::isnan(static_cast<opmath_t>(v)))
        {
          return v;
        }
        else if (minmax == 0)
        {
          return ::max(static_cast<opmath_t>(v), lim0_val);
        }
        else if (minmax == 1)
        {
          return ::min(static_cast<opmath_t>(v), lim0_val);
        }
        else
        {
          return ::min(::max(static_cast<opmath_t>(v), lim0_val), lim1_val);
        }
      },
      s);

    // abs: native/cuda/AbsKernel.cu:39, AbsFunctor:11
    transform(d_a, d_b, n, AbsFunctor<T>(), s);

    // mul(scalar=1.5): native/cuda/BinaryMulKernel.cu:39, MulFunctor via BUnaryFunctor
    transform(
      d_b,
      d_a,
      n,
      [mul_scalar] __device__(T a) -> T {
        return MulFunctor<opmath_t>()(a, mul_scalar);
      },
      s);

    // leaky_relu(slope=0.01): native/cuda/ActivationLeakyReluKernel.cu:31
    transform(
      d_a,
      d_b,
      n,
      [negval] __device__(T a) -> T {
        using opmath_t = opmath_type<T>;
        opmath_t aop   = static_cast<opmath_t>(a);
        return aop > opmath_t(0) ? aop : aop * negval;
      },
      s);

    // hardswish: native/cuda/ActivationHardswishKernel.cu:25
    transform(
      d_b,
      d_a,
      n,
      [zero, one_sixth, three, six] __device__(T self_val) -> T {
        using opmath_t = opmath_type<T>;
        opmath_t x     = static_cast<opmath_t>(self_val);
        return x * cuda::std::min(cuda::std::max(x + three, zero), six) * one_sixth;
      },
      s);

    // hardshrink(lambd=0.5): native/cuda/ActivationHardshrinkKernel.cu:29
    transform(
      d_a,
      d_b,
      n,
      [lambd] __device__(T a) -> T {
        return (a >= -lambd && a <= lambd) ? T(0) : a;
      },
      s);

    // hardsigmoid: native/cuda/ActivationHardsigmoidKernel.cu:30
    transform(
      d_b,
      d_a,
      n,
      [zero, one_sixth, three, six] __device__(T self_val) -> T {
        using opmath_t = opmath_type<T>;
        opmath_t x     = static_cast<opmath_t>(self_val);
        return cuda::std::min<opmath_t>(cuda::std::max<opmath_t>(x + three, zero), six) * one_sixth;
      },
      s);

    // gt(scalar=0): native/cuda/CompareKernels.cu:47
    CompareFunctor<T> comp_f(OpType::GT);
    transform(
      d_a,
      d_b,
      n,
      [=] __device__(T lhs) -> T {
        return comp_f(lhs, rhs);
      },
      s);
  });
}
catch (const std::bad_alloc&)
{
  state.skip("Skipping: out of memory.");
}

NVBENCH_BENCH_TYPES(many_inputs_many_instructions, NVBENCH_TYPE_AXES(element_types))
  .set_name("many_inputs_many_instructions")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

NVBENCH_BENCH_TYPES(many_inputs_few_instructions, NVBENCH_TYPE_AXES(element_types))
  .set_name("many_inputs_few_instructions")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

NVBENCH_BENCH_TYPES(few_inputs_many_instructions, NVBENCH_TYPE_AXES(element_types))
  .set_name("few_inputs_many_instructions")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));

NVBENCH_BENCH_TYPES(few_inputs_few_instructions, NVBENCH_TYPE_AXES(element_types))
  .set_name("few_inputs_few_instructions")
  .set_type_axes_names({"T{ct}"})
  .add_int64_power_of_two_axis("Elements{io}", nvbench::range(16, 28, 4));
