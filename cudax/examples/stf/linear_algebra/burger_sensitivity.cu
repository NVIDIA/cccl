//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Sensitivity analysis for Burger equation - analyze shock formation vs viscosity
 */

#include <cuda/experimental/stf.cuh>

#include <algorithm>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "cg_solver.cuh"
#include "dot.cuh"
#include "newton_solver.cuh"

using namespace cuda::experimental::stf;

#if !_CCCL_CTK_BELOW(12, 4)

void build_tridiagonal_csr_structure(size_t* row_offsets, size_t* col_indices, size_t N)
{
  size_t nnz     = 0;
  row_offsets[0] = 0;

  for (size_t row = 0; row < N; row++)
  {
    if (row == 0 || row == N - 1)
    {
      // Boundary rows: only diagonal entry (identity for BC: u[i] = prescribed_value)
      col_indices[nnz++] = row;
    }
    else
    {
      // Interior rows: tridiagonal structure (left, center, right)
      col_indices[nnz++] = row - 1; // left neighbor
      col_indices[nnz++] = row; // center (diagonal)
      col_indices[nnz++] = row + 1; // right neighbor
    }
    row_offsets[row + 1] = nnz;
  }
}

template <typename ctx_t>
void assemble_jacobian_full(
  ctx_t& ctx, vector_t<double> U, vector_t<double> values, size_t N, double h, double dt, double nu)
{
  ctx.parallel_for(box(N), U.read(), values.write()).set_symbol("assemble_jacobian_full")
      ->*[N, h, dt, nu] __device__(size_t row, auto dU, auto dvalues) {
            if (row == 0)
            {
              // Left boundary: u[0] = 0 (homogeneous Dirichlet)
              size_t val_idx   = 0;
              dvalues[val_idx] = 1.0;
            }
            else if (row == N - 1)
            {
              // Right boundary: u[N-1] = 0 (homogeneous Dirichlet)
              size_t val_idx   = 1 + 3 * (N - 2);
              dvalues[val_idx] = 1.0;
            }
            else
            {
              // Interior point: Burger's equation discretization
              double u_i   = dU[row];
              double u_ip1 = dU[row + 1];
              double u_im1 = dU[row - 1];

              // Jacobian entries: ∂F_i/∂u_{i-1}, ∂F_i/∂u_i, ∂F_i/∂u_{i+1}
              double left   = -u_i / (2 * h) - nu / (h * h);
              double center = 1.0 / dt + (u_ip1 - u_im1) / (2 * h) + 2.0 * nu / (h * h);
              double right  = u_i / (2 * h) - nu / (h * h);

              size_t val_idx       = 1 + 3 * (row - 1);
              dvalues[val_idx]     = left;
              dvalues[val_idx + 1] = center;
              dvalues[val_idx + 2] = right;
            }
          };
}

template <typename ctx_t, typename T>
void compute_residual_full(
  ctx_t& ctx, vector_t<T> U, vector_t<T> U_prev, vector_t<T> residual, size_t N, double h, double dt, double nu)
{
  ctx.parallel_for(box(N), residual.write(), U.read(), U_prev.read()).set_symbol("compute_residual_full")
      ->*[N, h, dt, nu] __device__(size_t i, auto dresidual, auto dU, auto dU_prev) {
            if (i == 0)
            {
              dresidual(i) = dU(i) - 0.0;
            }
            else if (i == N - 1)
            {
              dresidual(i) = dU(i) - 0.0;
            }
            else
            {
              // Interior point: Burger's equation F_i = ∂u/∂t + u*∂u/∂x - nu*∂²u/∂x²
              double u_i   = dU(i);
              double u_ip1 = dU(i + 1);
              double u_im1 = dU(i - 1);

              double term_time = (u_i - dU_prev(i)) / dt;
              double term_conv = u_i * (u_ip1 - u_im1) / (2 * h);
              double term_diff = -nu * (u_im1 - 2 * u_i + u_ip1) / (h * h);

              dresidual(i) = term_time + term_conv + term_diff;
            }
          };
}

// Shock detection: compute maximum gradient magnitude
template <typename ctx_t>
void detect_shock(
  ctx_t& ctx, vector_t<double>& U, stackable_logical_data<scalar_view<double>>& max_gradient, size_t N, double h)
{
  ctx.parallel_for(box(N - 1), U.read(), max_gradient.reduce(reducer::maxval<double>{})).set_symbol("detect_shock")
      ->*[h] __device__(size_t i, auto dU, double& dmax_grad) {
            double gradient = fabs(dU(i + 1) - dU(i)) / h;
            dmax_grad       = fmax(dmax_grad, gradient);
          };
}

// Callback function objects for Burger's equation
struct BurgerResidualCallback
{
  size_t N;
  double h, dt, nu;

  template <typename ctx_t>
  void
  operator()(ctx_t& ctx, const vector_t<double>& x, const vector_t<double>& x_prev, vector_t<double>& residual) const
  {
    compute_residual_full(ctx, x, x_prev, residual, N, h, dt, nu);
  }
};

struct BurgerJacobianCallback
{
  size_t N;
  double h, dt, nu;

  template <typename ctx_t>
  void operator()(ctx_t& ctx, const vector_t<double>& x, vector_t<double>& jacobian_values) const
  {
    assemble_jacobian_full(ctx, x, jacobian_values, N, h, dt, nu);
  }
};

// Generate nu values around target with given distribution
std::vector<double> generate_nu_samples(double nu_target, double nu_std, size_t num_samples)
{
  std::vector<double> nu_values;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> dist(nu_target, nu_std);

  // Generate samples and ensure they are positive
  for (size_t i = 0; i < num_samples; ++i)
  {
    double nu_sample = dist(gen);
    // Ensure nu > 0 for physical validity
    if (nu_sample > 1e-6)
    {
      nu_values.push_back(nu_sample);
    }
    else
    {
      // Retry if we get non-physical values
      i--;
    }
  }

  // Sort for better output organization
  std::sort(nu_values.begin(), nu_values.end());
  return nu_values;
}

// Initialize sensitivity analysis output file
void initialize_sensitivity_file(const char* filename, double nu_target, double nu_std, size_t num_samples)
{
  FILE* fp = fopen(filename, "w");
  if (fp)
  {
    fprintf(fp, "# Burger equation sensitivity analysis\n");
    fprintf(fp, "# Target nu: %.6e, std: %.6e, samples: %zu\n", nu_target, nu_std, num_samples);
    fprintf(fp, "# Format: nu_value  shock_time  max_gradient  final_time\n");
    fprintf(fp, "# shock_time: time when max gradient exceeds threshold (or -1 if no shock)\n");
    fprintf(fp, "# max_gradient: maximum gradient achieved\n");
    fprintf(fp, "# final_time: total simulation time reached\n");
    fprintf(fp, "#\n");
    fclose(fp);
  }
}

// Initialize shock solutions output file
void initialize_shock_file(const char* filename, double shock_threshold)
{
  FILE* fp = fopen(filename, "w");
  if (fp)
  {
    fprintf(fp, "# Burger equation shock solutions\n");
    fprintf(fp, "# Solutions dumped when gradient exceeds threshold: %.1f\n", shock_threshold);
    fprintf(fp, "# Each shock is a separate data block, separated by blank lines\n");
    fprintf(fp, "# Block header: Sample ID, nu value, shock time, max gradient\n");
    fprintf(fp, "# Block format: x_coordinate  u(x,t_shock)\n");
    fprintf(fp, "#\n");
    fprintf(fp,
            "# Use in gnuplot: plot for [i=0:*] 'shock_solutions.dat' index i with lines title sprintf('Sample %%d', "
            "i+1)\n");
    fprintf(fp, "#\n");
    fclose(fp);
    printf("Initialized shock solutions file: %s\n", filename);
  }
}

// Dump solution when shock is detected
template <typename ctx_t>
void dump_shock_solution(
  ctx_t& ctx,
  vector_t<double>& U,
  double nu,
  double shock_time,
  double max_gradient,
  size_t sample_id,
  size_t N,
  double h,
  const char* filename = "shock_solutions.dat")
{
  ctx.host_launch(U.read()).set_symbol("dump shock solution")
      ->*
    [nu, shock_time, max_gradient, sample_id, h, N, filename](auto hU) {
      FILE* fp = fopen(filename, "a"); // Append to file
      if (fp)
      {
        fprintf(
          fp, "# Sample %zu: nu=%.6e, shock_time=%.6e, max_gradient=%.2f\n", sample_id, nu, shock_time, max_gradient);
        fprintf(fp, "# Format: x_coordinate  u(x,t_shock)\n");

        for (size_t i = 0; i < N; i++)
        {
          double x = i * h;
          fprintf(fp, "%.10e %.10e\n", x, hU(i));
        }

        fprintf(fp, "\n"); // Blank line to separate datasets
        fclose(fp);

        printf(" -> Solution saved to %s", filename);
      }
      else
      {
        printf(" -> Error: Could not save solution to %s", filename);
      }
    };
}

template <typename ctx_t>
void run_single_nu_simulation(
  ctx_t& ctx,
  double nu,
  vector_t<double>& U,
  vector_t<double>& csr_values,
  const vector_t<size_t>& csr_row_offsets,
  const vector_t<size_t>& csr_col_ind,
  size_t N,
  double h,
  double dt,
  double max_time,
  double shock_threshold,
  size_t sample_id,
  double& shock_time,
  double& max_gradient,
  double& final_time)
{
  // Reset solution to initial condition
  ctx.parallel_for(U.shape(), U.write()).set_symbol("reset_initial_condition")->*[h, N] __device__(size_t i, auto dU) {
    double x = i * h;
    dU(i)    = (i == 0 || i == N - 1) ? 0.0 : sin(M_PI * x);
  };

  auto current_time    = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("current_time");
  auto max_grad_global = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("max_grad_global");
  auto shock_detected  = ctx.logical_data(shape_of<scalar_view<int>>()).set_symbol("shock_detected");

  // Initialize tracking variables
  ctx.parallel_for(box(1), current_time.write(), max_grad_global.write(), shock_detected.write())
      .set_symbol("init_tracking")
      ->*[] __device__(size_t i, auto dtime, auto dmax_grad, auto dshock) {
            *dtime     = 0.0;
            *dmax_grad = 0.0;
            *dshock    = 0; // 0 = no shock, 1 = shock detected
          };

  // Time evolution loop with shock detection
  {
    auto while_guard = ctx.while_graph_scope();

    // Create callback function objects
    BurgerResidualCallback residual_callback{N, h, dt, nu};
    BurgerJacobianCallback jacobian_callback{N, h, dt, nu};

    // Solve the nonlinear system
    newton_solver(ctx, U, csr_values, csr_row_offsets, csr_col_ind, residual_callback, jacobian_callback);

    // Update time
    ctx.parallel_for(box(1), current_time.rw()).set_symbol("update_time")->*[dt] __device__(size_t i, auto dtime) {
      *dtime += dt;
    };

    // Detect shock by computing maximum gradient
    auto current_grad = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("current_grad");
    detect_shock(ctx, U, current_grad, N, h);

    // Update global maximum gradient and check for shock
    ctx.parallel_for(box(1), max_grad_global.rw(), current_grad.read(), shock_detected.rw())
        .set_symbol("update_shock_detection")
        ->*[shock_threshold] __device__(size_t i, auto dmax_grad, auto dcurrent_grad, auto dshock) {
              double grad = *dcurrent_grad;
              if (grad > *dmax_grad)
              {
                *dmax_grad = grad;
              }
              if (grad > shock_threshold && *dshock == 0)
              {
                *dshock = 1; // First time shock threshold is exceeded
              }
            };

    // Continue while time < max_time and no shock detected
    while_guard.update_cond(current_time.read(), shock_detected.read())->*[max_time] __device__(auto dtime, auto dshock) {
      return (*dtime < max_time) && (*dshock == 0);
    };
  }

  // Extract results to host variables
  ctx.host_launch(current_time.read(), max_grad_global.read(), shock_detected.read()).set_symbol("extract_results")
      ->*[&shock_time, &max_gradient, &final_time, shock_threshold](auto htime, auto hmax_grad, auto hshock) {
            final_time   = *htime;
            max_gradient = *hmax_grad;
            shock_time   = (*hshock == 1) ? *htime : -1.0; // -1 indicates no shock

            if (shock_time > 0)
            {
              printf("shock at t=%.4f, max_grad=%.1f\n", shock_time, max_gradient);
            }
            else
            {
              printf("no shock, max_grad=%.1f\n", max_gradient);
            }
          };

  // Dump solution if shock was detected
  if (shock_time > 0)
  {
    dump_shock_solution(ctx, U, nu, shock_time, max_gradient, sample_id, N, h);
  }
}

#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  // Usage: ./burger_sensitivity [N] [nu_target] [nu_std] [num_samples] [shock_threshold]

  size_t N           = 1000; // Smaller grid for sensitivity analysis
  double nu_target   = 0.02; // Target viscosity
  double nu_std      = 0.01; // Standard deviation for nu distribution
  size_t num_samples = 20; // Number of nu samples to test

  double shock_threshold = 15.0; // Gradient threshold to detect shock (du/dx magnitude)

  if (argc > 1)
  {
    N = atoi(argv[1]);
  }
  if (argc > 2)
  {
    nu_target = atof(argv[2]);
  }
  if (argc > 3)
  {
    nu_std = atof(argv[3]);
  }
  if (argc > 4)
  {
    num_samples = atoi(argv[4]);
  }
  if (argc > 5)
  {
    shock_threshold = atof(argv[5]);
  }

  double h        = 1.0 / (N - 1);
  double dt       = 0.001; // Fixed time step
  double max_time = 2.0; // Maximum simulation time per sample

  fprintf(stderr, "=== Sensitivity Analysis Parameters ===\n");
  fprintf(stderr, "Grid: N=%zu, h=%e\n", N, h);
  fprintf(stderr, "Viscosity: target=%e, std=%e, samples=%zu\n", nu_target, nu_std, num_samples);
  fprintf(stderr, "Time: dt=%e, max_time=%e\n", dt, max_time);
  fprintf(stderr, "Shock threshold: %.1f (gradient magnitude)\n", shock_threshold);
  fprintf(stderr, "======================================\n");

  stackable_ctx ctx;

  // Generate nu samples
  auto nu_values = generate_nu_samples(nu_target, nu_std, num_samples);

  // Set up CSR structure
  size_t nz = 3 * N - 4;
  size_t* row_offsets;
  size_t* col_indices;
  cuda_safe_call(cudaHostAlloc(&row_offsets, (N + 1) * sizeof(size_t), cudaHostAllocMapped));
  cuda_safe_call(cudaHostAlloc(&col_indices, nz * sizeof(size_t), cudaHostAllocMapped));
  build_tridiagonal_csr_structure(row_offsets, col_indices, N);

  auto csr_row_offsets = ctx.logical_data(make_slice(row_offsets, N + 1)).set_symbol("csr_row");
  auto csr_col_ind     = ctx.logical_data(make_slice(col_indices, nz)).set_symbol("csr_col");
  csr_row_offsets.set_read_only();
  csr_col_ind.set_read_only();

  // Initialize output files
  initialize_sensitivity_file("sensitivity_results.dat", nu_target, nu_std, num_samples);
  initialize_shock_file("shock_solutions.dat", shock_threshold);

  // Run sensitivity analysis
  printf("Running sensitivity analysis with %zu samples...\n", num_samples);

  {
    auto g = ctx.graph_scope();
    for (size_t i = 0; i < nu_values.size(); ++i)
    {
      double nu = nu_values[i];
      double shock_time, max_gradient, final_time;

      printf("Sample %zu/%zu: nu=%.6e... ", i + 1, nu_values.size(), nu);
      fflush(stdout);

      auto csr_values = ctx.logical_data(shape_of<slice<double>>(nz)).set_symbol("csr_val");
      auto U          = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("U");

      run_single_nu_simulation(
        ctx,
        nu,
        U,
        csr_values,
        csr_row_offsets,
        csr_col_ind,
        N,
        h,
        dt,
        max_time,
        shock_threshold,
        i + 1,
        shock_time,
        max_gradient,
        final_time);
    }
  }

  ctx.finalize();
#endif
}
