//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Sparse conjugate gradient algorithm
 */

#include <cuda/experimental/stf.cuh>

#include <string>
#include <type_traits>
#include <vector>

#include "cg_solver.cuh"
#include "dot.cuh"

using namespace cuda::experimental::stf;

#if !_CCCL_CTK_BELOW(12, 4)

template <typename ctx_t, typename T>
void SPMV(ctx_t& ctx, csr_matrix<T>& a, vector_t<T>& x, vector_t<T>& y)
{
  ctx.parallel_for(y.shape(), a.val_handle.read(), a.col_handle.read(), a.row_handle.read(), x.read(), y.write())
      .set_symbol("SPMV")
      ->*[] _CCCL_DEVICE(size_t row, auto da_val, auto da_col, auto da_row, auto dx, auto dy) {
            int row_start = da_row(row);
            int row_end   = da_row(row + 1);

            double sum = 0.0;
            for (int elt = row_start; elt < row_end; elt++)
            {
              sum += da_val(elt) * dx(da_col(elt));
            }

            dy(row) = sum;
          };
}

void build_full_csr_structure(size_t* row_offsets, size_t* col_indices, size_t N)
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
              // Jacobian row: [1, 0, 0, ..., 0]
              size_t val_idx   = 0; // First entry in CSR values array
              dvalues[val_idx] = 1.0;
            }
            else if (row == N - 1)
            {
              // Right boundary: u[N-1] = 0 (homogeneous Dirichlet)
              // Jacobian row: [0, ..., 0, 1]
              size_t val_idx   = 1 + 3 * (N - 2); // Last entry in CSR values array
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

              // CSR indexing for interior row i: starts at 1 + 3*(i-1)
              size_t val_idx       = 1 + 3 * (row - 1);
              dvalues[val_idx]     = left; // ∂F_i/∂u_{i-1}
              dvalues[val_idx + 1] = center; // ∂F_i/∂u_i
              dvalues[val_idx + 2] = right; // ∂F_i/∂u_{i+1}
            }
          };
}

// residual: length N (full system including boundaries)
template <typename ctx_t, typename T>
void compute_residual_full(
  ctx_t& ctx, vector_t<T> U, vector_t<T> U_prev, vector_t<T> residual, size_t N, double h, double dt, double nu)
{
  ctx.parallel_for(box(N), residual.write(), U.read(), U_prev.read()).set_symbol("compute_residual_full")
      ->*[N, h, dt, nu] __device__(size_t i, auto dresidual, auto dU, auto dU_prev) {
            if (i == 0)
            {
              // Left boundary condition: u[0] = 0
              dresidual(i) = dU(i) - 0.0;
            }
            else if (i == N - 1)
            {
              // Right boundary condition: u[N-1] = 0
              dresidual(i) = dU(i) - 0.0;
            }
            else
            {
              // Interior point: Burger's equation F_i = ∂u/∂t + u*∂u/∂x - nu*∂²u/∂x²
              double u_i   = dU(i);
              double u_ip1 = dU(i + 1);
              double u_im1 = dU(i - 1);

              double term_time = (u_i - dU_prev(i)) / dt; // ∂u/∂t
              double term_conv = u_i * (u_ip1 - u_im1) / (2 * h); // u * ∂u/∂x (nonlinear convection)
              double term_diff = -nu * (u_im1 - 2 * u_i + u_ip1) / (h * h); // -nu * ∂²u/∂x²

              dresidual(i) = term_time + term_conv + term_diff;
            }
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

/**
 * Generic Newton solver for nonlinear systems F(x) = 0
 *
 * @tparam ctx_t STF context type
 * @tparam ResidualCallback Callback to compute residual F(x)
 * @tparam JacobianCallback Callback to assemble Jacobian J = ∂F/∂x
 *
 * The callbacks must be callable with these signatures:
 * - ResidualCallback: void fn(ctx_t&, const vector_t<double>& x, const vector_t<double>& x_prev, vector_t<double>&
 * residual)
 * - JacobianCallback: void fn(ctx_t&, const vector_t<double>& x, vector_t<double>& jacobian_values)
 */
template <typename ctx_t, typename ResidualCallback, typename JacobianCallback>
void newton_solver(
  ctx_t& ctx,
  vector_t<double>& U,
  vector_t<double>& csr_values,
  const vector_t<size_t>& csr_row_offsets,
  const vector_t<size_t>& csr_col_ind,
  ResidualCallback compute_residual_fn,
  JacobianCallback assemble_jacobian_fn,
  size_t max_newton = 20,
  double newton_tol = 1e-10,
  size_t max_cg     = 100)
{
  auto U_prev = ctx.logical_data(U.shape()).set_symbol("U_prev");

  ctx.parallel_for(U.shape(), U_prev.write(), U.read()).set_symbol("init_guess")
      ->*[] __device__(size_t i, auto dU_prev, auto dU) {
            dU_prev(i) = dU(i);
          };

  auto newton_norm2 = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("newton_norm2");
  auto newton_iter  = ctx.logical_data(shape_of<scalar_view<size_t>>()).set_symbol("newton_iter");
  ctx.parallel_for(box(1), newton_iter.write()).set_symbol("init_newton_iter")->*[] _CCCL_DEVICE(size_t i, auto diter) {
    *diter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    auto residual = ctx.logical_data(U.shape()).set_symbol("residual");
    auto delta    = ctx.logical_data(U.shape()).set_symbol("delta");

    // Compute residual F(U)
    compute_residual_fn(ctx, U, U_prev, residual);

    // Compute Newton residual norm for convergence check
    DOT(ctx, residual, residual, newton_norm2);

    // Assemble Jacobian J = ∂F/∂U
    assemble_jacobian_fn(ctx, U, csr_values);

    auto rhs = ctx.logical_data(U.shape()).set_symbol("rhs");

    // Set up RHS: rhs = -F(U)
    ctx.parallel_for(rhs.shape(), rhs.write(), residual.read()).set_symbol("rhs = -residual")
        ->*[] __device__(size_t i, auto drhs, auto dresidual) {
              drhs(i) = -dresidual(i);
            };

    csr_matrix<double> A(csr_values, csr_row_offsets, csr_col_ind);

    // Solve linear system: J * delta = -F(U)
    double cg_tol = 1e-8;
    cg_solver(ctx, A, delta, rhs, cg_tol, max_cg);

    // Newton update: U = U + delta (no special boundary handling needed)
    ctx.parallel_for(U.shape(), U.rw(), delta.read()).set_symbol("newton_update")
        ->*[] __device__(size_t i, auto dU, auto ddelta) {
              dU(i) += ddelta(i);
            };

    while_guard.update_cond(newton_norm2.read(), newton_iter.rw())
        ->*[newton_tol, max_newton] __device__(auto dnorm2, auto diter) {
              (*diter)++; // increment iteration counter
              bool converged = (*dnorm2 < newton_tol * newton_tol);
              return !converged && (*diter < max_newton);
            };
  }
}

// Initialize the solution output file (call once at simulation start)
void initialize_solution_file(const char* filename, size_t N, double h)
{
  FILE* fp = fopen(filename, "w");
  if (fp)
  {
    fprintf(fp, "# Burger equation solution - block format\n");
    fprintf(fp, "# Each timestep is a separate block, separated by blank lines\n");
    fprintf(fp, "# Format: x_coordinate  u(x,t)\n");
    fprintf(fp, "# Grid points: %zu, h=%.6e\n", N, h);
    fprintf(fp,
            "# Use in gnuplot: plot for [i=0:*] 'solution.dat' index i with lines title sprintf('step %%d', i*10)\n");
    fprintf(fp, "#\n");
    fclose(fp);
    printf("Initialized solution file: %s (block format)\n", filename);
  }
  else
  {
    printf("Error: Could not create %s for writing\n", filename);
  }
}

// Function to append timestep block to solution file (simple and reliable)
template <typename ctx_t>
void dump_solution(
  ctx_t& ctx, vector_t<double>& U, size_t timestep, size_t N, double h, double dt, const char* filename = "solution.dat")
{
  ctx.host_launch(U.read()).set_symbol("dump solution")->*[timestep, h, N, dt, filename](auto hU) {
    FILE* fp = fopen(filename, "a"); // Simple append - no read/modify/write
    if (fp)
    {
      fprintf(fp, "# Timestep %zu, t=%.6e\n", timestep, timestep * dt);

      for (size_t i = 0; i < N; i++)
      {
        double x = i * h;
        fprintf(fp, "%.10e %.10e\n", x, hU(i));
      }

      fprintf(fp, "\n"); // Blank line to separate datasets
      fclose(fp);

      printf("Appended timestep %zu (t=%.4e) to %s\n", timestep, timestep * dt, filename);
    }
    else
    {
      printf("Error: Could not open %s for appending\n", filename);
    }
  };
}
#endif

int main([[maybe_unused]] int argc, [[maybe_unused]] char** argv)
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving test: conditional nodes are only available since CUDA 12.4.\n");
  return 0;
#else
  // Usage: ./burger [N] [nsteps] [nu]
  // N      = Grid points (default: 100000)
  // nsteps = Time steps (default: 10000)
  // nu     = Viscosity (default: 0.05, try 0.001 for shocks)

  size_t N = 100000; // Large system - auto-scaled parameters maintain stability

  stackable_ctx ctx;

  if (argc > 1)
  {
    N = atoi(argv[1]);
    fprintf(stderr, "N = %zu\n", N);
  }

  double h = 1.0 / (N - 1);

  // Set reasonable parameters - implicit method allows larger time steps
  double nu = 0.05; // Default viscosity
  if (argc > 3)
  {
    nu = atof(argv[3]);
    fprintf(stderr, "nu = %e\n", nu);
  }

  double dt_diffusion = 0.5 * h * h / nu; // Diffusion-limited time step
  double dt_fixed     = 0.001; // Fixed reasonable time step
  double dt           = std::max(dt_diffusion, dt_fixed); // Use larger of the two

  // For very fine grids, cap the time step to prevent tiny steps
  if (N > 10000)
  {
    dt = std::min(dt, 0.01); // Cap at 0.01 for large grids
  }

  size_t nsteps = 10000;
  if (argc > 2)
  {
    nsteps = atol(argv[2]);
    fprintf(stderr, "nsteps = %ld\n", nsteps);
  }

  double total_time = nsteps * dt;

  fprintf(stderr, "=== Simulation Parameters ===\n");
  fprintf(stderr, "Grid: N=%zu, h=%e\n", N, h);
  fprintf(stderr, "Time: dt=%e, nsteps=%zu, total_time=%e\n", dt, nsteps, total_time);
  fprintf(stderr, "Physics: nu=%e (viscosity)\n", nu);
  fprintf(stderr, "Diffusion number: nu*dt/h^2 = %e\n", nu * dt / (h * h));
  fprintf(stderr, "System size: %zu unknowns, %zu non-zeros (full N×N system with BCs)\n", N, 3 * N - 4);
  if (N > 1000000)
  {
    fprintf(stderr, "WARNING: Very large system! Consider using iterative preconditioners for N > 1M\n");
  }
  fprintf(stderr, "=============================\n");

  // Full N×N system: boundary rows have 1 entry each, interior rows have 3 entries each
  // Total: 2*1 + (N-2)*3 = 3*N - 4 non-zeros
  size_t nz = 3 * N - 4;

  size_t* row_offsets;
  size_t* col_indices;
  cuda_safe_call(cudaHostAlloc(&row_offsets, (N + 1) * sizeof(size_t), cudaHostAllocMapped));
  cuda_safe_call(cudaHostAlloc(&col_indices, nz * sizeof(size_t), cudaHostAllocMapped));

  build_full_csr_structure(row_offsets, col_indices, N);

  auto csr_row_offsets = ctx.logical_data(make_slice(row_offsets, N + 1)).set_symbol("csr_row");
  auto csr_col_ind     = ctx.logical_data(make_slice(col_indices, nz)).set_symbol("csr_col");
  auto csr_values      = ctx.logical_data(shape_of<slice<double>>(nz)).set_symbol("csr_val");

  auto U = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("U");

  // This will prevent erroneous modifications and may allow access from concurrent graphs
  csr_row_offsets.set_read_only();
  csr_col_ind.set_read_only();

  // Initial condition
  ctx.parallel_for(U.shape(), U.write()).set_symbol("init conditions")->*[h, N] __device__(size_t i, auto dU) {
    double x = i * h;
    if (i == 0 || i == N - 1)
    {
      dU(i) = 0.0; // Homogeneous Dirichlet boundary conditions
    }
    else
    {
      dU(i) = sin(M_PI * x);
    }
  };

  // Initialize solution output file
  initialize_solution_file("solution.dat", N, h);

  // Parameters are now set above with auto-scaling
  size_t substeps         = 100;
  size_t outer_iterations = nsteps / substeps;

  for (size_t outer = 0; outer < outer_iterations; outer++)
  {
    auto g = ctx.graph_scope();

    // Repeat substeps inner iterations using STF repeat block
    {
      auto repeat_guard = ctx.repeat_graph_scope(substeps);

      // Create callback function objects for Burger's equation
      BurgerResidualCallback residual_callback{N, h, dt, nu};
      BurgerJacobianCallback jacobian_callback{N, h, dt, nu};

      // Solve the nonlinear system using generic Newton solver
      newton_solver(ctx, U, csr_values, csr_row_offsets, csr_col_ind, residual_callback, jacobian_callback);
    } // repeat_guard automatically manages the loop condition

    // Dump solution after each substep block
    size_t current_timestep = (outer + 1) * substeps;
    dump_solution(ctx, U, current_timestep, N, h, dt);
  }

  // Final solution information
  printf("\n=== Simulation complete ===\n");
  printf("Solution file: solution.dat (block format)\n");
  printf("  - Each timestep is a separate data block\n");
  printf("  - Blocks separated by blank lines\n");
  printf("  - Format: x_coordinate  u(x,t)\n");
  printf("Now you can run the animation commands shown at the beginning!\n");

  ctx.finalize();
#endif
}
