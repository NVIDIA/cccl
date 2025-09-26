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
#include <vector>

using namespace cuda::experimental::stf;

#if !_CCCL_CTK_BELOW(12, 4)
using vector_t  = stackable_logical_data<slice<double>>;
using scalar_t  = stackable_logical_data<scalar_view<double>>;
using context_t = stackable_ctx;

struct csr_matrix
{
  csr_matrix(stackable_logical_data<slice<double>> _val_handle,
             stackable_logical_data<slice<size_t>> _row_handle,
             stackable_logical_data<slice<size_t>> _col_handle)
      : val_handle(mv(_val_handle))
      , row_handle(mv(_row_handle))
      , col_handle(mv(_col_handle))
  {}

  /* Description of the CSR */
  mutable stackable_logical_data<slice<double>> val_handle;
  mutable stackable_logical_data<slice<size_t>> row_handle;
  mutable stackable_logical_data<slice<size_t>> col_handle;
};

// Note that a and b might be the same logical data
void DOT(context_t& ctx, vector_t& a, vector_t& b, scalar_t& res)
{
  ctx.parallel_for(a.shape(), a.read(), b.read(), res.reduce(reducer::sum<double>{})).set_symbol("DOT")->*
    [] __device__(size_t i, auto da, auto db, double& dres) {
      dres += da(i) * db(i);
    };
};

void SPMV(context_t& ctx, csr_matrix& a, vector_t& x, vector_t& y)
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

void build_tridiag_csr_structure(size_t* row_offsets, size_t* col_indices, size_t N)
{
  size_t n_unknowns = N - 2;
  size_t nnz        = 0;
  row_offsets[0]    = 0;

  for (size_t row = 0; row < n_unknowns; row++)
  {
    // For interior point i (global index = row + 1), the matrix equation involves:
    // - Left neighbor: matrix column = row-1 (if row > 0)
    // - Center: matrix column = row
    // - Right neighbor: matrix column = row+1 (if row < n_unknowns-1)

    if (row > 0)
    {
      col_indices[nnz++] = row - 1; // left neighbor
    }
    col_indices[nnz++] = row; // center
    if (row < n_unknowns - 1)
    {
      col_indices[nnz++] = row + 1; // right neighbor
    }
    row_offsets[row + 1] = nnz;
  }
}

void assemble_jacobian(context_t& ctx, vector_t U, vector_t values, size_t N, double h, double dt, double nu)
{
  size_t n_unknowns = N - 2;
  ctx.parallel_for(box(n_unknowns), U.read(), values.write()).set_symbol("assemble_jacobian")
      ->*[n_unknowns, h, dt, nu] __device__(size_t row, auto dU, auto dvalues) {
            size_t global = row + 1; // global grid index for this interior point
            double u_i    = dU[global];
            double u_ip1  = dU[global + 1];
            double u_im1  = dU[global - 1];

            double left   = -u_i / (2 * h) - nu / (h * h);
            double center = 1.0 / dt + (u_ip1 - u_im1) / (2 * h) + 2.0 * nu / (h * h);
            double right  = u_i / (2 * h) - nu / (h * h);

            // Calculate the starting index for this row's values in the CSR values array
            size_t val_idx = 0;
            if (row == 0)
            {
              // First row: starts at index 0, has center + right
              val_idx              = 0;
              dvalues[val_idx]     = center;
              dvalues[val_idx + 1] = right;
            }
            else if (row == n_unknowns - 1)
            {
              // Last row: starts at index 2 + 3*(n_unknowns-2), has left + center
              val_idx              = 2 + 3 * (n_unknowns - 2);
              dvalues[val_idx]     = left;
              dvalues[val_idx + 1] = center;
            }
            else
            {
              // Middle rows: start at index 2 + 3*(row-1), have left + center + right
              val_idx              = 2 + 3 * (row - 1);
              dvalues[val_idx]     = left;
              dvalues[val_idx + 1] = center;
              dvalues[val_idx + 2] = right;
            }
          };
}

// residual: length N-2
void compute_residual(
  context_t& ctx, vector_t U, vector_t U_prev, vector_t residual, size_t /* N */, double h, double dt, double nu)
{
  ctx.parallel_for(residual.shape(), residual.write(), U.read(), U_prev.read()).set_symbol("compute_residual")
      ->*[h, dt, nu] __device__(size_t i, auto dresidual, auto dU, auto dU_prev) {
            size_t global = i + 1;
            double u_i    = dU[global];
            double u_ip1  = dU[global + 1];
            double u_im1  = dU[global - 1];

            double term_time = (u_i - dU_prev[global]) / dt;
            double term_conv = u_i * (u_ip1 - u_im1) / (2 * h);
            double term_diff = -nu * (u_im1 - 2 * u_i + u_ip1) / (h * h);

            dresidual(i) = term_time + term_conv + term_diff;
          };
}

void cg_solver(context_t& ctx, csr_matrix& A, vector_t& X, vector_t& B, double cg_tol = 1e-10, size_t max_cg = 1000)
{
  // Initial guess X = 0 (better for Newton corrections)
  ctx.parallel_for(X.shape(), X.write()).set_symbol("init_guess")->*[] _CCCL_DEVICE(size_t i, auto dX) {
    dX(i) = 0.0;
  };

  // Residual R initialized to B
  auto R = ctx.logical_data(B.shape()).set_symbol("R");
  ctx.parallel_for(R.shape(), R.write(), B.read()).set_symbol("R=B")->*[] _CCCL_DEVICE(size_t i, auto dR, auto dB) {
    dR(i) = dB(i);
  };

  // R = R - A*X
  auto Ax = ctx.logical_data(X.shape()).set_symbol("Ax");
  SPMV(ctx, A, X, Ax);
  ctx.parallel_for(R.shape(), R.rw(), Ax.read()).set_symbol("R -= Ax")->*[] _CCCL_DEVICE(size_t i, auto dR, auto dAx) {
    dR(i) -= dAx(i);
  };

  // P = R;
  auto P = ctx.logical_data(R.shape()).set_symbol("P");
  ctx.parallel_for(P.shape(), P.write(), R.read()).set_symbol("P=R")->*[] _CCCL_DEVICE(size_t i, auto dP, auto dR) {
    dP(i) = dR(i);
  };

  // RSOLD = R'*R
  auto rsold = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("rsold");
  DOT(ctx, R, R, rsold);

  // CG iteration counter
  auto cg_iter = ctx.logical_data(shape_of<scalar_view<int>>()).set_symbol("cg_iter");
  ctx.parallel_for(box(1), cg_iter.write()).set_symbol("init_cg_iter")->*[] _CCCL_DEVICE(size_t i, auto diter) {
    *diter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    // Ap = A*P
    auto Ap = ctx.logical_data(P.shape()).set_symbol("Ap");
    SPMV(ctx, A, P, Ap);

    // We don't compute alpha explicitly
    // alpha = rsold / (p' * Ap);
    auto pAp = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("pAp");
    DOT(ctx, P, Ap, pAp);

    // x = x + alpha * p;
    ctx.parallel_for(X.shape(), X.rw(), rsold.read(), pAp.read(), P.read()).set_symbol("X+=alpha*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dX, auto drsold, auto dpAp, auto dP) {
              double alpha = (*drsold / *dpAp);
              dX(i) += alpha * dP(i);
            };

    // r = r - alpha * Ap;
    ctx.parallel_for(R.shape(), R.rw(), rsold.read(), pAp.read(), Ap.read()).set_symbol("R-=alpha*Ap")
        ->*[] _CCCL_DEVICE(size_t i, auto dR, auto drsold, auto dpAp, auto dAp) {
              double alpha = (*drsold / *dpAp);
              dR(i) -= alpha * dAp(i);
            };

    // rsnew = r' * r;
    auto rsnew = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("rsnew");
    DOT(ctx, R, R, rsnew);

    while_guard.update_cond(rsnew.read(), cg_iter.rw())->*[cg_tol, max_cg] __device__(auto drsnew, auto diter) {
      (*diter)++; // increment iteration counter
      bool converged = (*drsnew < cg_tol * cg_tol);
      // printf("CG iter %d: RES %e (tol=%e)\n", *diter, sqrt(*drsnew), cg_tol);
      return !converged && (*diter < max_cg);
    };

    // p = r + (rsnew / rsold) * p;
    ctx.parallel_for(P.shape(), P.rw(), R.read(), rsnew.read(), rsold.read()).set_symbol("P=r+(rsnew/rsold)*P")
        ->*[] _CCCL_DEVICE(size_t i, auto dP, auto dR, auto drsnew, auto drsold) {
              dP(i) = dR(i) + (*drsnew / *drsold) * dP(i);
            };

    // update old residual
    ctx.parallel_for(box(1), rsold.write(), rsnew.read()).set_symbol("update_rsold")
        ->*[] _CCCL_DEVICE(size_t i, auto drsold, auto drsnew) {
              *drsold = *drsnew;
            };
  }

#  if 0
  fprintf(stderr,
          "CG solver converged after %d iterations (final residual %e, tolerance %e)\n",
          ctx.wait(cg_iter),
          std::sqrt(ctx.wait(rsold)),
          cg_tol);
#  endif
}

// Newton solver for the nonlinear system arising from implicit time discretization
void newton_solver(
  context_t& ctx,
  vector_t& U,
  vector_t& U_prev,
  vector_t& residual,
  vector_t& rhs,
  vector_t& delta,
  stackable_logical_data<slice<double>>& csr_values,
  stackable_logical_data<slice<size_t>>& csr_row_offsets,
  stackable_logical_data<slice<size_t>>& csr_col_ind,
  size_t N,
  double h,
  double dt,
  double nu,
  size_t max_newton = 20,
  double newton_tol = 1e-10,
  size_t max_cg     = 100)
{
  auto newton_norm2 = ctx.logical_data(shape_of<scalar_view<double>>()).set_symbol("newton_norm2");
  auto newton_iter  = ctx.logical_data(shape_of<scalar_view<size_t>>()).set_symbol("newton_iter");
  ctx.parallel_for(box(1), newton_iter.write()).set_symbol("init_newton_iter")->*[] _CCCL_DEVICE(size_t i, auto diter) {
    *diter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    compute_residual(ctx, U, U_prev, residual, N, h, dt, nu);

    // Compute Newton residual norm for adaptive CG tolerance
    DOT(ctx, residual, residual, newton_norm2);

    // Adaptive CG tolerance: fixed for now
    double cg_tol = 1e-8;

    assemble_jacobian(ctx, U, csr_values, N, h, dt, nu);

    ctx.parallel_for(rhs.shape(), rhs.write(), residual.read()).set_symbol("rhs -= residual")
        ->*[] __device__(size_t i, auto drhs, auto dresidual) {
              drhs(i) = -dresidual(i);
            };

    csr_matrix A(csr_values, csr_row_offsets, csr_col_ind);

    // Solve A * delta = rhs with adaptive tolerance
    cg_solver(ctx, A, delta, rhs, cg_tol, max_cg);

    // Update solution: interior unknowns get delta corrections, boundaries stay zero
    ctx.parallel_for(U.shape(), U.rw(), delta.read()).set_symbol("apply delta")
        ->*[N] __device__(size_t i, auto dU, auto ddelta) {
              if (i == 0 || i == N - 1)
              {
                dU(i) = 0.0; // Enforce boundary conditions
              }
              else
              {
                dU(i) += ddelta(i - 1); // Interior: delta[i-1] corresponds to interior unknown at global index i
              }
            };

    while_guard.update_cond(newton_norm2.read(), newton_iter.rw())
        ->*[newton_tol, max_newton] __device__(auto dnorm2, auto diter) {
              (*diter)++; // increment iteration counter
              bool converged = (*dnorm2 < newton_tol * newton_tol);
              return !converged && (*diter < max_newton);
            };
  }

#  if 0
  ctx.host_launch(newton_iter.read(), newton_norm2.read()).set_symbol("report_convergence")->*
  [max_newton, newton_tol](auto diter, auto dnorm2) {
    printf("Newton solver converged after %ld iterations (final residual %e, tolerance %e)\n",
           *diter, sqrt(*dnorm2), newton_tol);
  };
#  endif
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
void dump_solution(
  context_t& ctx, vector_t& U, size_t timestep, size_t N, double h, double dt, const char* filename = "solution.dat")
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

  context_t ctx;

  if (argc > 1)
  {
    N = atoi(argv[1]);
    fprintf(stderr, "N = %zu\n", N);
  }

  double h          = 1.0 / (N - 1);
  size_t n_unknowns = N - 2;

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
  fprintf(stderr, "System size: %zu unknowns, %zu non-zeros\n", n_unknowns, 3 * n_unknowns - 2);
  if (N > 1000000)
  {
    fprintf(stderr, "WARNING: Very large system! Consider using iterative preconditioners for N > 1M\n");
  }
  fprintf(stderr, "=============================\n");

  // First and last rows have 2 entries each, middle rows have 3 entries each
  // Total: 2 + 3*(n_unknowns-2) + 2 = 3*n_unknowns - 2
  size_t nz = 3 * n_unknowns - 2;

  size_t* row_offsets;
  size_t* col_indices;
  cuda_safe_call(cudaHostAlloc(&row_offsets, (n_unknowns + 1) * sizeof(size_t), cudaHostAllocMapped));
  cuda_safe_call(cudaHostAlloc(&col_indices, nz * sizeof(size_t), cudaHostAllocMapped));

  build_tridiag_csr_structure(row_offsets, col_indices, N);

  auto csr_row_offsets = ctx.logical_data(make_slice(row_offsets, n_unknowns + 1)).set_symbol("csr_row");
  auto csr_col_ind     = ctx.logical_data(make_slice(col_indices, nz)).set_symbol("csr_col");
  auto csr_values      = ctx.logical_data(shape_of<slice<double>>(nz)).set_symbol("csr_val");

  auto U      = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("U");
  auto U_prev = ctx.logical_data(shape_of<slice<double>>(N)).set_symbol("U_prev");

  auto residual = ctx.logical_data(shape_of<slice<double>>(n_unknowns)).set_symbol("residual");
  auto rhs      = ctx.logical_data(shape_of<slice<double>>(n_unknowns)).set_symbol("rhs");
  auto delta    = ctx.logical_data(shape_of<slice<double>>(n_unknowns)).set_symbol("delta");

  // This will prevent erroneous modifications and may allow access from concurrent graphs
  csr_row_offsets.set_read_only();
  csr_col_ind.set_read_only();

  // Initial condition
  ctx.parallel_for(U_prev.shape(), U_prev.write()).set_symbol("init conditions")
      ->*[h, N] __device__(size_t i, auto dU_prev) {
            double x = i * h;
            if (i == 0 || i == N - 1)
            {
              dU_prev(i) = 0.0; // Homogeneous Dirichlet boundary conditions
            }
            else
            {
              dU_prev(i) = sin(M_PI * x);
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

      // initial guess: u = u_prev (with boundary conditions)
      ctx.parallel_for(U.shape(), U.write(), U_prev.read()).set_symbol("init_guess")
          ->*[] __device__(size_t i, auto dU, auto dU_prev) {
                dU(i) = dU_prev(i);
              };

      // Solve the nonlinear system using Newton's method
      newton_solver(ctx, U, U_prev, residual, rhs, delta, csr_values, csr_row_offsets, csr_col_ind, N, h, dt, nu);

      // accept timestep
      ctx.parallel_for(U.shape(), U_prev.write(), U.read()).set_symbol("accept timestep")
          ->*[] __device__(size_t i, auto dU_prev, auto dU) {
                dU_prev(i) = dU(i);
              };
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
