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
 * @brief Simulation of a fluid over a regular grid with an implicit Jacobi solver
 */

#include <cuda/experimental/stf.cuh>

#include <chrono>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std::chrono;
using namespace cuda::experimental::stf;

/* wall-clock time */
double gettime()
{
  auto now = system_clock::now().time_since_epoch();
  return duration_cast<duration<double>>(now).count();
}

void writeplotfile(int m, int n, int scale)
{
  FILE* gnuplot = EXPECT(fopen("cfd.plt", "w"));
  SCOPE(exit)
  {
    EXPECT(fclose(gnuplot) == 0);
  };

  fprintf(gnuplot,
          "set terminal pngcairo\n"
          "set output 'cfd_output.png'\n"
          "set size square\n"
          "set key off\n"
          "unset xtics\n"
          "unset ytics\n");

  fprintf(gnuplot, "set xrange [%i:%i]\n", 1 - scale, m + scale);
  fprintf(gnuplot, "set yrange [%i:%i]\n", 1 - scale, n + scale);

  fprintf(gnuplot,
          "plot \"colourmap.dat\" w rgbimage, \"velocity.dat\" u "
          "1:2:(%d*0.75*$3/sqrt($3**2+$4**2)):(%d*0.75*$4/sqrt($3**2+$4**2)) with vectors  lc rgb \"#7F7F7F\"",
          scale,
          scale);

  // printf("\nWritten gnuplot script 'cfd.plt'\n");
}

double colfunc(double x)
{
  double x1 = 0.2;
  double x2 = 0.5;

  double absx = fabs(x);

  if (absx > x2)
  {
    return 0.0;
  }
  else if (absx < x1)
  {
    return 1.0;
  }
  else
  {
    return 1.0 - pow((absx - x1) / (x2 - x1), 2);
  }
}

void hue2rgb(double hue, int& r, int& g, int& b)
{
  int rgbmax = 255;

  r = (int) (rgbmax * colfunc(hue - 1.0));
  g = (int) (rgbmax * colfunc(hue - 0.5));
  b = (int) (rgbmax * colfunc(hue));
}

void writedatafiles(context& ctx, logical_data<slice<double, 2>> lpsi, int m, int n, int scale)
{
  auto lvel = ctx.logical_data(shape_of<slice<double, 3>>(m, n, 2)).set_symbol("vel");
  auto lrgb = ctx.logical_data(shape_of<slice<int, 3>>(m, n, 3)).set_symbol("rgb");

  ctx.host_launch(lpsi.read(), lvel.write(), lrgb.write()).set_symbol("writedatafiles")
      ->*[=](auto psi, auto vel, auto rgb) {
            // printf("\n\nWriting data files ...\n");

            // calculate velocities and hues

            for (int i = 0; i < m; i++)
            {
              for (int j = 0; j < n; j++)
              {
                vel(i, j, 0) = (psi(i + 1, j + 2) - psi(i + 1, j)) / 2.0;
                vel(i, j, 1) = -(psi(i + 2, j + 1) - psi(i, j + 1)) / 2.0;

                double v1 = vel(i, j, 0);
                double v2 = vel(i, j, 1);

                double modvsq = v1 * v1 + v2 * v2;

                double hue = pow(modvsq, 0.4);

                hue2rgb(hue, rgb(i, j, 0), rgb(i, j, 1), rgb(i, j, 2));
              }
            }

            // write data

            FILE* cfile = EXPECT(fopen("colourmap.dat", "w"));
            SCOPE(exit)
            {
              fclose(cfile);
            };
            FILE* vfile = EXPECT(fopen("velocity.dat", "w"));
            SCOPE(exit)
            {
              fclose(vfile);
            };

            for (int i = 0; i < m; i++)
            {
              int ix = i + 1;

              for (int j = 0; j < n; j++)
              {
                int iy = j + 1;

                fprintf(cfile, "%i %i %i %i %i\n", ix, iy, rgb(i, j, 0), rgb(i, j, 1), rgb(i, j, 2));

                if ((ix - 1) % scale == (scale - 1) / 2 && (iy - 1) % scale == (scale - 1) / 2)
                {
                  fprintf(vfile, "%i %i %f %f\n", ix, iy, vel(i, j, 0), vel(i, j, 1));
                }
              }
            }

            // printf("... done!\n");

            writeplotfile(m, n, scale);
          };
}

void jacobistep(context& ctx, logical_data<slice<double, 2>> lpsinew, logical_data<slice<double, 2>> lpsi, int m, int n)
{
  ctx.parallel_for(box<2>({1, m + 1}, {1, n + 1}), lpsinew.write(), lpsi.read()).set_symbol("jacobi_step")
      ->*[] __device__(size_t i, size_t j, auto psinew, auto psi) {
            psinew(i, j) = 0.25 * (psi(i - 1, j) + psi(i + 1, j) + psi(i, j + 1) + psi(i, j - 1));
          };
}

void jacobistepvort(
  context& ctx,
  logical_data<slice<double, 2>> lzetnew,
  logical_data<slice<double, 2>> lpsinew,
  logical_data<slice<double, 2>> lzet,
  logical_data<slice<double, 2>> lpsi,
  int m,
  int n,
  double re)
{
  ctx.parallel_for(box<2>({1, m + 1}, {1, n + 1}), lpsinew.write(), lpsi.read(), lzet.read())
      .set_symbol("jacobi_step_psi")
      ->*[] __device__(size_t i, size_t j, auto psinew, auto psi, auto zet) {
            psinew(i, j) = 0.25 * (psi(i - 1, j) + psi(i + 1, j) + psi(i, j + 1) + psi(i, j - 1)) - zet(i, j);
          };

  ctx.parallel_for(box<2>({1, m + 1}, {1, n + 1}), lzetnew.write(), lzet.read(), lpsi.read())
      .set_symbol("jacobi_step_zet")
      ->*[=] __device__(size_t i, size_t j, auto zetnew, auto zet, auto psi) {
            zetnew(i, j) = 0.25 * (zet(i - 1, j) + zet(i + 1, j) + zet(i, j + 1) + zet(i, j - 1))
                         - re / 16.0
                             * ((psi(i, j + 1) - psi(i, j - 1)) * (zet(i + 1, j) - zet(i - 1, j))
                                - (psi(i + 1, j) - psi(i - 1, j)) * (zet(i, j + 1) - zet(i, j - 1)));
          };
}

template <typename T>
T transfer_host(context& ctx, logical_data<slice<T>>& ldata)
{
  T out;

  bool is_graph = ctx.is_graph_ctx();

  if (is_graph)
  {
    ctx.host_launch(ldata.read()).set_symbol("transfer_host")->*[&](auto data) {
      out = data(0);
    };

    /* This forces the completion of the host callback, so that the host
     * thread can use the content for dynamic control flow */
    cudaStreamSynchronize(ctx.task_fence());
  }
  else
  {
    ctx.task(exec_place::host, ldata.read()).set_symbol("transfer_host")->*[&](cudaStream_t stream, auto data) {
      cuda_safe_call(cudaStreamSynchronize(stream));
      out = data(0);
    };
  }

  return out;
}

double
deltasq(context& ctx, logical_data<slice<double, 2>> lnewarr, logical_data<slice<double, 2>> loldarr, int m, int n)
{
  auto ldsq = ctx.logical_data(shape_of<slice<double>>({1})).set_symbol("tmp_accumulator");

  //
  //    for (i = 1; i <= m; i++) {
  //        for (j = 1; j <= n; j++) {
  //            double tmp = newarr[i * (m + 2) + j] - oldarr[i * (m + 2) + j];
  //            dsq += tmp * tmp;
  //        }
  //    }

  auto spec = con(con<128>(hw_scope::thread));
  ctx.launch(spec, ldsq.write(), lnewarr.read(), loldarr.read()).set_symbol("deltasq")->*
    [m, n] __device__(auto th, auto dsq, auto newarr, auto oldarr) {
      if (th.rank() == 0)
      {
        dsq(0) = 0.0;
      }
      th.sync();

      // Each thread computes the sum of elements assigned to it
      double local_sum = 0.0;
      for (auto [i, j] :
           th.apply_partition(box<2>({1, m + 1}, {1, n + 1}), std::tuple<blocked_partition, cyclic_partition>()))
      {
        double tmp = newarr(i, j) - oldarr(i, j);
        local_sum += tmp * tmp;
      }

      auto ti = th.inner();

      __shared__ double block_sum[th.static_width(1)];
      block_sum[ti.rank()] = local_sum;

      for (size_t s = ti.size() / 2; s > 0; s /= 2)
      {
        if (ti.rank() < s)
        {
          block_sum[ti.rank()] += block_sum[ti.rank() + s];
        }
        ti.sync();
      }

      if (ti.rank() == 0)
      {
        atomicAdd(&dsq(0), block_sum[0]);
      }
    };

  return transfer_host(ctx, ldsq);
}

void boundarypsi(context& ctx, logical_data<slice<double, 2>> lpsi, int m, int /*n*/, int b, int h, int w)
{
  // BCs on bottom edge
  ctx.parallel_for(box({b + 1, b + w}), lpsi.rw()).set_symbol("boundary_bottom")->*[=] __device__(size_t i, auto psi) {
    psi(i, 0) = double(i - b);
  };

  ctx.parallel_for(box<1>({b + w, m + 1}), lpsi.rw()).set_symbol("boundary_bottom")->*[=] __device__(size_t i, auto psi) {
    psi(i, 0) = double(w);
  };

  // BCS on RHS
  ctx.parallel_for(box({1, h + 1}), lpsi.rw()).set_symbol("boundary_right")->*[=] __device__(size_t j, auto psi) {
    psi(m + 1, j) = double(w);
  };

  ctx.parallel_for(box({h + 1, h + w}), lpsi.rw()).set_symbol("boundary_right")->*[=] __device__(size_t j, auto psi) {
    psi(m + 1, j) = (double) (w - j + h);
  };
}

void boundaryzet(context& ctx, logical_data<slice<double, 2>> lzet, logical_data<slice<double, 2>> lpsi, int m, int n)
{
  // set top/bottom BCs:
  ctx.parallel_for(box({1, m + 1}), lzet.rw(), lpsi.read()).set_symbol("boundary_topbottom")
      ->*[=] __device__(size_t i, auto zet, auto psi) {
            zet(i, 0)     = 2.0 * (psi(i, 1) - psi(i, 0));
            zet(i, n + 1) = 2.0 * (psi(i, n) - psi(i, n + 1));
          };

  // set left and right BCs:
  ctx.parallel_for(box({1, n + 1}), lzet.rw(), lpsi.read()).set_symbol("boundary_leftright")
      ->*[=] __device__(size_t j, auto zet, auto psi) {
            zet(0, j)     = 2.0 * (psi(1, j) - psi(0, j));
            zet(m + 1, j) = 2.0 * (psi(m, j) - psi(m + 1, j));
          };
}

int main(int argc, char** argv)
{
  context ctx;

  int printfreq    = 10; // output frequency
  double error     = -1.0;
  double tolerance = 0.0001; //-1.0;  // 0.0001; //tolerance for convergence. <=0 means do not check

  // command line arguments
  int scalefactor = 1, numiter = 10;

  double re = -1.0; // Reynold's number - must be less than 3.7

  // simulation sizes
  int bbase = 10;
  int hbase = 15;
  int wbase = 5;
  int mbase = 32;
  int nbase = 32;

  int irrotational = 1, checkerr = 0;

  // do we stop because of tolerance?
  if (tolerance > 0)
  {
    checkerr = 1;
  }

  // check command line parameters and parse them

  if (argc > 5)
  {
    printf("Usage: cfd <scale> <numiter> [reynolds] [use_graphs]\n");
    return 0;
  }

  if (argc > 1)
  {
    scalefactor = atoi(argv[1]);
  }

  if (argc > 2)
  {
    numiter = atoi(argv[2]);
  }

  if (argc > 3)
  {
    re           = atof(argv[3]);
    irrotational = 0;
  }

  // Use a CUDA graph backend
  if (argc > 4)
  {
    if (atoi(argv[4]) == 1)
    {
      ctx = graph_ctx();
    }
    fprintf(stderr, "Using %s backend.\n", ctx.to_string().c_str());
  }

  // if (!checkerr) {
  //     printf("Scale Factor = %i, iterations = %i\n", scalefactor, numiter);
  // } else {
  //     printf("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor, numiter, tolerance);
  // }

  // if (irrotational) {
  //     printf("Irrotational flow\n");
  // } else {
  //     printf("Reynolds number = %f\n", re);
  // }

  tolerance /= scalefactor;

  // Calculate b, h & w and m & n
  int b = bbase * scalefactor;
  int h = hbase * scalefactor;
  int w = wbase * scalefactor;
  int m = mbase * scalefactor;
  int n = nbase * scalefactor;

  re /= scalefactor;

  // printf("Running CFD on %d x %d grid in serial\n", m, n);

  // main arrays and their temporary versions
  logical_data<slice<double, 2>> lzet, lzettmp, lpsi, lpsitmp;

  // allocate arrays
  lpsi    = ctx.logical_data(shape_of<slice<double, 2>>(m + 2, n + 2)).set_symbol("psi");
  lpsitmp = ctx.logical_data(lpsi.shape()).set_symbol("psi_tmp");

  // zero the psi array
  ctx.parallel_for(lpsi.shape(), lpsi.write()).set_symbol("InitPsi")->*[] __device__(size_t i, size_t j, auto psi) {
    psi(i, j) = 0.0;
  };

  if (!irrotational)
  {
    lzet    = ctx.logical_data(lpsi.shape()).set_symbol("zet");
    lzettmp = ctx.logical_data(lpsi.shape()).set_symbol("zet_tmp");

    // zero the zeta array
    ctx.parallel_for(lzet.shape(), lzet.write()).set_symbol("InitZet")->*[] __device__(size_t i, size_t j, auto zet) {
      zet(i, j) = 0.0;
    };
  }

  // set the psi boundary conditions
  boundarypsi(ctx, lpsi, m, n, b, h, w);

  // compute normalisation factor for error
  auto lbnorm = ctx.logical_data(shape_of<slice<double>>({1})).set_symbol("bnorm");

  nvtxRangePush("Compute_Normalization");

  // bnorm += psi * psi
  auto spec = con(con<32>());
  ctx.launch(spec, lbnorm.write(), lpsi.read()).set_symbol("Compute_Normalization")
      ->*[] __device__(auto th, auto bnorm, auto psi) {
            if (th.rank() == 0)
            {
              bnorm(0) = 0.0;
            }
            th.sync();
            // Each thread computes the sum of elements assigned to it
            double local_sum = 0.0;
            for (auto [i, j] : th.apply_partition(shape(psi)))
            {
              local_sum += psi(i, j) * psi(i, j);
            }

            auto ti = th.inner();

            __shared__ double block_sum[th.static_width(1)];
            block_sum[ti.rank()] = local_sum;

            for (size_t s = ti.size() / 2; s > 0; s /= 2)
            {
              if (ti.rank() < s)
              {
                block_sum[ti.rank()] += block_sum[ti.rank() + s];
              }
              ti.sync();
            }

            if (ti.rank() == 0)
            {
              atomicAdd(&bnorm(0), block_sum[0]);
            }
          };

  if (!irrotational)
  {
    // update zeta BCs that depend on psi
    boundaryzet(ctx, lzet, lpsi, m, n);

    // update normalisation
    ctx.launch(spec, lbnorm.rw(), lzet.read()).set_symbol("Compute_Normalization")
        ->*[] __device__(auto th, auto bnorm, auto zet) {
              // Each thread computes the sum of elements assigned to it
              double local_sum = 0.0;
              for (auto [i, j] : th.apply_partition(shape(zet)))
              {
                local_sum += zet(i, j) * zet(i, j);
              }

              auto ti = th.inner();

              __shared__ double block_sum[th.static_width(1)];
              block_sum[ti.rank()] = local_sum;

              for (size_t s = ti.size() / 2; s > 0; s /= 2)
              {
                if (ti.rank() < s)
                {
                  block_sum[ti.rank()] += block_sum[ti.rank() + s];
                }
                ti.sync();
              }

              if (ti.rank() == 0)
              {
                atomicAdd(&bnorm(0), block_sum[0]);
              }
            };
  }

  double bnorm = transfer_host(ctx, lbnorm);
  bnorm        = sqrt(bnorm);

  // begin iterative Jacobi loop

  // printf("\nStarting main loop...\n\n");

  double tstart = gettime();
  nvtxRangePush("Overall_Iteration");

  int iter = 1;
  for (; iter <= numiter; iter++)
  {
    // calculate psi for next iteration
    if (irrotational)
    {
      jacobistep(ctx, lpsitmp, lpsi, m, n);
    }
    else
    {
      jacobistepvort(ctx, lzettmp, lpsitmp, lzet, lpsi, m, n, re);
    }

    // calculate current error if required
    bool compute_error = (iter == numiter) || (checkerr && (iter % printfreq == 0));
    if (compute_error)
    {
      error = deltasq(ctx, lpsitmp, lpsi, m, n);

      if (!irrotational)
      {
        error += deltasq(ctx, lzettmp, lzet, m, n);
      }

      error = sqrt(error);
      error = error / bnorm;

      if (checkerr && (error < tolerance))
      {
        // printf("Converged on iteration %d\n", iter);
        break;
      }
    }

    // copy back
    ctx.parallel_for(box<2>({1, m + 1}, {1, n + 1}), lpsi.rw(), lpsitmp.read()).set_symbol("SwitchPsi")
        ->*[] __device__(size_t i, size_t j, auto psi, auto psitmp) {
              psi(i, j) = psitmp(i, j);
            };

    if (!irrotational)
    {
      ctx.parallel_for(box<2>({1, m + 1}, {1, n + 1}), lzet.rw(), lzettmp.read()).set_symbol("SwitchZet")
          ->*[] __device__(size_t i, size_t j, auto zet, auto zettmp) {
                zet(i, j) = zettmp(i, j);
              };
    }

    if (!irrotational)
    {
      // update zeta BCs that depend on psi
      boundaryzet(ctx, lzet, lpsi, m, n);
    }

    // if (iter % printfreq == 0) {
    //     if (!checkerr) {
    //         printf("Completed iteration %d\n", iter);
    //     } else {
    //         printf("Completed iteration %d, error = %g\n", iter, error);
    //     }
    // }
  }
  nvtxRangePop(); // pop

  if (iter > numiter)
  {
    iter = numiter;
  }

  double tstop = gettime();

  double ttot  = tstop - tstart;
  double titer = ttot / (double) iter;

  // output results

  writedatafiles(ctx, lpsi, m, n, scalefactor);

  ctx.finalize();
  // print out some stats

  // printf("\n... finished\n");
  printf("After %d iterations, the error is %g\n", iter, error);
  printf("Time for %d iterations was %g seconds\n", iter, ttot);
  printf("Each iteration took %g seconds\n", titer);

  // printf("... finished\n");

  return 0;
}
