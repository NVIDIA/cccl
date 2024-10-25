//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/stf.cuh>

#include <random>

using namespace cuda::experimental::stf;

struct body
{
  // mass
  double mass;
  // position
  double pos[3];
  // speed
  double vel[3];
  // acceleration
  double acc[3];
};

int main()
{
  constexpr double kSofteningSquared = 1e-3;
  constexpr double kG                = 6.67259e-11;

  size_t BODY_CNT = 4096;

  double dt    = 0.1;
  size_t NITER = 25;

  context ctx = graph_ctx();

  std::vector<body> particles;
  particles.resize(BODY_CNT);

  // Initialize particles
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (auto& p : particles)
  {
    p.mass = 1.0;

    p.pos[0] = dis(gen);
    p.pos[1] = dis(gen);
    p.pos[2] = dis(gen);

    p.vel[0] = dis(gen);
    p.vel[1] = dis(gen);
    p.vel[2] = dis(gen);

    p.acc[0] = 0.0;
    p.acc[1] = 0.0;
    p.acc[2] = 0.0;
  }

  auto h_particles = ctx.logical_data(make_slice(&particles[0], BODY_CNT));

  ctx.repeat(NITER)->*[&](context ctx, size_t) {
    // Compute accelerations
    ctx.parallel_for(h_particles.shape(), h_particles.rw())->*[=] _CCCL_DEVICE __host__(size_t i, slice<body> p) {
      double acc[3];
      for (size_t k = 0; k < 3; k++)
      {
        acc[k] = p(i).acc[k];
      }

      for (size_t j = 0; j < p.extent(0); j++)
      {
        if (i != j)
        {
          double d[3];
          for (size_t k = 0; k < 3; k++)
          {
            d[k] = p(j).pos[k] - p(i).pos[k];
          }

          double dist     = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + kSofteningSquared;
          double dist_inv = 1.0 / sqrt(dist);

          for (size_t k = 0; k < 3; k++)
          {
            acc[k] += d[k] * kG * p(j).mass * dist_inv * dist_inv * dist_inv;
          }
        }
      }

      for (size_t k = 0; k < 3; k++)
      {
        p(i).acc[k] = acc[k];
      }
    };

    // Update velocity and positions
    ctx.parallel_for(h_particles.shape(), h_particles.rw())->*[=] __host__ __device__(size_t i, slice<body> p) {
      for (size_t k = 0; k < 3; k++)
      {
        p(i).vel[k] += p(i).acc[k] * dt;
      }

      for (size_t k = 0; k < 3; k++)
      {
        p(i).pos[k] += p(i).vel[k] * dt;
      }

      for (size_t k = 0; k < 3; k++)
      {
        p(i).acc[k] = 0.0;
      }
    };
  };

  ctx.finalize();
}
