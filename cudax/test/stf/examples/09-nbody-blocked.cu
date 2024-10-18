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
};

// Function to write VTK file for a single time step
void writeVTKFile(context& ctx,
                  const std::string& filename,
                  size_t BLOCK_SIZE,
                  size_t BODY_CNT,
                  std::vector<logical_data<slice<body>>> parts)
{
  std::ofstream outfile(filename);

  if (!outfile)
  {
    std::cerr << "Error opening file: " << filename << std::endl;
    return;
  }

  outfile << "# vtk DataFile Version 4.2\n";
  outfile << "Position Data\n";
  outfile << "ASCII\n";
  outfile << "DATASET UNSTRUCTURED_GRID\n";
  outfile << "POINTS " << BODY_CNT << " float\n";

  std::vector<double> dump(3 * BODY_CNT);

  for (size_t b = 0; b < parts.size(); b++)
  {
    ctx.task(exec_place::host, parts[b].read())->*[&](cudaStream_t s, slice<body> p) {
      cuda_safe_call(cudaStreamSynchronize(s));
      for (size_t i = 0; i < p.size(); i++)
      {
        for (size_t k = 0; k < 3; k++)
        {
          dump[3 * (i + b * BLOCK_SIZE) + k] = p(i).pos[k];
        }
      }
    };
  }

  for (size_t p = 0; p < BODY_CNT; p++)
  {
    outfile << dump[3 * p] << " " << dump[3 * p + 1] << " " << dump[3 * p + 2] << "\n";
  }

  outfile.close();
}

void load_input_file(std::string filename, std::vector<body>& particles)
{
  std::ifstream infile(filename);
  if (!infile)
  {
    std::cerr << "Error opening file." << std::endl;
    abort();
    return;
  }

  double mass, posX, posY, posZ, velX, velY, velZ;
  size_t ind = 0;

  // Loop until we reach the end of the file
  while (infile >> mass >> posX >> posY >> posZ >> velX >> velY >> velZ)
  {
    body p;
    p.mass = mass;

    p.pos[0] = posX;
    p.pos[1] = posY;
    p.pos[2] = posZ;

    p.vel[0] = velX;
    p.vel[1] = velY;
    p.vel[2] = velZ;

    // // Display first bodies
    // if (ind < 10) {
    //    fprintf(stderr, "body xyz %e %e %e dxyz %e %e %e m %e\n", p.pos[0], p.pos[1], p.pos[2], p.vel[0],
    //            p.vel[1], p.vel[2], p.mass);
    //}

    ind++;
    particles.push_back(p);
  }

  fprintf(stderr, "Loaded %zu bodies from %s...\n", ind, filename.c_str());
}

int main(int argc, char** argv)
{
  constexpr double kSofteningSquared = 1e-9;
  //    constexpr double kG = 6.67259e-11;
  constexpr double kG = 1.0;

  size_t BODY_CNT   = 128ULL * 1024ULL;
  size_t BLOCK_SIZE = 16 * 1024ULL;

  std::vector<body> particles;

  // Initialize particles
  if (argc > 1)
  {
    // Get dataset from file
    std::string filename = argv[1];

    load_input_file(filename, particles);

    BODY_CNT   = particles.size();
    BLOCK_SIZE = (BODY_CNT + 7) / 8;
  }
  else
  {
    // Random distribution
    BODY_CNT = 32ULL * 1024ULL;
    particles.resize(BODY_CNT);

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
    }
  }

  cuda_safe_call(cudaHostRegister(&particles[0], BODY_CNT * sizeof(body), cudaHostRegisterPortable));

  double dt    = 0.005;
  size_t NITER = 7; // 7000;

  context ctx;

  std::vector<logical_data<slice<body>>> parts;

  // Accelerations
  std::vector<logical_data<slice<double, 2>>> acc_parts;

  size_t block_cnt = (BODY_CNT + BLOCK_SIZE - 1) / BLOCK_SIZE;
  for (size_t i = 0; i < block_cnt; i++)
  {
    size_t first = i * BLOCK_SIZE;
    size_t last  = std::min((i + 1) * BLOCK_SIZE, BODY_CNT);
    auto p_i     = ctx.logical_data(make_slice(&particles[first], last - first));
    parts.push_back(p_i);

    auto acc_p_i = ctx.logical_data(shape_of<slice<double, 2>>(last - first, 3));
    acc_parts.push_back(acc_p_i);
  }

  int ngpus;
  cuda_safe_call(cudaGetDeviceCount(&ngpus));

  cudaEvent_t start;
  cuda_safe_call(cudaEventCreate(&start));
  cuda_safe_call(cudaEventRecord(start, ctx.task_fence()));

  for (size_t iter = 0; iter < NITER; iter++)
  {
    // Initialize acceleration to 0
    for (size_t b = 0; b < block_cnt; b++)
    {
      ctx.launch(exec_place::device(b % ngpus), acc_parts[b].write())
          //.set_symbol("init_acc")
          ->*[=] _CCCL_DEVICE(auto t, slice<double, 2> acc) {
                for (size_t i = t.rank(); i < acc.extent(0); i += t.size())
                {
                  for (size_t k = 0; k < 3; k++)
                  {
                    acc(i, k) = 0.0;
                  }
                }
              };
    }

    // Compute accelerations
    for (size_t b = 0; b < block_cnt; b++)
    {
      for (size_t b_other = 0; b_other < block_cnt; b_other++)
      {
        ctx.launch(exec_place::device(b % ngpus), parts[b].read(), parts[b_other].read(), acc_parts[b].rw())
            //.set_symbol("compute_acc")
            ->*[=] _CCCL_DEVICE(auto t, slice<body> p, slice<body> p_other, slice<double, 2> acc) {
                  for (size_t i = t.rank(); i < p.extent(0); i += t.size())
                  {
                    for (size_t j = 0; j < p_other.extent(0); j++)
                    {
                      if ((b * BLOCK_SIZE + i) != (b_other * BLOCK_SIZE + j))
                      {
                        double d[3];
                        for (size_t k = 0; k < 3; k++)
                        {
                          d[k] = p_other(j).pos[k] - p(i).pos[k];
                        }

                        double dist     = d[0] * d[0] + d[1] * d[1] + d[2] * d[2] + kSofteningSquared;
                        double dist_inv = 1.0 / sqrt(dist);

                        for (size_t k = 0; k < 3; k++)
                        {
                          acc(i, k) += d[k] * kG * p_other(j).mass * dist_inv * dist_inv * dist_inv;
                        }
                      }
                    }
                  }
                };
      }
    }

    for (size_t b = 0; b < block_cnt; b++)
    {
      // Update velocity and positions
      ctx.launch(exec_place::device(b % ngpus), parts[b].rw(), acc_parts[b].read())
          //.set_symbol("update")
          ->*[=] _CCCL_DEVICE(auto t, slice<body> p, slice<double, 2> acc) {
                for (size_t i = t.rank(); i < p.extent(0); i += t.size())
                {
                  for (size_t k = 0; k < 3; k++)
                  {
                    p(i).vel[k] += acc(i, k) * dt;
                  }

                  for (size_t k = 0; k < 3; k++)
                  {
                    p(i).pos[k] += p(i).vel[k] * dt;
                  }
                }
              };
    }

    // Write the VTK file for this time step
    const char* dump_freq_str = getenv("DUMP_FREQ");
    if (dump_freq_str && iter % atoi(dump_freq_str) == 0)
    {
      std::string filename = "time_step_" + std::to_string(iter) + ".vtk";
      writeVTKFile(ctx, filename, BLOCK_SIZE, BODY_CNT, parts);
    }
  }

  cudaEvent_t stop;
  cuda_safe_call(cudaEventCreate(&stop));
  cuda_safe_call(cudaEventRecord(stop, ctx.task_fence()));

  ctx.finalize();

  float elapsed;
  cuda_safe_call(cudaEventElapsedTime(&elapsed, start, stop));

  // rough approximation !
  double FLOP_COUNT = 21.0 * (1.0 * BODY_CNT) * (1.0 * BODY_CNT) * NITER;

  printf("NBODY: elapsed %f ms, %f GFLOPS\n", elapsed, FLOP_COUNT / elapsed / 1000000.0);
}
