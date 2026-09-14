#pragma once

#include <thrust/system/cuda/memory.h>
#include <thrust/system_error.h>

#include <vector>

#include <unittest/testframework.h>

class CUDATestDriver : public UnitTestDriver
{
public:
  int current_device_architecture() const;

  bool run_tests(const ArgumentSet& args, const ArgumentMap& kwargs) override;

protected:
  bool post_test_smoke_check(const UnitTest& test, bool concise) override;

private:
  std::vector<int> target_devices(const ArgumentMap& kwargs);

  bool check_cuda_error(bool concise);
};

UnitTestDriver& driver_instance(thrust::system::cuda::tag);
