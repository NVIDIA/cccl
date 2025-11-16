#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/transform_output_iterator.h>

#include <iostream>

int main()
{
  thrust::host_vector<float> u{4, 3, 2, 1};
  thrust::host_vector<float> v{-1, 1, 1, -1};
  thrust::host_vector<int> idx{3, 0, 1};
  thrust::host_vector<float> w{0, 0, 0};

  thrust::device_vector<float> U(u);
  thrust::device_vector<float> V(v);
  thrust::device_vector<int> IDX(idx);
  thrust::device_vector<float> W(w);

  // gather multiple elements and apply a function before writing result in memory
  thrust::gather(IDX.begin(),
                 IDX.end(),
                 thrust::make_zip_iterator(U.begin(), V.begin()),
                 thrust::make_transform_output_iterator(W.begin(), [] __device__(const auto& tuple) {
                   const float x = thrust::get<0>(tuple);
                   const float y = thrust::get<1>(tuple);
                   return x * y * 2.0f / 3.0f;
                 }));

  std::cout << "result= [ ";
  for (const auto& value : W)
  {
    std::cout << value << " ";
  }
  std::cout << "] \n";

  return 0;
}
