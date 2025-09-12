#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>

int main()
{
  // H holds 4 integers
  thrust::host_vector<int> H{14, 20, 38, 46};

  // H.size() returns the size of vector H
  std::cout << "H has size " << H.size() << std::endl;

  // print contents of H
  for (size_t i = 0; i < H.size(); i++)
  {
    std::cout << "H[" << i << "] = " << H[i] << std::endl;
  }

  // resize H
  H.resize(2);

  std::cout << "H now has size " << H.size() << std::endl;

  // Copy host_vector H to device_vector D
  thrust::device_vector<int> D = H;

  // elements of D can be modified
  D[0] = 99;
  D[1] = 88;

  // print contents of D
  for (size_t i = 0; i < D.size(); i++)
  {
    std::cout << "D[" << i << "] = " << D[i] << std::endl;
  }

  // H and D are automatically deleted when the function returns
  return 0;
}
