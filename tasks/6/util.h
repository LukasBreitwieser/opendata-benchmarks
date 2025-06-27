
#ifndef UTIL_H_
#define UTIL_H_

#include <stdexcept>
#include <vector>

template <typename T>
void cudaMallocAndCopy(const std::vector<T> &host_vector,
                       T **ptr_to_device_array) {
  if (cudaMalloc(reinterpret_cast<void **>(ptr_to_device_array),
                 host_vector.size() * sizeof(T)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }
  cudaMemcpy(*ptr_to_device_array, host_vector.data(),
             host_vector.size() * sizeof(T), cudaMemcpyHostToDevice);
}

template <typename T>
void cudaCopy2HostAndFree(std::vector<T> &host_vector, T *device_array) {
  cudaMemcpy(host_vector.data(), device_array, host_vector.size() * sizeof(T),
             cudaMemcpyDeviceToHost);
  cudaFree(device_array);
}

#endif // UTIL_H_
