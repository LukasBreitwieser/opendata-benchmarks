#ifndef FLATTENED_JAGGED_VEC_H_
#define FLATTENED_JAGGED_VEC_H_

#include "ROOT/RVec.hxx"
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

template <typename T> using RVec = ROOT::RVec<T>;
template <typename T> using JaggedVec = std::vector<RVec<T>>;

template <typename T> class FlattenedJaggedVec {
public:
  FlattenedJaggedVec();
  FlattenedJaggedVec(const JaggedVec<T> &jagged);
  void CopyToDevice();
  // FlattenedJaggedVec operator=(const JaggedVec<T> &jagged_vector);

  struct Vec {
    T data_;
    std::size_t size_;

    __device__ std::size_t size() const;
    __device__ T operator[](std::size_t idx) const;
  };

  __device__ Vec operator[](std::size_t idx) const;

private:
  std::vector<T> host_flattened_data_;
  std::vector<int64_t> host_sizes_;
  std::vector<int64_t> host_offsets_;

  T *device_flattened_data_;
  int64_t *device_sizes_;
  int64_t *device_offsets_;
};

template <typename T> FlattenedJaggedVec<T>::FlattenedJaggedVec() {}

template <typename T>
FlattenedJaggedVec<T>::FlattenedJaggedVec(const JaggedVec<T> &jagged) {
  host_sizes_.reserve(jagged.size());
  host_offsets_.reserve(jagged.size());
  for (auto &vec : jagged) {
    if (!host_offsets_.empty()) {
      host_offsets_.push_back(host_offsets_.back() + host_sizes_.back());
    } else {
      host_offsets_.push_back(0);
    }
    host_sizes_.push_back(vec.size());
  }

  host_flattened_data_.resize(host_offsets_.back() + host_sizes_.back());

  // copy data
  // TODO use openmp
  for (int64_t i = 0; i < jagged.size(); ++i) {
    auto offset = host_offsets_[i];
    auto &vec = jagged[i];
    for (int64_t j = 0; j < vec.size(); ++j) {
      host_flattened_data_[offset + j] = vec[j];
    }
  }
}

template <typename T> void FlattenedJaggedVec<T>::CopyToDevice() {
  if (cudaMalloc(reinterpret_cast<void **>(&device_flattened_data_),
                 host_flattened_data_.size() * sizeof(T)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_sizes_),
                 host_sizes_.size() * sizeof(int64_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_offsets_),
                 host_offsets_.size() * sizeof(int64_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  cudaMemcpy(device_flattened_data_, host_flattened_data_.data(),
             host_flattened_data_.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(device_sizes_, host_sizes_.data(),
             host_sizes_.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_offsets_, host_offsets_.data(),
             host_offsets_.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
}

template <typename T>
typename FlattenedJaggedVec<T>::Vec
FlattenedJaggedVec<T>::operator[](std::size_t idx) const {
  return FlattenedJaggedVec::Vec(&device_flattened_data_[device_offsets_[idx]],
                                 device_sizes_[idx]);
}

template <typename T> std::size_t FlattenedJaggedVec<T>::Vec::size() const {
  return size_;
}

template <typename T>
__device__ T FlattenedJaggedVec<T>::Vec::operator[](std::size_t idx) const { return data_[idx]; }

#endif // FLATTENED_JAGGED_VEC_H_
