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
  template <typename U>
  void ReserveDataAndCopySizesAndOffsetsToDevice(
      const FlattenedJaggedVec<U> &flattened);

  struct DeviceAttr {
    T *flattened_data;
    int64_t *sizes;
    int64_t *offsets;

    class Vec {
      T *data_;
      std::size_t size_;

    public:
      __device__ Vec(T *data, std::size_t size) : data_(data), size_(size) {}
      __device__ std::size_t size() const;
      __device__ T operator[](std::size_t idx) const;
    };

    __device__ Vec operator[](std::size_t idx) const;
  };

  DeviceAttr GetDeviceAttr() const { return device_attributes_; }

private:
  template <typename U> friend class FlattenedJaggedVec;

  std::vector<T> host_flattened_data_;
  std::vector<int64_t> host_sizes_;
  std::vector<int64_t> host_offsets_;

  DeviceAttr device_attributes_;
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
  if (cudaMalloc(reinterpret_cast<void **>(&device_attributes_.flattened_data),
                 host_flattened_data_.size() * sizeof(T)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_attributes_.sizes),
                 host_sizes_.size() * sizeof(int64_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_attributes_.offsets),
                 host_offsets_.size() * sizeof(int64_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  cudaMemcpy(device_attributes_.flattened_data, host_flattened_data_.data(),
             host_flattened_data_.size() * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(device_attributes_.sizes, host_sizes_.data(),
             host_sizes_.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(device_attributes_.offsets, host_offsets_.data(),
             host_offsets_.size() * sizeof(int64_t), cudaMemcpyHostToDevice);
}

template <typename T>
template <typename U>
void FlattenedJaggedVec<T>::ReserveDataAndCopySizesAndOffsetsToDevice(
    const FlattenedJaggedVec<U> &other) {
  if (other.host_flattened_data_.empty()) {
    std::cout
        << "Warning: "
           "FlattenedJaggedVec<T>::ReserveDataAndCopySizesAndOffsetsToDevice: "
        << "the given argument has zero size. " << std::endl;
  }

  // allocate device memory
  if (cudaMalloc(reinterpret_cast<void **>(&device_attributes_.flattened_data),
                 other.host_flattened_data_.size() * sizeof(T)) !=
      cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_attributes_.sizes),
                 other.host_sizes_.size() * sizeof(int64_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  if (cudaMalloc(reinterpret_cast<void **>(&device_attributes_.offsets),
                 other.host_offsets_.size() * sizeof(int64_t)) != cudaSuccess) {
    throw std::runtime_error("cudaMalloc failed!");
  }

  // TODO avoid copies
  // copy data
  cudaMemcpy(device_attributes_.sizes, other.host_sizes_.data(),
             other.host_sizes_.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(device_attributes_.offsets, other.host_offsets_.data(),
             other.host_offsets_.size() * sizeof(int64_t),
             cudaMemcpyHostToDevice);
}

template <typename T>
__device__ typename FlattenedJaggedVec<T>::DeviceAttr::Vec
FlattenedJaggedVec<T>::DeviceAttr::operator[](std::size_t idx) const {
  return FlattenedJaggedVec::DeviceAttr::Vec(&flattened_data[offsets[idx]],
                                             sizes[idx]);
}

template <typename T>
__device__ std::size_t FlattenedJaggedVec<T>::DeviceAttr::Vec::size() const {
  return size_;
}

template <typename T>
__device__ T
FlattenedJaggedVec<T>::DeviceAttr::Vec::operator[](std::size_t idx) const {
  return data_[idx];
}

#endif // FLATTENED_JAGGED_VEC_H_
