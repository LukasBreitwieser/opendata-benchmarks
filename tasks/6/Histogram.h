#include "TH1F.h"
#include "util.h"

#ifndef DEVICE_HISTOGRAM_H_
#define DEVICE_HISTOGRAM_H_

class Histogram {
public:
  Histogram(const char *name, const char *title, Int_t nbinsx, float xlow,
            float xup);

  struct DeviceHistogram {
    int num_bins_;
    float low_;
    float high_;
    float *bins_ = nullptr;

    void Fill(float value);
  };

  DeviceHistogram GetDeviceHistogram();
  void CopyToDevice();
  void CopyToHost();

  TH1F &GetTH1F();

private:
  TH1F h_;
  std::vector<float> host_bins_;
  DeviceHistogram device_h_;
};

Histogram::Histogram(const char *name, const char *title, Int_t nbinsx,
                     float xlow, float xup)
    : h_(name, title, nbinsx, xlow, xup), host_bins_(nbinsx + 2),
      device_h_{nbinsx, xlow, xup} {
  host_bins_.resize(nbinsx + 2);
}

void Histogram::CopyToDevice() {
  cudaMallocAndCopy(host_bins_, &device_h_.bins_);
}

void Histogram::CopyToHost() {
  cudaCopy2HostAndFree(host_bins_, device_h_.bins_);
  device_h_.bins_ = nullptr;
  // Set bin contents (ROOT bins are 1-indexed!)
  for (int i = 0; i < h_.GetNbinsX(); ++i) {
    std::cout << "host_bins_[i] " << host_bins_[i] << std::endl;
    h_.SetBinContent(i, host_bins_[i]);
  }
}

__device__ void Histogram::DeviceHistogram::Fill(float value) {
  int bin_idx = 0;
  if (value > high_) {
    bin_idx = num_bins_ + 1;
  } else if (value > low_) {
    auto bin_width = (high_ - low_) / num_bins_;
    bin_idx = (value - low_) / bin_width;
  }
  // printf("value %f, low %f, high %f, bin_idx %d - num_bins %d\n", value,
  // low_, high_, bin_idx, num_bins_);
  atomicAdd(&bins_[bin_idx], 1);
}

TH1F &Histogram::GetTH1F() { return h_; }

Histogram::DeviceHistogram Histogram::GetDeviceHistogram() { return device_h_; }

#endif // DEVICE_HISTOGRAM_H_
