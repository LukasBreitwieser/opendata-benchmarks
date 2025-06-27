
#ifndef DEVICE_H_
#define DEVICE_H_

#include "DeviceLorentzVector.h"
#include "DevicePtEtaPhiMVector.h"
#include "DevicePxPyPzE4D.h"
#include "flattened_jagged_vec.h"
#include "Histogram.h"
#include <cmath>
#include <cstddef>

typedef DeviceLorentzVector<DevicePxPyPzE4D<double>> DeviceXYZTVector;
typedef DeviceLorentzVector<DevicePtEtaPhiM4D<double> > DevicePtEtaPhiMVector;
using DeviceAttr = FlattenedJaggedVec<float>::DeviceAttr;
using Vec = DeviceAttr::Vec;

__device__ void
find_trijet(FlattenedJaggedVec<DeviceXYZTVector>::DeviceAttr::Vec jets,
            std::size_t *Trijet_index) {
  constexpr std::size_t n = 3;
  float distance = 1e9;
  const auto top_mass = 172.5;

  for (std::size_t i = 0; i <= jets.size() - n; i++) {
    auto p1 = jets[i];
    for (std::size_t j = i + 1; j <= jets.size() - n + 1; j++) {
      auto p2 = jets[j];
      for (std::size_t k = j + 1; k <= jets.size() - n + 2; k++) {
        auto p3 = jets[k];
        const auto tmp_mass = (p1 + p2 + p3).mass();
        const auto tmp_distance = std::abs(tmp_mass - top_mass);
        if (tmp_distance < distance) {
          distance = tmp_distance;
          Trijet_index[0] = i;
          Trijet_index[1] = j;
          Trijet_index[2] = k;
        }
      }
    }
  }
}

__device__ float trijet_pt(Vec pt, Vec eta, Vec phi,
                           Vec mass, std::size_t* idx) {
  auto p1 = DevicePtEtaPhiMVector(pt[idx[0]], eta[idx[0]], phi[idx[0]],
                                        mass[idx[0]]);
  auto p2 = DevicePtEtaPhiMVector(pt[idx[1]], eta[idx[1]], phi[idx[1]],
                                        mass[idx[1]]);
  auto p3 = DevicePtEtaPhiMVector(pt[idx[2]], eta[idx[2]], phi[idx[2]],
                                        mass[idx[2]]);
  return (p1 + p2 + p3).pt();
}

__device__ float TakeMax(Vec v, std::size_t *indices, std::size_t indices_size) {
  float max = v[indices[0]];
  for (int i = 1; i < indices_size; i++) {
    auto val = v[indices[i]];
    if (val > max) {
      max = val;
    }
  }
  return max;
}

__global__ void
AnalysisKernel(uint64_t num_events, UInt_t *nJets, DeviceAttr Jet_pts,
               DeviceAttr Jet_etas, DeviceAttr Jet_phis, DeviceAttr Jet_masses,
               DeviceAttr Jet_btags,
               FlattenedJaggedVec<DeviceXYZTVector>::DeviceAttr Jet_xyzts,
               Histogram::DeviceHistogram trijet_pt_histogram, 
               Histogram::DeviceHistogram trijet_btag_histogram) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_events) {
    return;
  }

  auto JetXYZT = Jet_xyzts[idx];
  for (int i = 0; i < nJets[idx]; i++) {
    DevicePtEtaPhiM4D ptEtaPhiMVector(Jet_pts[idx][i], Jet_etas[idx][i],
                                      Jet_phis[idx][i], Jet_masses[idx][i]);
    JetXYZT[i] = DeviceXYZTVector(ptEtaPhiMVector); // TODO avoid copy
  }
  std::size_t Trijet_idx[3];
  find_trijet(JetXYZT, Trijet_idx);
  //printf("Thread index: %d, : Trijet_idx: %f %f %f\n", idx, Trijet_idx[0], Trijet_idx[1], Trijet_idx[2]);
  //return;
  float Trijet_pt = trijet_pt(Jet_pts[idx], Jet_etas[idx], Jet_phis[idx], Jet_masses[idx], Trijet_idx);
  trijet_pt_histogram.Fill(Trijet_pt);

  auto Trijet_leadingBtag = TakeMax(Jet_btags[idx], Trijet_idx, 3);
  trijet_btag_histogram.Fill(Trijet_leadingBtag);
}

#endif // DEVICE_H_
