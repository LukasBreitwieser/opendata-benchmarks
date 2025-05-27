
#ifndef DEVICE_H_
#define DEVICE_H_

#include "flattened_jagged_vec.h"
#include "DeviceLorentzVector.h"
#include "DevicePtEtaPhiMVector.h"
#include "DevicePxPyPzE4D.h"

typedef DeviceLorentzVector<DevicePxPyPzE4D<double>> DeviceXYZTVector;
using DeviceAttr = FlattenedJaggedVec<float>::DeviceAttr;

__global__ void
AnalysisKernel(uint64_t num_events, UInt_t *nJets, DeviceAttr Jet_pts,
               DeviceAttr Jet_etas, DeviceAttr Jet_phis, DeviceAttr Jet_masses,
               FlattenedJaggedVec<DeviceXYZTVector>::DeviceAttr Jet_xyzts,
               float *trijet_pt_bins) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= num_events) {
    return;
  }

  for (int i = 0; i < nJets[idx]; i++) {
     DevicePtEtaPhiM4D ptEtaPhiMVector(Jet_pts[idx][i], Jet_etas[idx][i], Jet_phis[idx][i], Jet_masses[idx][i]);
     DeviceXYZTVector JetXYZT(ptEtaPhiMVector);
  }
  // auto JetXYZT = Construct<XYZTVector>(Construct<PtEtaPhiMVector>(pt, eta,
  // phi, m));}, Trijet_idx = find_trijet(JetXYZT); Trijet_pt = trijet_pt(pt,
  // eta, phi, m, Trijet_idx);
  //  histogram
  // atomicAdd(&trijet_pt_bins[bin_idx], 1);
}

#ifdef OFF
template <typename T> using Vec = const ROOT::RVec<T> &;
using ROOT::Math::XYZTVector;

XYZTVector typedef LorentzVector<PxPyPzE4D<double>> XYZTVector;
Construct PtEtaPhiMVector operator+ pt() Construct

    __device__ ROOT::RVec<std::size_t> find_trijet(Vec<XYZTVector> jets) {
  constexpr std::size_t n = 3;
  float distance = 1e9;
  const auto top_mass = 172.5;
  std::size_t idx1 = 0, idx2 = 1, idx3 = 2;

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
          idx1 = i;
          idx2 = j;
          idx3 = k;
        }
      }
    }
  }
  return {idx1, idx2, idx3};
}

__device__ float trijet_pt(Vec<float> pt, Vec<float> eta, Vec<float> phi,
                           Vec<float> mass, Vec<std::size_t> idx) {
  auto p1 = ROOT::Math::PtEtaPhiMVector(pt[idx[0]], eta[idx[0]], phi[idx[0]],
                                        mass[idx[0]]);
  auto p2 = ROOT::Math::PtEtaPhiMVector(pt[idx[1]], eta[idx[1]], phi[idx[1]],
                                        mass[idx[1]]);
  auto p3 = ROOT::Math::PtEtaPhiMVector(pt[idx[2]], eta[idx[2]], phi[idx[2]],
                                        mass[idx[2]]);
  return (p1 + p2 + p3).pt();
}
#endif

#endif // DEVICE_H_
