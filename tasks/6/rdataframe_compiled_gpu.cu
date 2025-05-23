#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <vector>

template <typename T> using RVec = ROOT::RVec<T>;

class AnalysisWorkflow {
public:
  AnalysisWorkflow(const std::string &filename);
  void Run();

private:
  std::string filename_;
  std::vector<UInt_t> nJets = {};
  std::vector<RVec<float>> Jet_pts = {};
  std::vector<RVec<float>> Jet_etas = {};
  std::vector<RVec<float>> Jet_phis = {};
  std::vector<RVec<float>> Jet_masses = {};

  void LoadAndFilterData();
  void CopyHostToDevice();
  void RunAnalysis();
  void CopyDeviceToHost();
};

#ifdef FOO
template <typename T> using Vec = const ROOT::RVec<T> &;
using ROOT::Math::XYZTVector;

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

AnalysisWorkflow::AnalysisWorkflow(const std::string &filename)
    : filename_(filename) {}

void AnalysisWorkflow::Run() { LoadAndFilterData(); }

void AnalysisWorkflow::LoadAndFilterData() {
  ROOT::EnableImplicitMT(); // Optional: Enable multi-threading
  std::string treename = "Events";
  ROOT::RDataFrame df(treename, filename_);
  auto df2 = df.Filter([](unsigned int n) { return n >= 3; }, {"nJet"},
                       "At least three jets");
  nJets = df2.Take<UInt_t>("nJet").GetValue();
  Jet_pts = df2.Take<ROOT::RVec<Float_t>>("Jet_pt").GetValue();
  Jet_etas = df2.Take<ROOT::RVec<Float_t>>("Jet_eta").GetValue();
  Jet_phis = df2.Take<ROOT::RVec<Float_t>>("Jet_phi").GetValue();
  Jet_masses = df2.Take<ROOT::RVec<Float_t>>("Jet_mass").GetValue();
}

//__global__ void GPUAnalysisKernel() {
// auto JetXYZT = Construct<XYZTVector>(
// Construct<PtEtaPhiMVector>(Jet_pt, Jet_eta, Jet_phi, Jet_mass));
// auto Trijet_idx = find_trijet(JetXYZT);

// auto Trijet_pt = trijet_pt(Jet_pt, Jet_eta, Jet_mass, Trijet_idx);
// auto Trijet_leadingBtag = Max(Take(Jet_btag, Trijet_idx));
//}

int main() {
  AnalysisWorkflow workflow("../../../data/Run2012B_SingleMu.root");
  workflow.Run();
  return 0;
}
