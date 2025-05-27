#include "DeviceLorentzVector.h"
#include "DevicePtEtaPhiMVector.h"
#include "DevicePxPyPzE4D.h"
#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1F.h"
#include "flattened_jagged_vec.h"
#include "util.h"
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <vector>

typedef DeviceLorentzVector<DevicePxPyPzE4D<double>> DeviceXYZTVector;

__global__ void AnalysisKernel();

class AnalysisWorkflow {
public:
  AnalysisWorkflow(const std::string &filename);
  ~AnalysisWorkflow();
  void Run();

private:
  // host attributes
  std::string filename_;
  std::vector<UInt_t> nJets;
  JaggedVec<float> Jet_pts;
  JaggedVec<float> Jet_etas;
  JaggedVec<float> Jet_phis;
  JaggedVec<float> Jet_masses;
  std::vector<float> host_trijet_pt_bins_;

  // tranformed attributes to be used on the GPU
  // FlattendJaggedVec contains host and device members
  FlattenedJaggedVec<float> flattened_Jet_pts;
  FlattenedJaggedVec<float> flattened_Jet_etas;
  FlattenedJaggedVec<float> flattened_Jet_phis;
  FlattenedJaggedVec<float> flattened_Jet_masses;

  // device attribute without transformation
  UInt_t *device_nJets = nullptr;
  float *device_trijet_pt_bins = nullptr;

  // attribute only needed on the GPU -> no host equivalent
  FlattenedJaggedVec<DeviceXYZTVector> device_Jet_xyzts;

  int num_threads_per_block_ = 128;
  int num_histogram_bins_ = 100;

  void LoadAndFilterData();
  void FlattenJaggedAttributes();
  void CopyToDevice();
  void RunAnalysis();
  void CopyToHost();
  void GeneratePlots();
};

AnalysisWorkflow::AnalysisWorkflow(const std::string &filename)
    : filename_(filename) {}

AnalysisWorkflow::~AnalysisWorkflow() {
  if (device_nJets) {
    cudaFree(device_nJets);
    device_nJets = nullptr;
  }
}

void AnalysisWorkflow::Run() {
  LoadAndFilterData();
  FlattenJaggedAttributes();
  CopyToDevice();
  RunAnalysis();
  CopyToHost();
  GeneratePlots();
}

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

  // initialize histrogram bins
  host_trijet_pt_bins_.resize(num_histogram_bins_ + 2); // +2 for over/underflow
}

void AnalysisWorkflow::FlattenJaggedAttributes() {
  flattened_Jet_pts = Jet_pts;
  flattened_Jet_etas = Jet_etas;
  flattened_Jet_phis = Jet_phis;
  flattened_Jet_masses = Jet_masses;
}

void AnalysisWorkflow::CopyToDevice() {
  cudaMallocAndCopy(nJets, device_nJets);
  cudaMallocAndCopy(host_trijet_pt_bins_, device_trijet_pt_bins);

  flattened_Jet_pts.CopyToDevice();
  flattened_Jet_etas.CopyToDevice();
  flattened_Jet_phis.CopyToDevice();
  flattened_Jet_masses.CopyToDevice();

  device_Jet_xyzts.ReserveDataAndCopySizesAndOffsetsToDevice(flattened_Jet_pts);
}

void AnalysisWorkflow::RunAnalysis() {
  int num_blocks =
      (nJets.size() + num_threads_per_block_ - 1) / num_threads_per_block_;
  AnalysisKernel<<<num_threads_per_block_, 1>>>();
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream s;
    s << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error(s.str());
  }
}

void AnalysisWorkflow::CopyToHost() {
  // copy histogram bins back to host
  cudaMallocAndCopy(host_trijet_pt_bins_, device_trijet_pt_bins);
}

void AnalysisWorkflow::GeneratePlots() {
  TH1F h1("", ";Trijet pt (GeV);N_{Events}", /*nbins*/ 100, /*xin*/ 15,
          /*xmax*/ 40);
  // int nbins = binContents.size();
  // double xmin = 0.0;
  // double xmax = static_cast<double>(nbins);

  // Set bin contents (ROOT bins are 1-indexed!)
  for (int i = 0; i < h1.GetNbinsX(); ++i) {
    h1.SetBinContent(i, host_trijet_pt_bins_[i]);
  }
  TCanvas c;
  // c.Divide(2, 1);
  // c.cd(1);
  h1.Draw();
  // c.cd(2);
  // h2->Draw();
  c.SaveAs("6_rdataframe_compiled.png");
}

__global__ void AnalysisKernel() {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  // if (idx >= aw.nJets.size()) {
  // return;
  //}

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

int main() {
  AnalysisWorkflow workflow("../../../data/Run2012B_SingleMu.root");
  workflow.Run();
  return 0;
}
