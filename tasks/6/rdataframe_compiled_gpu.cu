#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
#include "TH1F.h"
#include "device.h"
#include "flattened_jagged_vec.h"
#include "util.h"
#include <TFile.h>
#include <TTree.h>
#include <iostream>
#include <vector>

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

  // transformed attributes to be used on the GPU
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
  AnalysisKernel<<<num_threads_per_block_, num_blocks>>>(
      nJets.size(), device_nJets, flattened_Jet_pts.GetDeviceAttr(),
      flattened_Jet_etas.GetDeviceAttr(), flattened_Jet_phis.GetDeviceAttr(),
      flattened_Jet_masses.GetDeviceAttr(), device_Jet_xyzts.GetDeviceAttr(),
      device_trijet_pt_bins);
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

int main() {
  AnalysisWorkflow workflow("../../../data/Run2012B_SingleMu.root");
  workflow.Run();
  return 0;
}
