#include "Histogram.h"
#include "Math/Vector4D.h"
#include "ROOT/RDataFrame.hxx"
#include "ROOT/RVec.hxx"
#include "TCanvas.h"
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
  JaggedVec<float> Jet_btags;

  // transformed attributes to be used on the GPU
  // FlattendJaggedVec contains host and device members
  FlattenedJaggedVec<float> flattened_Jet_pts;
  FlattenedJaggedVec<float> flattened_Jet_etas;
  FlattenedJaggedVec<float> flattened_Jet_phis;
  FlattenedJaggedVec<float> flattened_Jet_masses;
  FlattenedJaggedVec<float> flattened_Jet_btags;

  // device attribute without transformation
  UInt_t *device_nJets = nullptr;

  // attribute only needed on the GPU -> no host equivalent
  FlattenedJaggedVec<DeviceXYZTVector> device_Jet_xyzts;

  int num_threads_per_block_ = 128;
  int num_histogram_bins_ = 100;
  Histogram trijet_pt_histogram_;
  Histogram trijet_leading_btag_histogram_;

  void LoadAndFilterData();
  void FlattenJaggedAttributes();
  void CopyToDevice();
  void RunAnalysis();
  void CopyToHost();
  void GeneratePlots();
};

AnalysisWorkflow::AnalysisWorkflow(const std::string &filename)
    : filename_(filename),
      trijet_pt_histogram_("", ";Trijet pt (GeV);N_{Events}", /*[>nbins*/ 100,
                           /*xin<]*/ 15,
                           /*xmax*/ 40),
      trijet_leading_btag_histogram_("", ";Trijet leading b-tag;N_{Events}",
                                     /*[>nbins*/ 100,
                                     /*xin<]*/ 0,
                                     /*xmax*/ 1) {}

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
  // ROOT::EnableImplicitMT(); // Optional: Enable multi-threading
  std::string treename = "Events";
  ROOT::RDataFrame df(treename, filename_);
  auto df2 = df.Filter([](unsigned int n) { return n >= 3; }, {"nJet"},
                       "At least three jets");
  nJets = df2.Take<UInt_t>("nJet").GetValue();
  Jet_pts = df2.Take<ROOT::RVec<Float_t>>("Jet_pt").GetValue();
  Jet_etas = df2.Take<ROOT::RVec<Float_t>>("Jet_eta").GetValue();
  Jet_phis = df2.Take<ROOT::RVec<Float_t>>("Jet_phi").GetValue();
  Jet_masses = df2.Take<ROOT::RVec<Float_t>>("Jet_mass").GetValue();
  Jet_btags = df2.Take<ROOT::RVec<Float_t>>("Jet_btag").GetValue();
}

void AnalysisWorkflow::FlattenJaggedAttributes() {
  flattened_Jet_pts = Jet_pts;
  flattened_Jet_etas = Jet_etas;
  flattened_Jet_phis = Jet_phis;
  flattened_Jet_masses = Jet_masses;
  flattened_Jet_btags = Jet_btags;
}

void AnalysisWorkflow::CopyToDevice() {
  cudaMallocAndCopy(nJets, &device_nJets);

  flattened_Jet_pts.CopyToDevice();
  flattened_Jet_etas.CopyToDevice();
  flattened_Jet_phis.CopyToDevice();
  flattened_Jet_masses.CopyToDevice();
  flattened_Jet_btags.CopyToDevice();

  device_Jet_xyzts.ReserveDataAndCopySizesAndOffsetsToDevice(flattened_Jet_pts);

  trijet_pt_histogram_.CopyToDevice();
  trijet_leading_btag_histogram_.CopyToDevice();
}

// For debugging purposes
__global__ void PrintBins(Histogram::DeviceHistogram trijet_pt_histogram) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= 1) {
    return;
  }
  for (int i = 0; i < trijet_pt_histogram.num_bins_; ++i) {
    printf("Bin %d: %f\n", i, trijet_pt_histogram.bins_[i]);
  }
}

void AnalysisWorkflow::RunAnalysis() {
  int num_blocks =
      (nJets.size() + num_threads_per_block_ - 1) / num_threads_per_block_;
  std::cout << "Num jets: " << nJets.size() << std::endl;
  std::cout << "Num threads: " << num_threads_per_block_ << std::endl;
  std::cout << "Num blocks: " << num_blocks << std::endl;

  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);

  std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
  std::cout << "Max block dim (x, y, z): " << prop.maxThreadsDim[0] << ", "
            << prop.maxThreadsDim[1] << ", " << prop.maxThreadsDim[2] << "\n";
  std::cout << "Max grid dim (x, y, z): " << prop.maxGridSize[0] << ", "
            << prop.maxGridSize[1] << ", " << prop.maxGridSize[2] << "\n";

  AnalysisKernel<<<num_blocks, num_threads_per_block_>>>(
      nJets.size(), device_nJets, flattened_Jet_pts.GetDeviceAttr(),
      flattened_Jet_etas.GetDeviceAttr(), flattened_Jet_phis.GetDeviceAttr(),
      flattened_Jet_masses.GetDeviceAttr(), flattened_Jet_btags.GetDeviceAttr(),
      device_Jet_xyzts.GetDeviceAttr(),
      trijet_pt_histogram_.GetDeviceHistogram(),
      trijet_leading_btag_histogram_.GetDeviceHistogram());
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::stringstream s;
    s << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
    throw std::runtime_error(s.str());
  }
  PrintBins<<<1, 1>>>(trijet_pt_histogram_.GetDeviceHistogram());
}

void AnalysisWorkflow::CopyToHost() {
  // copy histogram bins back to host
  trijet_pt_histogram_.CopyToHost();
  trijet_leading_btag_histogram_.CopyToHost();
}

void AnalysisWorkflow::GeneratePlots() {
  TCanvas c;
  c.Divide(2, 1);
  c.cd(1);
  trijet_pt_histogram_.GetTH1F().Draw();
  c.cd(2);
  trijet_leading_btag_histogram_.GetTH1F().Draw();
  c.SaveAs("6_rdataframe_compiled_gpu.pdf");
}

int main() {
  AnalysisWorkflow workflow("../../../data/Run2012B_SingleMu.root");
  workflow.Run();
  return 0;
}
