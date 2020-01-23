// #include <torch/script.h> // One-stop header.
// #include <iostream>
// #include <memory>
// #include <std::string>
#include "DASHGAN-dylib.h"
std::string DASHGAN::Gnormal(void) {
  // if (argc != 2) {
  //   std::cerr << "usage: example-app <path-to-exported-script-module>\n";
  //   return void;
  // }

  std::stringstream out;
  torch::jit::script::Module module;
  // try {
  //   // Deserialize the ScriptModule from a file using torch::jit::load().
  //   module = torch::jit::load(argv[1]);
  // }
  // catch (const c10::Error& e) {

  //   std::cerr << "error loading the model\n";
  //   std::cerr << argv[1] << "\n";
  //   std::cerr << e.msg() << "\n";

  module = torch::jit::load("/Users/victorjupin/kode/DASHGAN_orig/netG_epoch_99.pt");

    // return -1;
  // }

  // std::cout << "Loaded ok\n";
  auto options =
    torch::TensorOptions()
      .dtype(torch::kFloat32);
  // Create a vector of inputs.
  std::vector<torch::jit::IValue> inputs;
  inputs.push_back(torch::normal(0, 1, {1, 32, 1, 1}, nullptr, options));

  // Execute the model and turn its output into a tensor.
  at::Tensor output = module.forward(inputs).toTensor();
  out << output << '\n';
  // for (auto num: output.data()) {
  //   out += std::string(num) + " ";
  // }
  return out.str();
}
