#include <torch/torch.h>
#include <iostream>

#include "ongp/dataset/frame_dataset.h"
#include "ongp/base/tensor.h"

int main() {
  std::vector<std::vector<float>> vec{
    {1,2,3,4},
    {1,2,3,4},
    {1,2,3,4},
    {1,2,3,4},
  };

  std::cout << ongp::Array2dToTensor(vec) << std::endl;

  torch::Tensor tensor = torch::rand({4, 3});
  std::cout << tensor[1][1] << std::endl;

  std::string lego_test_dir = "/nerf/openNGP/data/lego/";
  std::string lego_test_data = "transforms_test.json";

  ongp::FrameDataset fd(lego_test_dir, lego_test_data);
  std::cout << fd.size() << std::endl;
  fd.get(10);
//  for (size_t i = 0; i < fd.size(); ++ i) 
//  {
//    std::cout << fd.get(i).cam().pose().mat44() << std::endl;
//  }

}