#include <torch/torch.h>

namespace ongp {
namespace renderer {
struct SampleResult {};
struct Pixel {};
struct RenderPassResult {
  std::string name;
  torch::Tensor sigmas;
  torch::Tensor geo_features;
  torch::Tensor rgbs;
  torch::Tensor weights;
  SampleResult samples;
  Pixel pixels;
};
class Context {};
}  // namespace renderer

}  // namespace ongp