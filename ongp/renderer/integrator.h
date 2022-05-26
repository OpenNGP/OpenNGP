#pragma one

#include <torch/torch.h>

namespace ongp {
namespace renderer {
torch::Tensor integrate_weight(const torch::Tensor& sigmas,
                               const torch::Tensor& deltas) {
  auto alphas = 1. - torch::exp(-sigmas * deltas);
  auto ones = torch::ones((alphas.size(0), 1));
  auto weights =
      alphas * torch::cumprod(torch::cat({ones, 1. - alphas + 1e-10}, -1), -1)
                   .index({"...", -1});
  return weights;
}

torch::Tensor volume_integrator(const torch::Tensor& sigmas,
                                const torch::Tensor& rgbs,
                                const torch::Tensor& deltas,
                                const torch::Tensor& z_vals) {
  auto weights = integrate_weight(sigmas, deltas);
  auto colors =
      torch::sum(weights.index({"...", torch::indexing::None}) * rgbs, -2);
  auto depths = torch::sum(weights * z_vals, -1);
  return weights;
}

torch::Tensor depth_integrator(const torch::Tensor& sigmas,
                               const torch::Tensor& deltas,
                               const torch::Tensor& z_vals) {
  auto weights = integrate_weight(sigmas, deltas);
  auto depths = torch::sum(weights * z_vals, -1);
  return weights;
}
class Integrator {
 public:
  Integrator() = default;
};
}  // namespace renderer
}  // namespace ongp
