#pragma one

#include <torch/torch.h>

#include <memory>

#include "ongp/base/ray.h"
#include "ongp/primitive/primitive.h"
#include "ongp/renderer/integrator.h"
#include "ongp/renderer/ray_marcher.h"
#include "ongp/renderer/render_defs.h"

namespace ongp {
class RenerPass {
 public:
  explicit RenderPass(const JSON& config){};
  std::unique_ptr<renderer::RenderPassResult> render_pixel(
      std::shared_ptr<Ray> ray, std::shared_ptr<Primitive> primitive,
      std::shared_ptr<renderer::Context> context){
    auto sample_ret = this->sampler->call(ray, primitive, nullptr, context);
  };
  protected:
  std::unique_ptr<RayMarcher> sampler;
  std::unique_ptr<Integrator> integrator;
};
}  // namespace ongp