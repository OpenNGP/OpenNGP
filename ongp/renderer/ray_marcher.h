#pragma one

#include <torch/torch.h>

#include <memory>

#include "ongp/base/ray.h"
#include "ongp/primitive/primitive.h"
#include "ongp/renderer/render_defs.h"

namespace ongp {

class RayMarcher {
 public:
  virtual std::unique_ptr<renderer::SampleResult> call(
      std::shared_ptr<Ray> ray, std::shared_ptr<Primitive> primitive,
      std::shared_ptr<renderer::RenderPassResult> last_pass_ret,
      std::shared_ptr<renderer::Context> context);
};
}  // namespace ongp
