#pragma one

#include <torch/torch.h>

#include <memory>

#include "ongp/base/ray.h"
#include "ongp/primitive/primitive.h"
#include "ongp/renderer/render_defs.h"

namespace ongp {
namespace renderer {
class RayMarcher {
 public:
  std::unique_ptr<SampleResult> call(
      std::shared_ptr<Ray> ray, std::shared_ptr<Primitive> primitive,
      std::shared_ptr<RenderPassResult> last_pass_ret,
      std::shared_ptr<Context> context);
};
}  // namespace renderer
}  // namespace ongp
