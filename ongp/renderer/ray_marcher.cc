#include "ongp/renderer/ray_marcher.h"

#include "ongp/renderer/dynamic_invoker.h"

namespace ongp {
namespace renderer {

std::unique_ptr<SampleResult> uniform_sampler(std::shared_ptr<Ray> ray,
                                              int N_samples, bool lindisp,
                                              bool perturb) {
  return SampleResult();
}

std::unique_ptr<SampleResult> importance_sampler(std::shared_ptr<Ray> ray,
                                                 int N_samples, bool lindisp,
                                                 bool perturb) {
  return SampleResult();
}

std::unique_ptr<SampleResult> SamplerFunction::call(
    std::shared_ptr<Ray> ray, std::shared_ptr<Primitive> primitive,
    std::shared_ptr<RenderPassResult> last_pass_ret,
    std::shared_ptr<Context> context) {
  auto method = context;
  // construct arg list
  std::list<boost::any> args;
  switch (method) {
    case "uniform_sampler":
      return dynamic_call(uniform_sampler, args);
    case "importance_sampler":
      return dynamic_call(importance_sampler, args);
    default:
      return SampleResult();
  }
}
}  // namespace renderer
}  // namespace ongp