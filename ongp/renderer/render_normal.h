#pragma one

#include <torch/torch.h>

#include "ongp/base/ray.h"
#include "ongp/base/scene.h"

namespace ongp
{
    torch::Tensor ray_color(const Ray& r, const Scene& scene) {
        RayHit hit;
        if (scene.Hit(r, 0, std::numeric_limits<double>::max(), hit)) {
            return 0.5 * (hit.normal + Array1dToTensor<double>({1,1,1}));
        }
        auto unit_direction = r.direction().norm();
        auto t = 0.5*(unit_direction[1] + 1.0);
        return (1.0-t)*Array1dToTensor<double>({1,1,1}) + t*Array1dToTensor<double>({0.5,0.7,1});
}

}

