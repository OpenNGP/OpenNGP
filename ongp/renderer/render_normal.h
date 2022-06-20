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

    void write_color(std::ostream& out, torch::Tensor pixel_color, int samples_per_pixel)
    {
        auto r = pixel_color.index({0}).item<double>();
        auto g = pixel_color.index({1}).item<double>();
        auto b = pixel_color.index({2}).item<double>();
        auto scale = 1.0 / samples_per_pixel;
        r *= scale;
        g *= scale;
        b *= scale;
        // Write the translated [0,255] value of each color component.
        out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';

    }

}

