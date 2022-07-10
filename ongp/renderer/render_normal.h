#pragma one

#include <torch/torch.h>
#include "delog/delog.h"

#include "ongp/base/ray.h"
#include "ongp/base/scene.h"

namespace ongp
{
    inline torch::Tensor ray_color_normal(const Ray& r, const Scene& scene) {
        RayHit hit;
        if (scene.Hit(r, 0, std::numeric_limits<double>::max(), hit)) {
       //     std::cout << -hit.normal << std::endl;
       //     std::cout << 0.5 * (-hit.normal + Array1dToTensor<float>({1,1,1})) << std::endl;
       //     PAUSE();
            return 0.5 * (-hit.normal + Array1dToTensor<float>({1,1,1}));
        }
        //auto unit_direction = r.direction().norm();
        //auto unit_direction = r.direction()[1].item<double>();
      //  std::cout << unit_direction.item<double>() << std::endl;
      //  PAUSE();
        auto t = 0.5*(r.direction()[1].item<float>() + 1.0);
     //   DELOG(t);
     //   std::cout << (1.0-t)*Array1dToTensor<float>({1.0,1.0,1.0}) << std::endl;
     //   std::cout << Array1dToTensor<float>({1,1,1})  << std::endl;
     //   PAUSE();
     //   std::cout << t*Array1dToTensor<float>({0.5,0.7,1}) << std::endl;
        return (1.0-t)*Array1dToTensor<float>({1,1,1}) + t*Array1dToTensor<float>({0.5,0.7,1});
    }

    void write_color(std::ostream& out, torch::Tensor pixel_color, int samples_per_pixel)
    {
        auto r = pixel_color.index({0}).item<double>();
        auto g = pixel_color.index({1}).item<double>();
        auto b = pixel_color.index({2}).item<double>();
      //  std::cout << pixel_color << std::endl;
      //  DELOG(r);
      //  DELOG(g);
      //  DELOG(b);

      // Divide the color by the number of samples and gamma-correct for gamma=2.0.
        auto scale = 1.0 / samples_per_pixel;
        r = sqrt(scale * r);
        g = sqrt(scale * g);
        b = sqrt(scale * b);
        // Write the translated [0,255] value of each color component.
        out << static_cast<int>(256 * clamp(r, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(g, 0.0, 0.999)) << ' '
            << static_cast<int>(256 * clamp(b, 0.0, 0.999)) << '\n';

    }

}

