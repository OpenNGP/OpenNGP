#pragma once

#include "ongp/base/ray.h"
#include "ongp/base/scene.h"

namespace ongp
{
    class Material {
    public:
        virtual bool Scatter(
        const Ray& r_in, const RayHit& rec, torch::Tensor& attenuation, Ray& scattered
        ) const = 0;
        virtual torch::Tensor emitted(double u, double v, const torch::Tensor& p) const {
            return Vector3({0,0,0});
        }
    };


    inline torch::Tensor ray_color(const Ray& r, const Scene& scene, int depth) {
        RayHit hit;
//        DELOG(depth);
        if (depth <= 0)
        {
          return Array1dToTensor<float>({0,0,0});
        }

        if (scene.Hit(r, 0, std::numeric_limits<double>::max(), hit)) {
       //     std::cout << -hit.normal << std::endl;
       //     std::cout << 0.5 * (-hit.normal + Array1dToTensor<float>({1,1,1})) << std::endl;
       //     PAUSE();

 //           std::cout << "n:" << hit.normal << std::endl;
 //           std::cout << "p:" << hit.point << std::endl;
 //           std::cout << "t:" << hit.t << std::endl;
 //           std::cout << "f:" << hit.front_face << std::endl;
 //           PAUSE();

            Ray scattered;
            torch::Tensor attenuation;
            if (hit.mat_ptr->Scatter(r, hit, attenuation, scattered))
                return attenuation * ray_color(scattered, scene, depth-1);
            return Array1dToTensor<float>({0,0,0});
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
}

