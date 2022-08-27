#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/object.h"

namespace ongp
{
class Material;

class Sphere: public Object
{
public:
    Sphere() = default;
    Sphere(torch::Tensor center, double r, std::shared_ptr<Material> mat_ptr= nullptr);

    bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const;
    bool OnSurface(const torch::Tensor& point) const;
    bool BoundingBox(AABB& output_box) const;

    static void get_sphere_uv(const torch::Tensor& p, double& u, double& v) {
            // p: a given point on the sphere of radius one, centered at the origin.
            // u: returned value [0,1] of angle around the Y axis from X=-1.
            // v: returned value [0,1] of angle from Y=-1 to Y=+1.
            //     <1 0 0> yields <0.50 0.50>       <-1  0  0> yields <0.00 0.50>
            //     <0 1 0> yields <0.50 1.00>       < 0 -1  0> yields <0.50 0.00>
            //     <0 0 1> yields <0.25 0.50>       < 0  0 -1> yields <0.75 0.50>

            auto theta = acos(-p[1]).item<float>();
            auto phi = atan2(-p[2], p[0]).item<float>() + M_PI;

            u = phi / (2*M_PI);
            v = theta / M_PI;
        }

protected:
    torch::Tensor center_;
    double radius_;
    std::shared_ptr<Material> mat_ptr_;
};

}