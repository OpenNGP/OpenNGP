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

protected:
    torch::Tensor center_;
    double radius_;
    std::shared_ptr<Material> mat_ptr_;
};

}