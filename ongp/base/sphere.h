#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/object.h"

namespace ongp
{
class Sphere: public Object
{
public:
    Sphere() = default;
    Sphere(torch::Tensor center, double r);

    bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const;

protected:
    torch::Tensor center_;
    double radius_;
};

}