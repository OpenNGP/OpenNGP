
#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/aabb.h"

namespace ongp
{
    class Object
    {
    public:
        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const = 0;
        virtual bool OnSurface(const torch::Tensor& point) const = 0;
        virtual bool BoundingBox(AABB& output_box) const = 0;

    };

    using ObjectSptr = std::shared_ptr<Object>;
}