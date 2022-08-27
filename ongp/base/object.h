
#pragma once

#include <torch/torch.h>

namespace ongp
{
    class Ray;
    class RayHit;
    class AABB;

    class Object
    {
    public:
        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const = 0;
        virtual bool OnSurface(const torch::Tensor& point) const = 0;
        virtual bool BoundingBox(AABB& output_box) const = 0;

    };

    using ObjectSptr = std::shared_ptr<Object>;
}