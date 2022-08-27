
#pragma once

#include <torch/torch.h>
#include "ongp/base/object.h"
#include "ongp/base/ray.h"

namespace ongp
{
    class Scene: public Object
    {
    public:
        Scene() = default;

        void Add(ObjectSptr object);

        bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const;
        bool OnSurface(const torch::Tensor& point) const;
        bool BoundingBox(AABB& output_box) const;

        std::vector<ObjectSptr> Objects() const { return objects_; }
        
    protected:
        std::vector<ObjectSptr> objects_;

    };
}