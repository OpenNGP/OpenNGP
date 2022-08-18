
#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/object.h"
#include "ongp/base/aabb.h"
#include "ongp/base/scene.h"

namespace ongp
{
class Material;

class Box: public Object
{
    public:
        Box() {}
        Box(const torch::Tensor& p0, const torch::Tensor& p1, std::shared_ptr<Material> ptr);

        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const override;

        virtual bool BoundingBox(AABB& output_box) const override {
            output_box = AABB(box_min, box_max);
            return true;
        }

    public:
        torch::Tensor box_min;
        torch::Tensor box_max;
        Scene sides; // object list
};
}