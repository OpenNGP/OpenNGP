#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/object.h"
#include "ongp/renderer/texture.h"

namespace ongp
{
class Material;

class ConstantMedium: public Object
{
    public:
        ConstantMedium(std::shared_ptr<Object> b, double d, std::shared_ptr<Texture> a)
            : boundary(b),
              neg_inv_density(-1/d),
              phase_function(std::make_shared<Isotropic>(a))
            {}

        ConstantMedium(std::shared_ptr<Object> b, double d, torch::Tensor c)
            : boundary(b),
              neg_inv_density(-1/d),
              phase_function(std::make_shared<Isotropic>(c))
            {}


        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const override;

        virtual bool BoundingBox(AABB& output_box) const override {
            return boundary->BoundingBox(output_box);
        }

    public:
        std::shared_ptr<Object> boundary;
        std::shared_ptr<Material> phase_function;
        double neg_inv_density;
};
}