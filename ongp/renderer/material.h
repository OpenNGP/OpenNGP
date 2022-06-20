#pragma once

#include "ongp/base/ray.h"

namespace ongp
{
    class material {
    public:
        virtual bool scatter(
        const Ray& r_in, const RayHit& rec, torch::Tensor& attenuation, Ray& scattered
        ) const = 0;
    };
}

