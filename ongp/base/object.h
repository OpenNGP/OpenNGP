
#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"

namespace ongp
{
    class Object
    {
        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const = 0;
    };
}