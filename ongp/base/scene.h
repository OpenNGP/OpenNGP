
#pragma once

#include <torch/torch.h>
#include "ongp/base/object.h"

namespace ongp
{
    class Scene: public Object
    {
    public:
        Scene() = default;

        void Add(ObjectSptr object);

        bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const;
    protected:
        std::vector<ObjectSptr> objects_;

    };
}