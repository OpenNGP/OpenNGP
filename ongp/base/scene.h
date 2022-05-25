
#pragma once

#include <torch/torch.h>
#include "ongp/base/object.h"

namespace ongp
{
    class Scene
    {
    public:
        Scene() = default;

        void Add(ObjectSptr object);
    protected:
        std::vector<ObjectSptr> objects_;

    };
}