
#pragma once

#include <torch/torch.h>
#include "ongp/base/object.h"

namespace ongp
{
    class Scene
    {
    public:
        Scene() = default;

        void Add(Object* object);
    protected:
        std::shared_ptr<Object> objects_;

    };
}