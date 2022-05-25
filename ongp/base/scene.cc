
#include "ongp/base/scene.h"

namespace ongp
{
    void Scene::Add(ObjectSptr object)
    {
        objects_.push_back(object);
    }
}