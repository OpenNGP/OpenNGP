
#include "ongp/base/scene.h"

namespace ongp
{
    void Scene::Add(Object* object)
    {
        objects_.push_back(object);
    }
}