#include "ongp/base/aabb.h"
#include "ongp/base/object.h"

namespace ongp
{
    bool box_compare(const ObjectSptr a, const ObjectSptr b, int axis) {
        AABB box_a;
        AABB box_b;

        if (!a->BoundingBox(box_a) || !b->BoundingBox(box_b))
            std::cerr << "No bounding box in bvh_node constructor.\n";

        return box_a.min()[axis].item<float>() < box_b.min()[axis].item<float>();
    }


    bool box_x_compare (const ObjectSptr a, const ObjectSptr b) {
        return box_compare(a, b, 0);
    }

    bool box_y_compare (const ObjectSptr a, const ObjectSptr b) {
        return box_compare(a, b, 1);
    }

    bool box_z_compare (const ObjectSptr a, const ObjectSptr b) {
        return box_compare(a, b, 2);
    }
}