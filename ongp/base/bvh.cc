
#include "ongp/base/bvh.h"

namespace ongp
{
    BVHNode::BVHNode(
        const std::vector<ObjectSptr>& src_objects,
        size_t start, size_t end, double time0, double time1
    ) {
        auto objects = src_objects; // Create a modifiable array of the source scene objects

        int axis = random_int(0,2);
        auto comparator = (axis == 0) ? box_x_compare
                        : (axis == 1) ? box_y_compare
                                    : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left = right = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start+1])) {
                left = objects[start];
                right = objects[start+1];
            } else {
                left = objects[start+1];
                right = objects[start];
            }
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            auto mid = start + object_span/2;
            left = make_shared<bvh_node>(objects, start, mid, time0, time1);
            right = make_shared<bvh_node>(objects, mid, end, time0, time1);
        }

        aabb box_left, box_right;

        if (  !left->bounding_box (time0, time1, box_left)
        || !right->bounding_box(time0, time1, box_right)
        )
            std::cerr << "No bounding box in bvh_node constructor.\n";

        box = surrounding_box(box_left, box_right);
    }


    bool BVHNode::BoundingBox(AABB& output_box) const {
        output_box = aabb_;
        return true;
    }

    bool BVHNode::Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const {
        if (!aabb_.Hit(r, t_min, t_max))
            return false;

        bool hit_left = left_->Hit(r, t_min, t_max, hit);
        bool hit_right = right_->Hit(r, t_min, hit_left ? hit.t : t_max, hit);

        return hit_left || hit_right;
    }

}