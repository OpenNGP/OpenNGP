
#include "ongp/base/bvh.h"

namespace ongp
{
    BVHNode::BVHNode(
        const std::vector<ObjectSptr>& src_objects,
        size_t start, size_t end
    ) {
        auto objects = src_objects; // Create a modifiable array of the source scene objects

        int axis = random_int(0,2);
        auto comparator = (axis == 0) ? box_x_compare
                        : (axis == 1) ? box_y_compare
                                    : box_z_compare;

        size_t object_span = end - start;

        if (object_span == 1) {
            left_ = right_ = objects[start];
        } else if (object_span == 2) {
            if (comparator(objects[start], objects[start+1])) {
                left_ = objects[start];
                right_ = objects[start+1];
            } else {
                left_ = objects[start+1];
                right_ = objects[start];
            }
        } else {
            std::sort(objects.begin() + start, objects.begin() + end, comparator);

            auto mid = start + object_span/2;
            left_ = std::make_shared<BVHNode>(objects, start, mid);
            right_ = std::make_shared<BVHNode>(objects, mid, end);
        }

        AABB box_left, box_right;

        if (  !left_->BoundingBox (box_left)
        || !right_->BoundingBox (box_right)
        )
            std::cerr << "No bounding box in bvh_node constructor.\n";

        aabb_ = AABB::surrounding_box(box_left, box_right);
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