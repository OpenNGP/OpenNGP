#pragma one

#include <torch/torch.h>
#include "ongp/base/object.h"
#include "ongp/base/ray.h"
#include "ongp/base/scene.h"

namespace ongp
{
    class BVHNode : public Object
    {
    public:
        BVHNode() = default;
        BVHNode(const Scene& list)
            : BVHNode(list.Objects(), 0, list.Objects().size())
        {}

        BVHNode(
            const std::vector<ObjectSptr>& src_objects,
            size_t start, size_t end);

        bool Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const;
        bool OnSurface(const torch::Tensor& point) const;
        bool BoundingBox(AABB& output_box) const;

        const ObjectSptr& Left() const { return left_; }
        const ObjectSptr& Right() const { return right_; }

    protected:
        ObjectSptr left_;
        ObjectSptr right_;
        AABB aabb_;



    };
}