
#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/object.h"
#include "ongp/base/aabb.h"

namespace ongp
{
class Material;

class RectXY: public Object
{
    public:
        RectXY() {}

        RectXY(double _x0, double _x1, double _y0, double _y1, double _k, 
            std::shared_ptr<Material> mat)
            : x0(_x0), x1(_x1), y0(_y0), y1(_y1), k(_k), mp(mat) {};

        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const override;

        virtual bool BoundingBox(AABB& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Z
            // dimension a small amount.
            output_box = AABB(Vector3({x0,y0, k-0.0001}), Vector3({x1, y1, k+0.0001}));
            return true;
        }

    public:
        std::shared_ptr<Material> mp;
        double x0, x1, y0, y1, k;
};

class RectXZ : public Object{
    public:
        RectXZ() {}

        RectXZ(double _x0, double _x1, double _z0, double _z1, double _k,
            std::shared_ptr<Material> mat)
            : x0(_x0), x1(_x1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

        virtual bool Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const override;

        virtual bool BoundingBox(AABB& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the Y
            // dimension a small amount.
            output_box = AABB(Vector3({x0,k-0.0001,z0}), Vector3({x1, k+0.0001, z1}));
            return true;
        }

    public:
        std::shared_ptr<material> mp;
        double x0, x1, z0, z1, k;
};

class RectYZ : public Object{
    public:
        RectYZ() {}

        RectYZ(double _y0, double _y1, double _z0, double _z1, double _k,
            std::shared_ptr<Material> mat)
            : y0(_y0), y1(_y1), z0(_z0), z1(_z1), k(_k), mp(mat) {};

        virtual bool hit(const Ray& r, double t_min, double t_max, RayHit& rec) const override;

        virtual bool BoundingBox(AABB& output_box) const override {
            // The bounding box must have non-zero width in each dimension, so pad the X
            // dimension a small amount.
            output_box = AABB(Vector3({k-0.0001, y0, z0}), Vector3({k+0.0001, y1, z1}));
            return true;
        }

    public:
        std::shared_ptr<material> mp;
        double y0, y1, z0, z1, k;
};

}