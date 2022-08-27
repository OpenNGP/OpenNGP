
#include "ongp/base/rect_aa.h"

namespace ongp
{

bool RectXY::Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const {
    auto t = (k-r.origin()[2].item<float>()) / r.direction()[2].item<float>();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin()[0].item<float>() + t*r.direction()[0].item<float>();
    auto y = r.origin()[1].item<float>() + t*r.direction()[1].item<float>();
    if (x < x0 || x > x1 || y < y0 || y > y1)
        return false;
    rec.u = (x-x0)/(x1-x0);
    rec.v = (y-y0)/(y1-y0);
    rec.t = t;
    auto outward_normal = Vector3({0, 0, 1});
    rec.SetFaceNormal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.point = r.At(t);
    return true;
}

bool RectXZ::Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const {
    auto t = (k-r.origin()[1].item<float>()) / r.direction()[1].item<float>();
    if (t < t_min || t > t_max)
        return false;
    auto x = r.origin()[0].item<float>() + t*r.direction()[0].item<float>();
    auto z = r.origin()[2].item<float>() + t*r.direction()[2].item<float>();
    if (x < x0 || x > x1 || z < z0 || z > z1)
        return false;
    rec.u = (x-x0)/(x1-x0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;
    auto outward_normal = Vector3({0, 0, 1});
    rec.SetFaceNormal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.point = r.At(t);
    return true;
}

bool RectYZ::Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const {
    auto t = (k-r.origin()[0].item<float>()) / r.direction()[0].item<float>();
    if (t < t_min || t > t_max)
        return false;
    auto y = r.origin()[1].item<float>() + t*r.direction()[1].item<float>();
    auto z = r.origin()[2].item<float>() + t*r.direction()[2].item<float>();
    if (y < y0 || y > y1 || z < z0 || z > z1)
        return false;
    rec.u = (y-y0)/(y1-y0);
    rec.v = (z-z0)/(z1-z0);
    rec.t = t;
    auto outward_normal = Vector3({0, 0, 1});
    rec.SetFaceNormal(r, outward_normal);
    rec.mat_ptr = mp;
    rec.point = r.At(t);
    return true;
}

}