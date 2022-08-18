
#include "ongp/base/box.h"
#include "ongp/base/rect_aa.h"

namespace ongp
{

Box::Box(const torch::Tensor& p0, const torch::Tensor& p1, std::shared_ptr<Material> ptr) {
    box_min = p0;
    box_max = p1;

    sides.Add(std::make_shared<RectXY>(p0.x(), p1.x(), p0.y(), p1.y(), p1.z(), ptr));
    sides.Add(std::make_shared<RectXY>(p0.x(), p1.x(), p0.y(), p1.y(), p0.z(), ptr));

    sides.Add(std::make_shared<RectXZ>(p0.x(), p1.x(), p0.z(), p1.z(), p1.y(), ptr));
    sides.Add(std::make_shared<RectXZ>(p0.x(), p1.x(), p0.z(), p1.z(), p0.y(), ptr));

    sides.Add(std::make_shared<RectYZ>(p0.y(), p1.y(), p0.z(), p1.z(), p1.x(), ptr));
    sides.Add(std::make_shared<RectYZ>(p0.y(), p1.y(), p0.z(), p1.z(), p0.x(), ptr));
}

bool Box::Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const {
    return sides.Hit(r, t_min, t_max, rec);
}
}