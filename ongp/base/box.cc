
#include "ongp/base/box.h"
#include "ongp/base/rect_aa.h"

namespace ongp
{

Box::Box(const torch::Tensor& p0, const torch::Tensor& p1, std::shared_ptr<Material> ptr) {
    box_min = p0;
    box_max = p1;

    sides.Add(std::make_shared<RectXY>(p0[0].item<float>(), p1[0].item<float>(), p0[1].item<float>(), p1[1].item<float>(), p1[2].item<float>(), ptr));
    sides.Add(std::make_shared<RectXY>(p0[0].item<float>(), p1[0].item<float>(), p0[1].item<float>(), p1[1].item<float>(), p0[2].item<float>(), ptr));

    sides.Add(std::make_shared<RectXZ>(p0[0].item<float>(), p1[0].item<float>(), p0[2].item<float>(), p1[2].item<float>(), p1[1].item<float>(), ptr));
    sides.Add(std::make_shared<RectXZ>(p0[0].item<float>(), p1[0].item<float>(), p0[2].item<float>(), p1[2].item<float>(), p0[1].item<float>(), ptr));

    sides.Add(std::make_shared<RectYZ>(p0[1].item<float>(), p1[1].item<float>(), p0[2].item<float>(), p1[2].item<float>(), p1[0].item<float>(), ptr));
    sides.Add(std::make_shared<RectYZ>(p0[1].item<float>(), p1[1].item<float>(), p0[2].item<float>(), p1[2].item<float>(), p0[0].item<float>(), ptr));
}

bool Box::Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const {
    return sides.Hit(r, t_min, t_max, rec);
}
}