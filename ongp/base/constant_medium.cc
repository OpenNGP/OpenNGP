
#include "ongp/base/constant_medium.h"

namespace ongp
{
bool ConstantMedium::Hit(const Ray& r, double t_min, double t_max, RayHit& rec) const {
    // Print occasional samples when debugging. To enable, set enableDebug true.
    const bool enableDebug = false;
    const bool debugging = enableDebug && random_double() < 0.00001;

    RayHit rec1, rec2;

    if (!boundary->Hit(r, -infinity, infinity, rec1))
        return false;

    if (!boundary->Hit(r, rec1.t+0.0001, infinity, rec2))
        return false;

    if (debugging) std::cerr << "\nt_min=" << rec1.t << ", t_max=" << rec2.t << '\n';

    if (rec1.t < t_min) rec1.t = t_min;
    if (rec2.t > t_max) rec2.t = t_max;

    if (rec1.t >= rec2.t)
        return false;

    if (rec1.t < 0)
        rec1.t = 0;

    const auto ray_length = r.direction().norm().item<float>();
    const auto distance_inside_boundary = (rec2.t - rec1.t) * ray_length;
    const auto hit_distance = neg_inv_density * log(random_double());

    if (hit_distance > distance_inside_boundary)
        return false;

    rec.t = rec1.t + hit_distance / ray_length;
    rec.point = r.At(rec.t);

    if (debugging) {
        std::cerr << "hit_distance = " <<  hit_distance << '\n'
                  << "rec.t = " <<  rec.t << '\n'
                  << "rec.p = " <<  rec.point << '\n';
    }

    rec.normal = Vector3({1,0,0});  // arbitrary
    rec.front_face = true;     // also arbitrary
    rec.mat_ptr = phase_function;

    return true;
}
}