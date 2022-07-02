#include "delog/delog.h"
#include "ongp/base/sphere.h"

namespace ongp
{
    Sphere::Sphere(torch::Tensor center, double r):
    center_(center), radius_(r)
    {}

    bool Sphere::Hit(const Ray& r, double t_min, double t_max, RayHit& hit) const {
 //       std::cout << r.origin().sizes() << std::endl;
 //       std::cout << center_.sizes() << std::endl;
 //       PAUSE();
        auto oc = r.origin() - center_;
        auto a = r.direction().square().sum();
        auto half_b = torch::dot(oc, r.direction());
        auto c = oc.square().sum() - radius_*radius_;

        auto discriminant = half_b*half_b - a*c;
        if (discriminant.item<float>() < 0.0) return false;
        auto sqrtd = sqrt(discriminant);

        // Find the nearest root that lies in the acceptable range.
        auto root = ((-half_b - sqrtd) / a).item<float>();
        if (root < t_min || t_max < root) {
            root = ((-half_b + sqrtd) / a).item<float>();
            if (root < t_min || t_max < root)
                return false;
        }

        hit.t = root;
        hit.point = r.At(hit.t);
        auto outward_normal = (hit.point - center_) / radius_;
        hit.SetFaceNormal(r, outward_normal);
        return true;
    }

}