#include "delog/delog.h"
#include "ongp/base/sphere.h"

namespace ongp
{
    Sphere::Sphere(torch::Tensor center, double r, std::shared_ptr<Material> mat_ptr)
    :center_(center), radius_(r), mat_ptr_(mat_ptr)
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
        //std::cout << "discriminant: " << discriminant << std::endl;
        auto sqrtd = torch::sqrt(discriminant);

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
        hit.mat_ptr = mat_ptr_;
        return true;
    }

    bool Sphere::OnSurface(const torch::Tensor& point) const
    {
        double dist = (point - center_).norm().item<float>();
        return fabs(dist - radius_) < 1e-3;
    }

    bool Sphere::BoundingBox(AABB& output_box) const {
    output_box = AABB(
        center_ - Vector3({radius_, radius_, radius_}),
        center_ + Vector3({radius_, radius_, radius_}));
    return true;
}

}