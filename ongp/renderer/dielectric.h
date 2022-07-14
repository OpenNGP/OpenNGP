#pragma once

#include "ongp/renderer/material.h"

namespace ongp
{
class Dielectric : public Material {
    public:
        Dielectric(double index_of_refraction) : ir(index_of_refraction) {}

        virtual bool Scatter(
            const Ray& r_in, const RayHit& hit, torch::Tensor& attenuation, Ray& scattered
        ) const override {
            attenuation = Array1dToTensor<float>({1.0, 1.0, 1.0});
            double refraction_ratio = hit.front_face ? (1.0/ir) : ir;

            auto unit_direction = unit_vector(r_in.direction());

            double cos_theta = std::fmin(torch::dot(-unit_direction, hit.normal).item<float>(), 1.0);
            double sin_theta = sqrt(1.0 - cos_theta*cos_theta);

            bool cannot_refract = refraction_ratio * sin_theta > 1.0;
            torch::Tensor direction;

            if (cannot_refract || reflectance(cos_theta, refraction_ratio) > random_float())
                direction = reflect(unit_direction, hit.normal);
            else
                direction = refract(unit_direction, hit.normal, refraction_ratio);

            scattered = Ray(hit.point, direction);

            return true;
        }

    public:
        double ir; // Index of Refraction

    private:
        static double reflectance(double cosine, double ref_idx) {
            // Use Schlick's approximation for reflectance.
            auto r0 = (1-ref_idx) / (1+ref_idx);
            r0 = r0*r0;
            return r0 + (1-r0)*pow((1 - cosine),5);
        }
};
}

