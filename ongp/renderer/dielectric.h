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
            auto refracted = refract(unit_direction, hit.normal, refraction_ratio);

            scattered = Ray(hit.point, refracted);
            return true;
        }

    public:
        double ir; // Index of Refraction
};
}

