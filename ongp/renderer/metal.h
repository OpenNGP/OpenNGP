#pragma once

#include "ongp/renderer/material.h"

namespace ongp
{
class Metal : public Material {
    public:
        Metal(const torch::Tensor& a, double f) : albedo(a), fuzz(f < 1 ? f : 1){}

        virtual bool Scatter(
            const Ray& r_in, const RayHit& hit, torch::Tensor& attenuation, Ray& scattered
        ) const override {
            auto reflected = reflect(unit_vector(r_in.direction()), hit.normal);
            scattered = Ray(hit.point, unit_vector(reflected+ fuzz*random_in_sphere()));
            attenuation = albedo;
            return (torch::dot(scattered.direction(), hit.normal).item<float>() > 0);
        }

    public:
        torch::Tensor albedo;
        double fuzz;
};

}

