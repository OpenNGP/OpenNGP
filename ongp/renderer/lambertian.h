#pragma once

#include "ongp/renderer/material.h"

namespace ongp
{
class Lambertian : public Material {
public:
    Lambertian(const torch::Tensor& a) : albedo(a) {}

    virtual bool Scatter(
        const Ray& r_in, const RayHit& hit, torch::Tensor& attenuation, Ray& scattered
    ) const override {
        auto scatter_direction = hit.normal + random_unit_vector();
        // Catch degenerate scatter direction
        if (near_zero(scatter_direction)) scatter_direction = hit.normal;
        scattered = Ray(hit.point, scatter_direction);
        attenuation = albedo;
        return true;
    }

public:
    torch::Tensor albedo;
};

}

