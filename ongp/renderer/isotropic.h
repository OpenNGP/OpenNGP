
#pragma once

#include "ongp/renderer/material.h"
#include "ongp/renderer/texture.h"

namespace ongp
{
class Isotropic : public Material {
    public:
        Isotropic(torch::Tensor c) : albedo(std::make_shared<SolidColor>(c)) {}
        Isotropic(std::shared_ptr<Texture> a) : albedo(a) {}

        virtual bool Scatter(
            const Ray& r_in, const RayHit& rec, torch::Tensor& attenuation, Ray& scattered
        ) const override {
            scattered = Ray(rec.point, random_in_sphere());
            attenuation = albedo->Value(rec.u, rec.v, rec.point);
            return true;
        }

    public:
        std::shared_ptr<Texture> albedo;
};
}