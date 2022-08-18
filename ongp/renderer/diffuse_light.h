
#pragma once

#include "ongp/renderer/material.h"
#include "ongp/renderer/texture.h"

namespace ongp
{
class DiffuseLight : public Material {
    public:
        DiffuseLight(std::shared_ptr<Texture> a) : emit(a) {}
        DiffuseLight(torch::Tensor c) : emit(std::make_shared<SolidColor>(c)) {}

        virtual bool Scatter(
            const Ray& r_in, const RayHit& rec, torch::Tensor& attenuation, Ray& scattered
        ) const override {
            return false;
        }

        virtual torch::Tensor Emitted(double u, double v, const torch::Tensor& p) const override {
            return emit->Value(u, v, p);
        }

    public:
        std::shared_ptr<Texture> emit;
};
}