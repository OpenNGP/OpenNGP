
#pragma one

#include <torch/torch.h>
#include "ongp/base/tensor.h"

namespace ongp
{
class Texture {
    public:
        virtual torch::Tensor Value(double u, double v, const torch::Tensor& p) const = 0;
};

class SolidColor : public Texture {
    public:
        SolidColor() {}
        SolidColor(torch::Tensor c) : color_value_(c) {}

        SolidColor(float red, float green, float blue)
          : SolidColor(Vector3<float>({red,green,blue})) {}

        virtual torch::Tensor Value(double u, double v, const torch::Tensor& p) const override {
            return color_value_;
        }

    private:
        torch::Tensor color_value_;
};

class CheckerTexture : public Texture {
    public:
        CheckerTexture() {}

        CheckerTexture(std::shared_ptr<Texture> _even, std::shared_ptr<Texture> _odd)
            : even(_even), odd(_odd) {}

        CheckerTexture(torch::Tensor c1, torch::Tensor c2)
            : even(std::make_shared<SolidColor>(c1)) , odd(std::make_shared<SolidColor>(c2)) {}

        virtual torch::Tensor Value(double u, double v, const torch::Tensor& p) const override {
            auto sines = sin(10*p.x())*sin(10*p.y())*sin(10*p.z());
            if (sines < 0)
                return odd->Value(u, v, p);
            else
                return even->Value(u, v, p);
        }

    public:
        std::shared_ptr<Texture> odd;
        std::shared_ptr<Texture> even;
};

}