
#pragma one

#include <torch/torch.h>
#include "ongp/base/tensor.h"
#include "ongp/external/stb_image.h"

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


class ImageTexture : public Texture {
    public:
        const static int bytes_per_pixel = 3;

        ImageTexture()
          : data(nullptr), width(0), height(0), bytes_per_scanline(0) {}

        ImageTexture(const char* filename) {
            auto components_per_pixel = bytes_per_pixel;

            data = stbi_load(
                filename, &width, &height, &components_per_pixel, components_per_pixel);

            if (!data) {
                std::cerr << "ERROR: Could not load texture image file '" << filename << "'.\n";
                width = height = 0;
            }

            bytes_per_scanline = bytes_per_pixel * width;
        }

        ~ImageTexture() {
            delete data;
        }

        virtual torch::Tensor Value(double u, double v, const torch::Tensor& p) const override {
            // If we have no texture data, then return solid cyan as a debugging aid.
            if (data == nullptr)
                return Vector3({0,1,1});

            // Clamp input texture coordinates to [0,1] x [1,0]
            u = clamp(u, 0.0, 1.0);
            v = 1.0 - clamp(v, 0.0, 1.0);  // Flip V to image coordinates

            auto i = static_cast<int>(u * width);
            auto j = static_cast<int>(v * height);

            // Clamp integer mapping, since actual coordinates should be less than 1.0
            if (i >= width)  i = width-1;
            if (j >= height) j = height-1;

            const auto color_scale = 1.0 / 255.0;
            auto pixel = data + j*bytes_per_scanline + i*bytes_per_pixel;

            return Vector3({color_scale*pixel[0], color_scale*pixel[1], color_scale*pixel[2]});
        }

    private:
        unsigned char *data;
        int width, height;
        int bytes_per_scanline;
};

}