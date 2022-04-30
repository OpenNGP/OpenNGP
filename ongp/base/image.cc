#include "ongp/base/image.h"

namespace ongp
{
    Image::Image(const torch::Tensor& data): data_(data)
    {}

    int Image::Width()
    {
        return data_.size(1);
    }

    int Image::Height()
    {
        return data_.size(0);
    }
}

