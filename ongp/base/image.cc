#include "ongp/base/image.h"

namespace ongp
{
    Image::Image(const torch::Tensor& data): data_(data)
    {}
}

