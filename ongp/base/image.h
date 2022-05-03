#pragma once

#include <torch/torch.h>
#include "ongp/base/macros.h"

namespace ongp
{
    class Image
    {
    public:
        Image() = default;
        explicit Image(const torch::Tensor& data);

        SET_MEMBER_FUNC(torch::Tensor, data)
        GET_MEMBER_FUNC(torch::Tensor, data)

        int Width();
        int Height();

    protected:
        torch::Tensor data_;
    };
}
