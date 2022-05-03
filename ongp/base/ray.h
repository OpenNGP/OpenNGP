#pragma once

#include <torch/torch.h>
#include "ongp/base/macros.h"
#include "ongp/base/tensor.h"

namespace ongp
{
    class Ray
    {
    public:
        Ray() = default;
        Ray(const torch::Tensor &origin, const torch::Tensor &direction);

        SET_GET_MEMBER_FUNC(torch::Tensor, origin)
        SET_GET_MEMBER_FUNC(torch::Tensor, direction)
        SET_GET_MEMBER_FUNC(float, near)
        SET_GET_MEMBER_FUNC(float, far)

    protected:
        torch::Tensor origin_ = Array1dToTensor<float>({0,0,0});
        torch::Tensor direction_ = Array1dToTensor<float>({0,0,1});
        float near_ = 0.1;
        float far_ = 1;
    };
}
