#ifndef ONGP_BASE_RAY_H_
#define ONGP_BASE_RAY_H_

#include <torch/torch.h>
#include "ongp/base/macros.h"

namespace ongp
{
    class Ray
    {
    public:
        Ray(const torch::Tensor &origin, const torch::Tensor &direction);

        SET_GET_MEMBER_FUNC(torch::Tensor, origin)
        SET_GET_MEMBER_FUNC(torch::Tensor, direction)
        SET_GET_MEMBER_FUNC(float, near)
        SET_GET_MEMBER_FUNC(float, far)

    protected:
        torch::Tensor origin_;
        torch::Tensor direction_;
        float near_;
        float far_;
    };
}

#endif // ONGP_BASE_RAY_H_
