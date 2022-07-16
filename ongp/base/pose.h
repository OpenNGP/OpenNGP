
#pragma once

#include <torch/torch.h>
#include "ongp/base/macros.h"
#include "ongp/base/tensor.h"

namespace ongp
{
    class Pose
    {
    public:
        Pose();
        explicit Pose(const torch::Tensor &mat44);
        Pose(torch::Tensor eye, torch::Tensor look_at, torch::Tensor up);


        SET_MEMBER_FUNC(torch::Tensor, mat44)
        GET_MEMBER_FUNC(torch::Tensor, mat44)

    protected:
        torch::Tensor mat44_ = torch::eye(4);
    };
}

