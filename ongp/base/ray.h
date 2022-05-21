#pragma once

#include <torch/torch.h>
#include "ongp/base/macros.h"
#include "ongp/base/tensor.h"

namespace ongp
{
    struct RayHit
    {
        torch::Tensor point;
        torch::Tensor normal;
        double t;
        bool front_face;

        inline void SetFaceNormal(const ray& r, const torch::Tensor& outward_normal) {
        front_face = torch::dot(r.direction(), outward_normal) < 0;
        normal = front_face ? outward_normal :-outward_normal;
        }
    };

    class Ray
    {
    public:
        Ray() = default;
        Ray(const torch::Tensor &origin, const torch::Tensor &direction);

        SET_GET_MEMBER_FUNC(torch::Tensor, origin)
        SET_GET_MEMBER_FUNC(torch::Tensor, direction)

        torch::Tensor At(double t) const; 

    protected:
        torch::Tensor origin_ = Array1dToTensor<float>({0,0,0});
        torch::Tensor direction_ = Array1dToTensor<float>({0,0,1});
    };

    class RaySegment: public Ray
    {
    public:
        RaySegment() = default;
        RaySegment(const torch::Tensor &origin, const torch::Tensor &direction, float near, float far);

        SET_GET_MEMBER_FUNC(float, near)
        SET_GET_MEMBER_FUNC(float, far)

    protected:
        float near_ = 0.1;
        float far_ = 1;
    };
}
