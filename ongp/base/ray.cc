#include "ongp/base/ray.h"
#include "ongp/base/tensor.h"

namespace ongp
{
    Ray::Ray(const torch::Tensor &origin, const torch::Tensor &direction)
    :origin_(origin), direction_(direction)
    {
    }

    torch::Tensor Ray::At(double t) const {
        return origin_ + t*direction_;
    }

    RaySegment::RaySegment(const torch::Tensor &origin, const torch::Tensor &direction, float near, float far)
    :Ray(origin, direction), near_(near), far_(far)
    {
    }

    torch::Tensor RaySegment::At(double t) const {
        if (t >= near_ && t <= far_)
            return origin_ + t*direction_;
        return torch::zeros({0,0,0});
    }
}
