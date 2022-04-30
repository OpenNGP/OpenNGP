#include "ongp/base/camera.h"

namespace ongp
{
    Camera::Camera(const torch::Tensor &k_mat): k_mat_(k_mat)
    {}

    Camera::Camera(const torch::Tensor &k_mat, const Pose &pose): pose_(pose), k_mat_(k_mat)
    {}

    Camera::Camera(const torch::Tensor &k_mat, const torch::Tensor& mat44): pose_(mat44), k_mat_(k_mat)
    {}

    Ray Camera::GenerateRay(int r, int c)
    {
        return Ray();
    }

    Ray Camera::GenerateRay(const torch::Tensor &r, const torch::Tensor &c)
    {
        return Ray();
    }
}

