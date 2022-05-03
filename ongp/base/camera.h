#pragma one

#include "ongp/base/pose.h"
#include "ongp/base/ray.h"
#include "ongp/base/macros.h"

namespace ongp
{
    class Camera
    {
    public:
        Camera() = default;
        explicit Camera(const torch::Tensor &k_mat);
        Camera(const torch::Tensor &k_mat, const Pose &pose);
        Camera(const torch::Tensor &k_mat, const torch::Tensor &mat44);

        SET_GET_MEMBER_FUNC(Pose, pose)
        SET_GET_MEMBER_FUNC(torch::Tensor, k_mat)

        Ray GenerateRay(int r, int c);
        Ray GenerateRay(const torch::Tensor &r, const torch::Tensor &c);

    protected:
        Pose pose_;
        torch::Tensor k_mat_;
    };
}

