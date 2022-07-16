#pragma one

#include "ongp/base/pose.h"
#include "ongp/base/ray.h"
#include "ongp/base/macros.h"

namespace ongp
{
    class Intrinsics
    {
    public:
        void SetFromFov(double fov, double aspect_ratio, int height);
        void SetFromKMat(double fx, double fy, double cx, double cy);

        torch::Tensor Get() const;

    protected:
        torch::Tensor k_mat_;
    };

    class Camera
    {
    public:
        Camera() = default;
        explicit Camera(const torch::Tensor &k_mat);
        explicit Camera(const Intrinsics& intrs);
        Camera(const torch::Tensor &k_mat, const Pose &pose);
        Camera(const torch::Tensor &k_mat, const torch::Tensor &mat44);
        Camera(const Intrinsics& intrs, const Pose &pose);

        SET_GET_MEMBER_FUNC(Pose, pose)
        SET_GET_MEMBER_FUNC(torch::Tensor, k_mat)

        Ray GenerateRay(int r, int c);
        Ray GenerateRay(const torch::Tensor &r, const torch::Tensor &c);

    protected:
        Pose pose_;
        torch::Tensor k_mat_;
        Intrinsics intrs_;
    };
}

