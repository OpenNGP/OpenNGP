#include "ongp/base/camera.h"
#include "ongp/base/tensor.h"

namespace ongp
{
    void Intrinsics::SetFromFov(double fov, double aspect_ratio, int height)
    {
        double f = (height/2)/tan(fov/2); // focus length
        double fx = f;
        double fy = f;
        double cx = aspect_ratio * height / 2;
        double cy = height / 2;
        SetFromKMat(fx, fy, cx, cy);
    }

    void Intrinsics::SetFromKMat(double fx, double fy, double cx, double cy)
    {
        k_mat_ = torch::zeros({3,3});
        k_mat_.index({0,0}) = fx;
        k_mat_.index({1,1}) = fy;
        k_mat_.index({0,2}) = cx;
        k_mat_.index({1,2}) = cy;
        k_mat_.index({2,2}) = 1;
    }


    torch::Tensor Intrinsics::Get() const
    {
        return k_mat_;
    }

    Camera::Camera(const Intrinsics& intrs): k_mat_(intrs.Get())
    {}

    Camera::Camera(const torch::Tensor &k_mat): k_mat_(k_mat)
    {}

    Camera::Camera(const torch::Tensor &k_mat, const Pose &pose): pose_(pose), k_mat_(k_mat)
    {}

    Camera::Camera(const torch::Tensor &k_mat, const torch::Tensor& mat44): pose_(mat44), k_mat_(k_mat)
    {}

    Ray Camera::GenerateRay(int r, int c)
    {
        // Camera Coordinate
        float x = static_cast<float>(r) - k_mat_.index({0,2}).item<float>() / k_mat_.index({0,0}).item<float>();
        float y = static_cast<float>(c) - k_mat_.index({1,2}).item<float>() / k_mat_.index({1,1}).item<float>();
        torch::Tensor Pc = Array1dToTensor<float>({x,y,1});
        torch::Tensor Oc = Array1dToTensor<float>({0,0,0});

        // World Coordinate
        torch::Tensor Pw = pose_.mat44().inverse() * Pc;
        torch::Tensor Ow = pose_.mat44().inverse() * Oc;

        return Ray(Ow, Pw);
    }

    Ray Camera::GenerateRay(const torch::Tensor &r, const torch::Tensor &c)
    {
        return GenerateRay(r.item<int>(), c.item<int>());
    }
}

