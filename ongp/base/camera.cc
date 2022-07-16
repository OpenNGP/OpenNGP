#include "ongp/base/camera.h"
#include "ongp/base/tensor.h"
#include "delog/delog.h"

namespace ongp
{
    void Intrinsics::SetFromFov(double fov, double aspect_ratio, int height)
    {
        double f = (height/2)/tan(fov/2); // focus length
        double fx = f;
        double fy = f;
        double cx = aspect_ratio * height / 2;
        double cy = height / 2;
       // DELOG(fx);
       // DELOG(cx);
       // PAUSE();
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

    Camera::Camera(const Intrinsics& intrs, const Pose &pose): pose_(pose), k_mat_(intrs.Get())
    {
      //  std::cout << pose_.mat44() << std::endl;
      //  PAUSE();
    }

    Ray Camera::GenerateRay(int r, int c)
    {
        // Camera Coordinate
        float x = (static_cast<float>(r) - k_mat_.index({0,2}).item<float>()) / k_mat_.index({0,0}).item<float>();
        float y = (static_cast<float>(c) - k_mat_.index({1,2}).item<float>()) / k_mat_.index({1,1}).item<float>();
        float l = sqrt(x * x + y * y + 1);
        torch::Tensor Pc = Array1dToTensor<float>({x/l,y/l,1/l,1});
//        DELOG(x);
//        DELOG(y);
   //     std::cout << k_mat_ << std::endl;
   //     std::cout << Pc << std::endl;
   //     PAUSE();
        torch::Tensor Oc = Array1dToTensor<float>({0,0,0,1});

        // World Coordinate
        auto Pw = torch::matmul(pose_.mat44().inverse(), Pc);
        auto Ow = torch::matmul(pose_.mat44().inverse(), Oc);
        auto PO = Pw - Ow;

    //    std::cout << Pw << std::endl;
    //    std::cout << Ow << std::endl;
    //    std::cout << PO << std::endl;
    //    PAUSE();


//        torch::Tensor Pw = pose_.mat44().inverse() * Pc;
//        torch::Tensor Ow = pose_.mat44().inverse() * Oc;
        using namespace torch::indexing;
        return Ray(Ow.index({Slice(0, 3)}), unit_vector(PO.index({Slice(0, 3)})));
    }

    Ray Camera::GenerateRay(const torch::Tensor &r, const torch::Tensor &c)
    {
        return GenerateRay(r.item<int>(), c.item<int>());
    }
}

