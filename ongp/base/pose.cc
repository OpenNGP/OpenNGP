#include "ongp/base/pose.h"

namespace ongp
{
    Pose::Pose(): mat44_(torch::eye(4))
    {
    }

    Pose::Pose(const torch::Tensor &mat44): mat44_(mat44)
    {
    }

    Pose::Pose(torch::Tensor eye, torch::Tensor look_at, torch::Tensor up)
    {
        auto z = unit_vector(look_at - eye);
        auto x = unit_vector(cross(up, z));
        auto y = cross(z, x);

        using namespace torch::indexing;
        mat44_.index_put_({Slice(0, 3), 0}, x);
        mat44_.index_put_({Slice(0, 3), 1}, y);
        mat44_.index_put_({Slice(0, 3), 2}, z);
        mat44_.index_put_({Slice(0, 3), 3}, eye);

        mat44_ = mat44_.inverse(); // Mwc -> Mcw
    }
}

