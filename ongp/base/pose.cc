#include "ongp/base/pose.h"

namespace ongp
{
    Pose::Pose(): mat44_(torch::eye(4))
    {
    }

    Pose::Pose(const torch::Tensor &mat44): mat44_(mat44)
    {
    }
}

