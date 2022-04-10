#ifndef ONGP_BASE_CAMERA_H_
#define ONGP_BASE_CAMERA_H_

#include "ongp/base/pose.h"
#include "ongp/base/macros.h"

namespace ongp
{
    class Camera
    {
    public:
        Camera() = default;
        explicit Camera(const Pose &pose);

        SET_MEMBER_FUNC(Pose, pose)
        GET_MEMBER_FUNC(Pose, pose)


    protected:
        Pose pose_;
    };
}

#endif // ONGP_BASE_CAMERA_H_
