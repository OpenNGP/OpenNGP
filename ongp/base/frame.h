#ifndef ONGP_BASE_FRAME_H_
#define ONGP_BASE_FRAME_H_

#include "ongp/base/camera.h"
#include "ongp/base/image.h"
#include "ongp/base/macros.h"

namespace ongp
{
    class Frame
    {
    public:
        Frame() = default;
        Frame(const Camera& cam, const Image& img);

        SET_MEMBER_FUNC(Camera, cam)
        GET_MEMBER_FUNC(Image, img)

    protected:
        Camera cam_;
        Image img_;
    };
}

#endif // ONGP_BASE_FRAME_H_
