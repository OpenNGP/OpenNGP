#include "ongp/base/frame.h"

namespace ongp
{
    Frame::Frame(const Camera& cam, const Image& img):
    cam_(cam), img_(img)
    {}
}