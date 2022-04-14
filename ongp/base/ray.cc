#include "ongp/base/ray.h"

namespace ongp
{
    Ray::Ray(const torch::Tensor &origin, const torch::Tensor &direction)
    :origin_(origin), direction_(direction)
    {
    }
}
