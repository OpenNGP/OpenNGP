#pragma one

#include <torch/torch.h>

namespace ongp
{
    color ray_color(const ray& r, const Object& object) {
        hit_record rec;
        if (world.hit(r, 0, infinity, rec)) {
            return 0.5 * (rec.normal + color(1,1,1));
        }
        vec3 unit_direction = unit_vector(r.direction());
        auto t = 0.5*(unit_direction.y() + 1.0);
        return (1.0-t)*color(1.0, 1.0, 1.0) + t*color(0.5, 0.7, 1.0);
}
}

