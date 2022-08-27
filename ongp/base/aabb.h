#pragma once

#include <torch/torch.h>
#include "ongp/base/ray.h"
#include "ongp/base/object.h"

namespace ongp
{
    class AABB {
    public:
        AABB() {}
        AABB(const torch::Tensor& a, const torch::Tensor& b) { minimum = a; maximum = b;}

        torch::Tensor min() const {return minimum; }
        torch::Tensor max() const {return maximum; }

        bool Hit(const Ray& r, double t_min, double t_max) const {
            for (int a = 0; a < 3; a++) {
                auto t0 = torch::fmin((minimum[a] - r.origin()[a]) / r.direction()[a],
                               (maximum[a] - r.origin()[a]) / r.direction()[a]);
                auto t1 = torch::fmax((minimum[a] - r.origin()[a]) / r.direction()[a],
                               (maximum[a] - r.origin()[a]) / r.direction()[a]);
                t_min = torch::fmax(t0, Vector3({t_min})).item<float>();
                t_max = torch::fmin(t1, Vector3({t_max})).item<float>();
                if (t_max <= t_min)
                    return false;
            }
            return true;
        }

        static AABB surrounding_box(AABB box0, AABB box1) {
            torch::Tensor small = Vector3({fmin(box0.min()[0].item<float>(), box1.min()[0].item<float>()),
                        fmin(box0.min()[1].item<float>(), box1.min()[1].item<float>()),
                        fmin(box0.min()[2].item<float>(), box1.min()[2].item<float>())});

            torch::Tensor big = Vector3({fmax(box0.max()[0].item<float>(), box1.max()[0].item<float>()),
                        fmax(box0.max()[1].item<float>(), box1.max()[1].item<float>()),
                        fmax(box0.max()[2].item<float>(), box1.max()[2].item<float>())});

            return AABB(small, big);
        }

        torch::Tensor minimum;
        torch::Tensor maximum;
};

bool box_compare(const ObjectSptr a, const ObjectSptr b, int axis);

    bool box_x_compare (const ObjectSptr a, const ObjectSptr b);
    bool box_y_compare (const ObjectSptr a, const ObjectSptr b);
    bool box_z_compare (const ObjectSptr a, const ObjectSptr b);
}