#ifndef ONGP_BASE_TENSOR_H_
#define ONGP_BASE_TENSOR_H_

#include <torch/torch.h>

namespace ongp
{
    template <class T>
    using Array1d = std::vector<T>;

    template <class T>
    using Array2d = std::vector<std::vector<T>>;

    template <class T>
    using Array3d = std::vector<std::vector<std::vector<T>>>;

    template <class T>
    std::vector<T> Linearize2d(const std::vector<std::vector<T>>& array_2d) {
        std::vector<T> array;
        for (const auto& v : array_2d) {
            for (auto d : v) {
                array.push_back(d);
            }
        }
        return array;
    }

    template <class T>
    torch::Tensor Array2dToTensor(const Array2d<T>& array_2d)
    {
        int m = array_2d.size();
        int n = array_2d[0].size();

        // torch::from_blob do not copy original data buffer!
        return torch::from_blob(Linearize2d(array_2d).data(), {m,n}).clone();
    }
}

#endif // ONGP_BASE_TENSOR_H_
