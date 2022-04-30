#ifndef ONGP_BASE_IMAGE_H_
#define ONGP_BASE_IMAGE_H_

#include <torch/torch.h>
#include "ongp/base/macros.h"

namespace ongp
{
    class Image
    {
    public:
        Image() = default;
        explicit Image(const torch::Tensor& data);

        SET_MEMBER_FUNC(torch::Tensor, data)
        GET_MEMBER_FUNC(torch::Tensor, data)

        int Width();
        int Height();

    protected:
        torch::Tensor data_;
    };
}

#endif // ONGP_BASE_IMAGE_H_
