#pragma one

#include <torch/torch.h>

#include "ongp/primitive/primitive.h"
#include "ongp/primitive/component/appearance/color.h"
#include "ongp/primitive/component/geometry/sigma.h"

namespace ongp
{
    class BaseNeRF: public Primitive
    {
    public:
        torch::Tensor QuerySigma(const torch::Tensor& xyzs) override
        {
            return sigma_->forward(xyzs);
        }
        torch::Tensor QueryColor(const torch::Tensor& geo_features, const torch::Tensor& views) override
        {
            return color_->forward(geo_features, views);
        }
    protected:
        Color color_;
        Sigma sigma_;
        
    };
}

