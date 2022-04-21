#pragma one

#define VARIABLE_NAME(Variable) (#Variable)

#include <torch/torch.h>

namespace ongp
{
    class Primitive
    {
    public:
        virtual torch::Tensor QuerySigma(const torch::Tensor& xyzs) = 0;
        virtual torch::Tensor QueryColor(const torch::Tensor& geo_features, const torch::Tensor& views) = 0;
    };
}

