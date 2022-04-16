#pragma one

#include <torch/torch.h>

#include "ongp/primitive/component/module/encoder/frequency.h"
#include "ongp/primitive/component/module/regressor/mlp.h"

namespace ongp
{
    struct ColorOptions{
    ColorOptions(int64_t n_emb, int64_t n_geo_emb, int64_t w)
    : w_(w), n_emb_(n_emb), n_geo_emb_(n_geo_emb)
    {}
        TORCH_ARG(int64_t, n_emb);
        TORCH_ARG(int64_t, n_geo_emb);
        TORCH_ARG(int64_t, w);
    };

    struct ColorImpl : torch::nn::Module {
    explicit ColorImpl(const ColorOptions& options): options(options)
    {
        encoder = Frequency(options.n_emb());
        regressor = MLP(MLPOptions(0, options.w(), 3*(1+2*options.n_emb())+options.n_geo_emb(), 3));
    }

    torch::Tensor forward(const torch::Tensor& geo_features, const torch::Tensor& views)
    {
        auto view_inputs = encoder->forward(views);
        auto color_inputs = torch::concat({geo_features, view_inputs}, -1);
        auto outputs = regressor->forward(color_inputs);
        return torch::sigmoid(outputs);
    }

    Frequency encoder;
    MLP regressor;

    ColorOptions options;
    };

    TORCH_MODULE(Color); 
}
